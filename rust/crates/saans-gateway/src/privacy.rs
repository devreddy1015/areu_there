//! The gateway's consent gate (ADR-001, ADR-005), in one place so it can
//! be audited.
//!
//! Spaces with `vitals = false` must never expose breathing or heart data,
//! not even as zeros: every JSON value that leaves the gateway for such a
//! space (API, WebSocket, event log) is passed through [`redact_value`],
//! and events that only exist because of vitals are dropped by
//! [`event_allowed`]. FeatureFrames that arrive carrying vitals for a
//! space that has not opted in are zeroed on ingest ([`gate_frame`]) and
//! counted, so a misconfigured node shows up in `/api/health`.
//!
//! The key test is deliberately broad (any key mentioning breath, bpm,
//! heart or pause), so a field added to `SpaceState` or an event later is
//! withheld by default rather than leaked by default.

use crate::roles::Principal;
use saans_dsp::features::{
    FeatureFrame, FLAG_BREATHING_VALID, FLAG_HEART_VALID, FLAG_VITALS_ENABLED,
};
use serde_json::Value;

const VITAL_WORDS: [&str; 4] = ["breath", "bpm", "heart", "pause"];

fn mentions_vitals(s: &str) -> bool {
    let s = s.to_ascii_lowercase();
    VITAL_WORDS.iter().any(|w| s.contains(w))
}

/// Remove every vitals-derived key, recursively, when vitals are off.
pub fn redact_value(v: &mut Value, vitals: bool) {
    if vitals {
        return;
    }
    match v {
        Value::Object(m) => {
            m.retain(|k, _| !mentions_vitals(k));
            m.values_mut().for_each(|x| redact_value(x, false));
        }
        Value::Array(a) => a.iter_mut().for_each(|x| redact_value(x, false)),
        _ => {}
    }
}

/// Whether an event (serialized, with its `type` tag) may be shown for a
/// space with the given vitals setting.
pub fn event_allowed(event: &Value, vitals: bool) -> bool {
    if vitals {
        return true;
    }
    let ty = event.get("type").and_then(Value::as_str).unwrap_or("");
    if mentions_vitals(ty) || ty.starts_with("rate_") {
        return false;
    }
    if ty == "trend_deviation" {
        let metric = event.get("metric").and_then(Value::as_str).unwrap_or("");
        if mentions_vitals(metric) || metric.contains("rate") || metric.contains("resp") {
            return false;
        }
    }
    true
}

/// Whether an active-alert name (`SpaceState::active_alerts`) may be shown.
pub fn alert_allowed(name: &str, vitals: bool) -> bool {
    vitals || !(mentions_vitals(name) || name.starts_with("rate_"))
}

/// True if a frame carries (or claims to carry) vitals.
pub fn frame_has_vitals(f: &FeatureFrame) -> bool {
    f.flags & (FLAG_VITALS_ENABLED | FLAG_BREATHING_VALID | FLAG_HEART_VALID) != 0
        || [
            f.breath_rms,
            f.breathing_bpm,
            f.breathing_conf,
            f.heart_bpm,
            f.heart_conf,
        ]
        .iter()
        .any(|&x| x != 0.0)
}

/// Zero the vitals fields and flags, as a node would have done.
pub fn gate_frame(mut f: FeatureFrame) -> FeatureFrame {
    f.flags &= !(FLAG_VITALS_ENABLED | FLAG_BREATHING_VALID | FLAG_HEART_VALID);
    f.breath_rms = 0.0;
    f.breathing_bpm = 0.0;
    f.breathing_conf = 0.0;
    f.heart_bpm = 0.0;
    f.heart_conf = 0.0;
    f
}

// ---------------------------------------------------------------------
// The second lock: who is asking (ADR-005, SAANS_PLAN §10).
//
// Consent decides whether a value may exist in the gateway's output at
// all; the functions above enforce that. These decide whether *this*
// caller may see what consent allowed, using the predicates on
// `Principal`. Both locks are applied, in that order, on every path out.

/// Alerts and events that exist only because the space is watched for a
/// person's wellbeing. `blind` is not one: "we cannot see this room" is a
/// coverage fact every role needs, including facilities.
fn is_health_alert(name: &str) -> bool {
    name != "blind"
}

/// Event types that describe the sensors rather than the occupants.
fn is_system_event(ty: &str) -> bool {
    matches!(
        ty,
        "link_down" | "link_up" | "link_degraded" | "calibration_stale" | "raw_mode_active"
    )
}

/// One published space as `principal` may see it, or `None` when the
/// space is outside their scope entirely.
///
/// Takes the already-consent-filtered value, so a space with
/// `vitals = false` has no vitals keys to remove by the time we get here;
/// the redaction below is the role lock, not a second consent check.
pub fn space_for(v: &Value, principal: &Principal) -> Option<Value> {
    let space_id = v.get("space_id").and_then(Value::as_str).unwrap_or("");
    if !principal.can_see_space(space_id) {
        return None;
    }
    let consented = v.get("vitals").and_then(Value::as_bool).unwrap_or(false);
    let mut out = v.clone();
    if !principal.can_see_vitals(space_id, consented) {
        redact_value(&mut out, false);
    }
    if !principal.can_see_health_alerts(space_id) {
        if let Value::Object(m) = &mut out {
            if let Some(Value::Array(a)) = m.get_mut("alerts") {
                a.retain(|x| !is_health_alert(x.get("name").and_then(Value::as_str).unwrap_or("")));
            }
            if let Some(Value::Array(a)) = m.get_mut("active_alerts") {
                a.retain(|x| !is_health_alert(x.as_str().unwrap_or("")));
            }
        }
    }
    Some(out)
}

/// Whether `principal` may see an event, and in what form. `None` means
/// withhold it; `Some` may be a redacted copy.
pub fn event_for(v: &Value, principal: &Principal) -> Option<Value> {
    let space_id = v.get("space_id").and_then(Value::as_str).unwrap_or("");
    let ty = v.get("type").and_then(Value::as_str).unwrap_or("");
    if is_system_event(ty) {
        // Sensor faults are not about the occupants, but they do name a
        // space, so scope still applies.
        if !(principal.can_see_system() && principal.can_see_space(space_id)) {
            return None;
        }
        return Some(v.clone());
    }
    if !principal.can_see_space(space_id) {
        return None;
    }
    // "Someone arrived", "the room went blind" — occupancy and coverage.
    let coverage = matches!(
        ty,
        "occupancy_started" | "occupancy_ended" | "space_blind" | "space_visible"
    );
    if !coverage && !principal.can_see_health_alerts(space_id) {
        return None;
    }
    let mut out = v.clone();
    // An event reaching here was allowed by consent; the role lock can
    // still require the vitals numbers inside it to go.
    if !matches!(
        principal.role,
        crate::roles::Role::Admin | crate::roles::Role::Safety
    ) {
        redact_value(&mut out, false);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roles::Role;
    use serde_json::json;

    #[test]
    fn vitals_keys_are_removed_recursively() {
        let mut v = json!({"occupied": true, "breathing": {"bpm": 14}, "bpm_median_5min": 14.0,
            "in_breathing_pause": false, "nested": [{"heart": 1, "motion_index": 0.3}], "still_for_s": 3});
        redact_value(&mut v, false);
        assert_eq!(
            v,
            json!({"occupied": true, "nested": [{"motion_index": 0.3}], "still_for_s": 3})
        );
        let mut w = json!({"breathing": null});
        redact_value(&mut w, true);
        assert_eq!(w, json!({"breathing": null}));
    }

    #[test]
    fn vitals_events_are_dropped_when_off() {
        for ty in ["breathing_pause", "rate_high", "rate_low"] {
            assert!(!event_allowed(&json!({"type": ty}), false));
            assert!(event_allowed(&json!({"type": ty}), true));
        }
        assert!(event_allowed(&json!({"type": "inactivity"}), false));
        assert!(!event_allowed(
            &json!({"type": "trend_deviation", "metric": "resting_bpm"}),
            false
        ));
        assert!(event_allowed(
            &json!({"type": "trend_deviation", "metric": "motion_bouts_per_h"}),
            false
        ));
        assert!(!alert_allowed("inactivity_no_breathing", false));
        assert!(!alert_allowed("rate_high", false));
        assert!(alert_allowed("inactivity", false) && alert_allowed("blind", false));
    }

    fn p(role: Role, spaces: Option<&[&str]>) -> Principal {
        Principal {
            name: "x".into(),
            role,
            spaces: spaces.map(|s| s.iter().map(|x| x.to_string()).collect()),
        }
    }

    fn consented_space() -> Value {
        json!({
            "space_id": "lab", "vitals": true, "occupied": true, "motion_index": 0.4,
            "breathing": {"bpm": 14.0, "confidence": 0.8}, "bpm_median_5min": 14.0,
            "active_alerts": ["blind", "inactivity"],
            "alerts": [{"name": "blind"}, {"name": "inactivity"}, {"name": "rate_high"}],
        })
    }

    #[test]
    fn a_space_outside_scope_is_not_returned_at_all() {
        assert!(space_for(&consented_space(), &p(Role::Safety, Some(&["other"]))).is_none());
        assert!(space_for(&consented_space(), &p(Role::Admin, Some(&["lab"]))).is_some());
    }

    #[test]
    fn ops_sees_the_room_but_never_the_person() {
        let v = space_for(&consented_space(), &p(Role::Ops, None)).expect("in scope");
        assert_eq!(v["occupied"], json!(true), "occupancy is not health data");
        assert_eq!(v["motion_index"], json!(0.4));
        assert!(v.get("breathing").is_none() && v.get("bpm_median_5min").is_none());
        // Only the coverage alert survives.
        assert_eq!(v["active_alerts"], json!(["blind"]));
        assert_eq!(v["alerts"], json!([{"name": "blind"}]));
    }

    #[test]
    fn safety_sees_vitals_only_where_the_space_consented() {
        let safety = p(Role::Safety, None);
        let v = space_for(&consented_space(), &safety).expect("in scope");
        assert_eq!(v["breathing"]["bpm"], json!(14.0));
        assert_eq!(v["active_alerts"], json!(["blind", "inactivity"]));
        // The same space without consent has already been stripped
        // upstream; the role lock must not put anything back.
        let mut no_consent = consented_space();
        no_consent["vitals"] = json!(false);
        redact_value(&mut no_consent, false);
        let w = space_for(&no_consent, &safety).expect("in scope");
        assert!(w.get("breathing").is_none());
        assert_eq!(w["occupied"], json!(true));
    }

    #[test]
    fn occupancy_role_gets_occupancy_and_coverage_only() {
        let v = space_for(&consented_space(), &p(Role::Occupancy, None)).expect("in scope");
        assert_eq!(v["occupied"], json!(true));
        assert!(v.get("breathing").is_none());
        assert_eq!(v["alerts"], json!([{"name": "blind"}]));
    }

    #[test]
    fn events_follow_the_same_two_locks() {
        let (admin, ops, occ) = (
            p(Role::Admin, None),
            p(Role::Ops, None),
            p(Role::Occupancy, None),
        );
        let pause = json!({"space_id": "lab", "type": "breathing_pause", "duration_s": 12.0});
        assert!(event_for(&pause, &admin).is_some());
        assert!(
            event_for(&pause, &ops).is_none(),
            "ops sees no health events"
        );
        assert!(event_for(&pause, &occ).is_none());

        let arrived = json!({"space_id": "lab", "type": "occupancy_started"});
        for who in [&admin, &ops, &occ] {
            assert!(
                event_for(&arrived, who).is_some(),
                "arrival is occupancy, not health"
            );
        }
        assert!(
            event_for(&arrived, &p(Role::Occupancy, Some(&["other"]))).is_none(),
            "out of scope"
        );

        let down = json!({"space_id": "lab", "type": "link_down", "link": {"rx": 1, "tx": 2}});
        assert!(event_for(&down, &ops).is_some() && event_for(&down, &admin).is_some());
        assert!(
            event_for(&down, &occ).is_none(),
            "a sensor fault is for ops"
        );

        // A health event an ops-like role may see at all still loses its
        // numbers: inactivity reaches safety, not ops.
        let inact = json!({"space_id": "lab", "type": "inactivity", "still_min": 31.0, "breathing_detected": true});
        assert!(event_for(&inact, &p(Role::Safety, None)).is_some());
        assert!(event_for(&inact, &ops).is_none());
    }

    #[test]
    fn gate_zeroes_vitals() {
        let f = FeatureFrame {
            flags: 0xFF,
            breathing_bpm: 14.0,
            breath_rms: 0.3,
            heart_conf: 0.5,
            ..Default::default()
        };
        assert!(frame_has_vitals(&f));
        let g = gate_frame(f);
        assert!(!frame_has_vitals(&g));
        assert_eq!(
            g.flags,
            !(FLAG_VITALS_ENABLED | FLAG_BREATHING_VALID | FLAG_HEART_VALID)
        );
    }
}
