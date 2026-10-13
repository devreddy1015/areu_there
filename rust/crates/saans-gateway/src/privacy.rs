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

use saans_dsp::features::{FeatureFrame, FLAG_BREATHING_VALID, FLAG_HEART_VALID, FLAG_VITALS_ENABLED};
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
        || [f.breath_rms, f.breathing_bpm, f.breathing_conf, f.heart_bpm, f.heart_conf].iter().any(|&x| x != 0.0)
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn vitals_keys_are_removed_recursively() {
        let mut v = json!({"occupied": true, "breathing": {"bpm": 14}, "bpm_median_5min": 14.0,
            "in_breathing_pause": false, "nested": [{"heart": 1, "motion_index": 0.3}], "still_for_s": 3});
        redact_value(&mut v, false);
        assert_eq!(v, json!({"occupied": true, "nested": [{"motion_index": 0.3}], "still_for_s": 3}));
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
        assert!(!event_allowed(&json!({"type": "trend_deviation", "metric": "resting_bpm"}), false));
        assert!(event_allowed(&json!({"type": "trend_deviation", "metric": "motion_bouts_per_h"}), false));
        assert!(!alert_allowed("inactivity_no_breathing", false));
        assert!(!alert_allowed("rate_high", false));
        assert!(alert_allowed("inactivity", false) && alert_allowed("blind", false));
    }

    #[test]
    fn gate_zeroes_vitals() {
        let f = FeatureFrame { flags: 0xFF, breathing_bpm: 14.0, breath_rms: 0.3, heart_conf: 0.5, ..Default::default() };
        assert!(frame_has_vitals(&f));
        let g = gate_frame(f);
        assert!(!frame_has_vitals(&g));
        assert_eq!(g.flags, !(FLAG_VITALS_ENABLED | FLAG_BREATHING_VALID | FLAG_HEART_VALID));
    }
}
