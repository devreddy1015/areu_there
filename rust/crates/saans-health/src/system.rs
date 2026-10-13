//! System health (SAANS_PLAN.md §2.2): the sensing system must say when
//! it cannot see. Watches each space's `LinkTrack`s and reports links that
//! went silent or came back, links too slow or lossy for vital signs, stale
//! calibrations, raw-CSI debug mode (a privacy notice, ADR-001) and spaces
//! with no usable link at all.
//!
//! Every event is rate-limited: once per episode, or once per stated
//! interval for conditions that persist by design (stale calibration, raw
//! mode).

use crate::fusion::{LinkTrack, SpaceFusion, MIN_RATE_VITALS_HZ};
use crate::types::{EventKind, HealthEvent, LinkId, Severity};
use saans_dsp::features::{FLAG_RAW_MODE, NOT_CALIBRATED};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// A link silent this long is down: ten missed 2 Hz frames, well past
/// fusion's 2.5 s freshness limit, so one late datagram is not an outage.
pub const LINK_DOWN_S: f64 = 5.0;
/// Below this achieved CSI rate breathing is not trustworthy (campus plan
/// constraint 4; the same floor fusion uses for vitals).
pub const DEGRADED_MIN_RATE_HZ: f64 = MIN_RATE_VITALS_HZ;
/// Sequence loss above this degrades gap filling and the breathing band.
pub const DEGRADED_MAX_LOSS: f64 = 0.10;
/// A link must be degraded this long before it is reported: the node's
/// rate and loss figures are 10 s averages and dip at startup.
pub const DEGRADED_SUSTAIN_S: f64 = 30.0;
/// ...and healthy this long before a new degradation episode can be
/// reported, so a link hovering at the limit does not alert every 30 s.
pub const DEGRADED_CLEAR_S: f64 = 30.0;
/// Calibration older than this is stale: furniture moves, doors and
/// HVAC change the empty-room level over a day.
pub const CALIB_STALE_H: f64 = 24.0;
/// Repeat a stale-calibration notice at most this often per link.
pub const CALIB_STALE_REPEAT_S: f64 = 6.0 * 3600.0;
/// Repeat the raw-mode privacy notice at most this often per link.
pub const RAW_MODE_REPEAT_S: f64 = 3600.0;
/// A space with no usable link for longer than this is reported blind.
pub const SPACE_BLIND_S: f64 = 10.0;

/// Per-link status for the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkReport {
    pub link: LinkId,
    /// A frame arrived within `LINK_DOWN_S`.
    pub up: bool,
    /// Seconds since the last frame (since monitoring began if none yet).
    pub silent_s: f64,
    pub rate_hz: f64,
    pub loss: f64,
    /// None until the node reports a completed calibration.
    pub calib_age_h: Option<f64>,
    pub best_bnr: f64,
    pub rssi_dbm: i8,
    pub noise_floor_dbm: i8,
    pub raw_mode: bool,
    pub frames: u64,
}

#[derive(Debug, Clone)]
struct LinkHealth {
    first_observed: f64,
    down_reported: bool,
    bad_since: Option<f64>,
    good_since: Option<f64>,
    degraded_reported: bool,
    last_stale_notice: Option<f64>,
    last_raw_notice: Option<f64>,
}

impl LinkHealth {
    fn new(t_s: f64) -> Self {
        Self {
            first_observed: t_s,
            down_reported: false,
            bad_since: None,
            good_since: None,
            degraded_reported: false,
            last_stale_notice: None,
            last_raw_notice: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct SpaceHealth {
    links: BTreeMap<LinkId, LinkHealth>,
    zero_since: Option<f64>,
    blind_reported: bool,
}

/// Seconds since the link's last frame, or since monitoring began.
fn silent_s(track: &LinkTrack, first_observed: Option<f64>, t_s: f64) -> f64 {
    if track.frames > 0 && track.last_seen_s.is_finite() {
        (t_s - track.last_seen_s).max(0.0)
    } else {
        first_observed.map_or(0.0, |f| (t_s - f).max(0.0))
    }
}

fn sys_event(space: &str, t_s: f64, t_start_s: Option<f64>, severity: Severity, evidence: String, kind: EventKind) -> HealthEvent {
    HealthEvent { space_id: space.to_string(), t_s, t_start_s, severity, confidence: 1.0, evidence, kind }
}

/// Tracks link and coverage health for any number of spaces.
#[derive(Debug, Clone, Default)]
pub struct SystemMonitor {
    spaces: HashMap<String, SpaceHealth>,
}

impl SystemMonitor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Call at 2 Hz per space with that space's fusion; returns
    /// system-health events.
    pub fn observe(&mut self, space_id: &str, fusion: &SpaceFusion, t_s: f64) -> Vec<HealthEvent> {
        let mut ev = Vec::new();
        if !self.spaces.contains_key(space_id) {
            self.spaces.insert(space_id.to_string(), SpaceHealth::default());
        }
        let sp = self.spaces.get_mut(space_id).expect("inserted above");

        for (&id, track) in &fusion.links {
            let lh = sp.links.entry(id).or_insert_with(|| LinkHealth::new(t_s));
            let silent = silent_s(track, Some(lh.first_observed), t_s);
            let up = track.frames > 0 && silent <= LINK_DOWN_S;

            if !up && silent > LINK_DOWN_S && !lh.down_reported {
                lh.down_reported = true;
                let evidence = if track.frames > 0 {
                    format!("no frames from link {id} for {silent:.1} s (last after {} frames)", track.frames)
                } else {
                    format!("no frames from link {id} since monitoring began {silent:.1} s ago")
                };
                let start = if track.frames > 0 { track.last_seen_s } else { lh.first_observed };
                ev.push(sys_event(space_id, t_s, Some(start), Severity::Advisory, evidence, EventKind::LinkDown {
                    link: id,
                    silent_s: silent,
                }));
            }
            if up && lh.down_reported {
                lh.down_reported = false;
                ev.push(sys_event(
                    space_id,
                    t_s,
                    None,
                    Severity::Info,
                    format!("link {id} is sending frames again ({:.1} Hz CSI)", track.rate_hz()),
                    EventKind::LinkUp { link: id },
                ));
            }
            if !up {
                lh.bad_since = None;
                lh.good_since = None;
                continue;
            }
            let Some(frame) = track.last else { continue };

            // Rate / loss degradation, sustained.
            let (rate, loss) = (track.rate_hz(), track.loss());
            let mut reasons = Vec::new();
            if rate < DEGRADED_MIN_RATE_HZ {
                reasons.push(format!("CSI rate {rate:.1} Hz below the {DEGRADED_MIN_RATE_HZ:.0} Hz breathing needs"));
            }
            if loss > DEGRADED_MAX_LOSS {
                reasons.push(format!(
                    "sequence loss {:.0}% above {:.0}%",
                    loss * 100.0,
                    DEGRADED_MAX_LOSS * 100.0
                ));
            }
            if reasons.is_empty() {
                lh.bad_since = None;
                if lh.degraded_reported {
                    let g = *lh.good_since.get_or_insert(t_s);
                    if t_s - g >= DEGRADED_CLEAR_S {
                        lh.degraded_reported = false;
                        lh.good_since = None;
                    }
                }
            } else {
                lh.good_since = None;
                let since = *lh.bad_since.get_or_insert(t_s);
                if !lh.degraded_reported && t_s - since >= DEGRADED_SUSTAIN_S {
                    lh.degraded_reported = true;
                    let reason = reasons.join("; ");
                    let evidence = format!(
                        "link {id}: {reason}, for {:.0} s; its breathing estimates are not used while this lasts",
                        t_s - since
                    );
                    ev.push(sys_event(space_id, t_s, Some(since), Severity::Advisory, evidence, EventKind::LinkDegraded {
                        link: id,
                        reason,
                    }));
                }
            }

            // Stale calibration.
            if frame.calib_age_s != NOT_CALIBRATED {
                let age_h = frame.calib_age_s as f64 / 3600.0;
                if age_h > CALIB_STALE_H && !lh.last_stale_notice.is_some_and(|l| t_s - l < CALIB_STALE_REPEAT_S) {
                    lh.last_stale_notice = Some(t_s);
                    let evidence = format!(
                        "link {id}'s empty-room calibration is {age_h:.1} h old (limit {CALIB_STALE_H:.0} h); \
                         recalibrate while the space is empty"
                    );
                    ev.push(sys_event(space_id, t_s, None, Severity::Advisory, evidence, EventKind::CalibrationStale {
                        link: id,
                        age_h,
                    }));
                }
            }

            // Raw-CSI debug mode: a privacy notice (ADR-001).
            if frame.flag(FLAG_RAW_MODE) && !lh.last_raw_notice.is_some_and(|l| t_s - l < RAW_MODE_REPEAT_S) {
                lh.last_raw_notice = Some(t_s);
                let evidence = format!(
                    "node {} is also streaming raw CSI on link {id} (debug mode); raw channel data is leaving the \
                     sensor",
                    id.rx
                );
                ev.push(sys_event(space_id, t_s, None, Severity::Advisory, evidence, EventKind::RawModeActive { link: id }));
            }
        }

        // Coverage: can the space be seen at all?
        let total = fusion.links.len();
        let usable = fusion.links.values().filter(|l| l.usable(t_s)).count();
        if usable == 0 {
            let since = *sp.zero_since.get_or_insert(t_s);
            if !sp.blind_reported && t_s - since > SPACE_BLIND_S {
                sp.blind_reported = true;
                let evidence = format!(
                    "none of the space's {total} links has been usable (fresh, calibrated, >= 5 Hz) for {:.1} s; \
                     its health outputs are unknown, not empty",
                    t_s - since
                );
                ev.push(sys_event(space_id, t_s, Some(since), Severity::Advisory, evidence, EventKind::SpaceBlind));
            }
        } else {
            sp.zero_since = None;
            if sp.blind_reported {
                sp.blind_reported = false;
                ev.push(sys_event(
                    space_id,
                    t_s,
                    None,
                    Severity::Info,
                    format!("{usable} of {total} links usable again"),
                    EventKind::SpaceVisible,
                ));
            }
        }
        ev
    }

    /// Per-link status of one space, for the API.
    pub fn link_report(&self, fusion: &SpaceFusion, t_s: f64) -> Vec<LinkReport> {
        fusion
            .links
            .iter()
            .map(|(&id, track)| {
                let first = self
                    .spaces
                    .get(&fusion.config.id)
                    .and_then(|s| s.links.get(&id))
                    .or_else(|| self.spaces.values().find_map(|s| s.links.get(&id)))
                    .map(|l| l.first_observed);
                let silent = silent_s(track, first, t_s);
                let f = track.last;
                LinkReport {
                    link: id,
                    up: track.frames > 0 && silent <= LINK_DOWN_S,
                    silent_s: silent,
                    rate_hz: track.rate_hz(),
                    loss: track.loss(),
                    calib_age_h: f.filter(|f| f.calib_age_s != NOT_CALIBRATED).map(|f| f.calib_age_s as f64 / 3600.0),
                    best_bnr: f.map_or(0.0, |f| f.best_bnr as f64),
                    rssi_dbm: f.map_or(0, |f| f.rssi_dbm),
                    noise_floor_dbm: f.map_or(saans_dsp::features::NOISE_UNKNOWN, |f| f.noise_floor_dbm),
                    raw_mode: f.is_some_and(|f| f.flag(FLAG_RAW_MODE)),
                    frames: track.frames,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SpaceConfig;
    use saans_dsp::features::{FeatureFrame, FLAG_CALIBRATED, FLAG_OCCUPIED};

    const A: LinkId = LinkId { rx: 1, tx: 9 };
    const B: LinkId = LinkId { rx: 2, tx: 9 };

    fn frame(rate_hz: f64, loss: f64, flags: u8) -> FeatureFrame {
        FeatureFrame {
            flags: FLAG_CALIBRATED | FLAG_OCCUPIED | flags,
            csi_rate_chz: (rate_hz * 100.0) as u16,
            loss_permille: (loss * 1000.0) as u16,
            presence_threshold: 0.05,
            motion_energy: 0.1,
            calib_age_s: 3600,
            best_bnr: 4.0,
            rssi_dbm: -50,
            noise_floor_dbm: -92,
            ..Default::default()
        }
    }

    fn fusion(links: &[LinkId]) -> SpaceFusion {
        SpaceFusion::new(SpaceConfig { id: "lab".into(), links: links.to_vec(), ..Default::default() })
    }

    /// Run at 2 Hz over [t0, t1): `send(t)` lists the frames to ingest.
    fn drive(
        sys: &mut SystemMonitor,
        fu: &mut SpaceFusion,
        t0: f64,
        t1: f64,
        send: impl Fn(f64) -> Vec<(LinkId, FeatureFrame)>,
    ) -> Vec<HealthEvent> {
        let n = ((t1 - t0) / 0.5).round() as usize;
        let mut ev = Vec::new();
        for i in 0..n {
            let t = t0 + i as f64 * 0.5;
            for (l, f) in send(t) {
                fu.ingest(l, f, t);
            }
            ev.extend(sys.observe("lab", fu, t));
        }
        ev
    }

    fn kinds(ev: &[HealthEvent]) -> Vec<&EventKind> {
        ev.iter().map(|e| &e.kind).collect()
    }

    #[test]
    fn link_down_then_up_once_each() {
        let (mut sys, mut fu) = (SystemMonitor::new(), fusion(&[A]));
        let ok = |_t: f64| vec![(A, frame(20.0, 0.0, 0))];
        let mut ev = drive(&mut sys, &mut fu, 0.0, 10.0, ok);
        assert!(ev.is_empty(), "{ev:?}");
        ev = drive(&mut sys, &mut fu, 10.0, 30.0, |_| vec![]);
        let downs: Vec<_> = ev.iter().filter(|e| matches!(e.kind, EventKind::LinkDown { .. })).collect();
        assert_eq!(downs.len(), 1);
        assert!(matches!(downs[0].kind, EventKind::LinkDown { link: A, silent_s } if silent_s > 5.0 && silent_s <= 6.0));
        assert!((downs[0].t_s - 15.0).abs() <= 0.6);
        let rep = sys.link_report(&fu, 29.5);
        assert!(!rep[0].up && rep[0].silent_s > 19.0 && rep[0].frames == 20);
        assert_eq!(ev.iter().filter(|e| e.kind == EventKind::SpaceBlind).count(), 1, "its only link");
        ev = drive(&mut sys, &mut fu, 30.0, 40.0, ok);
        assert_eq!(kinds(&ev), vec![&EventKind::LinkUp { link: A }, &EventKind::SpaceVisible]);
        let rep = sys.link_report(&fu, 39.5);
        assert!(rep[0].up && rep[0].rate_hz == 20.0 && rep[0].calib_age_h == Some(1.0));
        assert_eq!((rep[0].rssi_dbm, rep[0].noise_floor_dbm, rep[0].raw_mode), (-50, -92, false));
    }

    #[test]
    fn never_heard_link_is_reported_down() {
        let (mut sys, mut fu) = (SystemMonitor::new(), fusion(&[A, B]));
        let ev = drive(&mut sys, &mut fu, 100.0, 120.0, |_| vec![(A, frame(20.0, 0.0, 0))]);
        assert_eq!(kinds(&ev), vec![&EventKind::LinkDown { link: B, silent_s: 5.5 }]);
        let rep = sys.link_report(&fu, 119.5);
        let b = rep.iter().find(|r| r.link == B).unwrap();
        assert!(!b.up && b.frames == 0 && (b.silent_s - 19.5).abs() < 1e-9 && b.calib_age_h.is_none());
    }

    #[test]
    fn degraded_only_when_sustained() {
        let (mut sys, mut fu) = (SystemMonitor::new(), fusion(&[A]));
        // A 10 s dip is not reported.
        let ev = drive(&mut sys, &mut fu, 0.0, 60.0, |t| {
            vec![(A, frame(if (20.0..30.0).contains(&t) { 7.0 } else { 20.0 }, 0.0, 0))]
        });
        assert!(ev.is_empty(), "{ev:?}");
        // 40 s slow: reported once, about 30 s in.
        let ev = drive(&mut sys, &mut fu, 60.0, 100.0, |_| vec![(A, frame(7.0, 0.0, 0))]);
        assert_eq!(ev.len(), 1);
        assert!((ev[0].t_s - 90.0).abs() <= 0.6);
        match &ev[0].kind {
            EventKind::LinkDegraded { link, reason } => {
                assert_eq!(*link, A);
                assert!(reason.contains("7.0 Hz"), "{reason}");
            }
            k => panic!("{k:?}"),
        }
        // Loss instead of rate, without clearing first: same episode, no repeat.
        let ev = drive(&mut sys, &mut fu, 100.0, 140.0, |_| vec![(A, frame(20.0, 0.15, 0))]);
        assert!(ev.is_empty());
        // Clear for 30 s, then lossy again: a new episode.
        drive(&mut sys, &mut fu, 140.0, 180.0, |_| vec![(A, frame(20.0, 0.0, 0))]);
        let ev = drive(&mut sys, &mut fu, 180.0, 220.0, |_| vec![(A, frame(20.0, 0.15, 0))]);
        assert_eq!(ev.len(), 1);
        assert!(matches!(&ev[0].kind, EventKind::LinkDegraded { reason, .. } if reason.contains("loss 15%")));
    }

    #[test]
    fn raw_mode_notice_is_rate_limited() {
        let (mut sys, mut fu) = (SystemMonitor::new(), fusion(&[A]));
        let ev = drive(&mut sys, &mut fu, 0.0, 7300.0, |_| vec![(A, frame(20.0, 0.0, FLAG_RAW_MODE))]);
        let raw: Vec<_> = ev.iter().filter(|e| matches!(e.kind, EventKind::RawModeActive { .. })).collect();
        assert_eq!(raw.len(), 3, "at 0 s, 3600 s and 7200 s");
        assert!(raw.iter().all(|e| e.severity == Severity::Advisory));
        assert_eq!(raw[1].t_s, 3600.0);
        assert!(sys.link_report(&fu, 7299.5)[0].raw_mode);
    }

    #[test]
    fn calibration_stale_repeats_every_six_hours() {
        let (mut sys, mut fu) = (SystemMonitor::new(), fusion(&[A]));
        let stale = FeatureFrame { calib_age_s: 25 * 3600, ..frame(20.0, 0.0, 0) };
        let mut ev = Vec::new();
        for t in [0.0, 0.5, 3600.0, 6.0 * 3600.0, 6.0 * 3600.0 + 0.5] {
            fu.ingest(A, stale, t);
            ev.extend(sys.observe("lab", &fu, t));
        }
        let st: Vec<_> = ev.iter().filter(|e| matches!(e.kind, EventKind::CalibrationStale { .. })).collect();
        assert_eq!(st.len(), 2);
        assert!(matches!(st[0].kind, EventKind::CalibrationStale { age_h, .. } if age_h == 25.0));
    }

    #[test]
    fn space_blind_and_visible() {
        let (mut sys, mut fu) = (SystemMonitor::new(), fusion(&[A, B]));
        let both = |_t: f64| vec![(A, frame(20.0, 0.0, 0)), (B, frame(20.0, 0.0, 0))];
        assert!(drive(&mut sys, &mut fu, 0.0, 10.0, both).is_empty());
        let ev = drive(&mut sys, &mut fu, 10.0, 40.0, |_| vec![]);
        let blind: Vec<_> = ev.iter().filter(|e| e.kind == EventKind::SpaceBlind).collect();
        assert_eq!(blind.len(), 1);
        // Links go unusable 2.5 s after their last frame (9.5 s), blind > 10 s later.
        assert!(blind[0].t_s > 22.0 && blind[0].t_s < 23.5, "{}", blind[0].t_s);
        assert_eq!(ev.iter().filter(|e| matches!(e.kind, EventKind::LinkDown { .. })).count(), 2);
        let ev = drive(&mut sys, &mut fu, 40.0, 50.0, |_| vec![(B, frame(20.0, 0.0, 0))]);
        assert_eq!(kinds(&ev), vec![&EventKind::LinkUp { link: B }, &EventKind::SpaceVisible]);
    }

    #[test]
    fn a_short_outage_is_not_blindness() {
        let (mut sys, mut fu) = (SystemMonitor::new(), fusion(&[A]));
        let ev = drive(&mut sys, &mut fu, 0.0, 60.0, |t| {
            if (20.0..24.0).contains(&t) {
                vec![]
            } else {
                vec![(A, frame(20.0, 0.0, 0))]
            }
        });
        assert!(ev.is_empty(), "{ev:?}");
    }
}
