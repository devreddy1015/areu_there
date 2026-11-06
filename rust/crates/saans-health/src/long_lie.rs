//! Long lie (issue #38, `docs/ELDER_CARE.md` §3.3): a sudden movement,
//! then a person present and not moving for N minutes.
//!
//! The state machine is here; `SpaceMonitor` feeds it one sample at a time
//! and words the events. It reuses the possible-collapse arming unchanged
//! (a burst at or above `BURST_INDEX` that settles into stillness within
//! `COLLAPSE_SETTLE_S`), and then asks for time instead of posture, which
//! amplitude-only WiFi cannot see (`docs/CAPABILITY_LIMITS.md` §1):
//!
//! - **present** is measured occupancy, or occupancy held by the presence
//!   hold (`SpaceConfig::fall_hold_min`) when that is on. Short empty
//!   readings are tolerated the way the stillness episode tolerates them
//!   (`VACANCY_GRACE_S`); a room that stays empty disarms it.
//! - **not moving** is the monitor's own definition (`STILL_INDEX`,
//!   `MOVE_CONFIRM_SAMPLES`), so the long lie and the inactivity alert agree
//!   about what stillness is.
//! - It fires once per lie. Gross motion afterwards is reported as
//!   movement seen, and never resolves anything: moving is not the same as
//!   being fine. The alert record is the alert service's and stays open
//!   until a person resolves it; `SpaceState::active_alerts` shows the
//!   condition itself, so it lists `long_lie` only until motion is seen, as
//!   the inactivity and collapse entries do.
//! - Its confidence is the share of the lie in which the links *measured*
//!   someone present. Held presence keeps it running but does not count as
//!   seen.
//!
//! **Known limitation, stated where it lives:** a person who lies down
//! quickly to rest, sits down heavily to read or kneels to pray makes the
//! same burst and the same stillness. The time condition is all that
//! separates them, which is why it is per space and off by default, and
//! why `saans longlie` reports the false alarms next to the misses
//! (ADR-005).

use crate::detectors::{BURST_INDEX, COLLAPSE_SETTLE_S, VACANCY_GRACE_S};

/// A burst seen, and (once it settles) the start of the stillness after it.
#[derive(Debug, Clone, Copy)]
struct Arm {
    /// Last sample at or above `BURST_INDEX`.
    burst_t: f64,
    peak: f64,
    still_start: Option<f64>,
    /// Samples since `still_start`, and how many of them measured someone
    /// present (held presence is not measured).
    samples: u64,
    seen: u64,
}

/// The alert has fired; kept until gross motion is seen.
#[derive(Debug, Clone, Copy)]
struct Alerted {
    burst_t: f64,
    peak: f64,
    still_start: f64,
    alert_t: f64,
}

/// One space's long-lie tracker.
#[derive(Debug, Clone, Default)]
pub(crate) struct LongLie {
    arm: Option<Arm>,
    alerted: Option<Alerted>,
}

/// What the monitor knows about one sample.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Step {
    pub t: f64,
    /// Some usable link (`occupied.is_some()`).
    pub visible: bool,
    pub motion_index: f64,
    /// The monitor's confirmed gross motion.
    pub moving: bool,
    /// Occupied, measured or held.
    pub present: bool,
    /// Occupied as the links measured it: `present` and not held.
    pub measured: bool,
    /// The monitor's stillness episode is alive (it survives short empty or
    /// blind readings and ends on motion or a sustained vacancy).
    pub still_alive: bool,
    /// `long_lie_min * 60`; not positive and finite means off.
    pub limit_s: f64,
}

/// What happened on this sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Outcome {
    /// The long lie: `still_s` of stillness while present, after a burst.
    /// `seen_fraction` is the share of those samples in which the links
    /// measured someone (the alert's confidence).
    Alert {
        burst_t: f64,
        peak: f64,
        settle_s: f64,
        still_s: f64,
        seen_fraction: f64,
    },
    /// Gross motion after the alert.
    Moved {
        burst_t: f64,
        peak: f64,
        still_s: f64,
        after_alert_s: f64,
    },
}

impl LongLie {
    /// Whether the alert has fired and no motion has been seen since.
    pub(crate) fn alerting(&self) -> bool {
        self.alerted.is_some()
    }

    /// A gap in the samples: forget a lie in progress. An alert already
    /// fired stays, so motion after the gap is still reported.
    pub(crate) fn interrupt(&mut self) {
        self.arm = None;
    }

    pub(crate) fn step(&mut self, s: &Step) -> Option<Outcome> {
        if !(s.limit_s > 0.0 && s.limit_s.is_finite()) {
            *self = Self::default();
            return None;
        }
        let t = s.t;
        if s.visible && s.motion_index >= BURST_INDEX {
            // A burst: arms a lie, and is motion after one that has fired.
            let moved = self.moved(t);
            let peak = match self.arm {
                Some(a) if a.still_start.is_none() && t - a.burst_t <= COLLAPSE_SETTLE_S => {
                    a.peak.max(s.motion_index)
                }
                _ => s.motion_index,
            };
            self.arm = Some(Arm {
                burst_t: t,
                peak,
                still_start: None,
                samples: 0,
                seen: 0,
            });
            return moved;
        }
        if s.moving {
            let moved = self.moved(t);
            // Motion after it settled is somebody moving about. Motion before
            // it settles is the burst dying away; if it carries on past
            // COLLAPSE_SETTLE_S, the first still sample drops the arm below
            // ("never settled").
            if self.arm.is_some_and(|a| a.still_start.is_some()) {
                self.arm = None;
            }
            return moved;
        }
        let a = self.arm.as_mut()?;
        if !s.visible && a.still_start.is_none() {
            // No usable link: nothing is known about stillness, so the
            // clock does not start. As in `collapse_step`, a burst not seen
            // to settle in time is dropped.
            if t - a.burst_t > COLLAPSE_SETTLE_S {
                self.arm = None;
            }
            return None;
        }
        let still_start = match a.still_start {
            Some(ss) => ss,
            None if t - a.burst_t <= COLLAPSE_SETTLE_S => *a.still_start.insert(t),
            None => {
                // Never settled: not a sudden stop.
                self.arm = None;
                return None;
            }
        };
        a.samples += 1;
        if s.measured {
            a.seen += 1;
        }
        if !s.still_alive {
            // The room has read empty (or blind) for too long: the
            // stillness episode is over, so this lie is too. The episode
            // gets the same grace to start as it has to survive a gap.
            if t - still_start > VACANCY_GRACE_S {
                self.arm = None;
            }
            return None;
        }
        if !s.present {
            return None;
        }
        let still_s = t - still_start;
        if still_s < s.limit_s {
            return None;
        }
        let (burst_t, peak) = (a.burst_t, a.peak);
        let seen_fraction = a.seen as f64 / a.samples.max(1) as f64;
        self.arm = None;
        self.alerted = Some(Alerted {
            burst_t,
            peak,
            still_start,
            alert_t: t,
        });
        Some(Outcome::Alert {
            burst_t,
            peak,
            settle_s: still_start - burst_t,
            still_s,
            seen_fraction,
        })
    }

    /// Gross motion: report it once if the alert has fired.
    fn moved(&mut self, t: f64) -> Option<Outcome> {
        self.alerted.take().map(|a| Outcome::Moved {
            burst_t: a.burst_t,
            peak: a.peak,
            still_s: t - a.still_start,
            after_alert_s: t - a.alert_t,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::detectors::{SpaceMonitor, SAMPLE_INTERVAL_S};
    use crate::types::{EventKind, HealthEvent, Severity, SpaceConfig, SpaceSample};

    const N_MIN: f64 = 15.0;

    fn cfg(long_lie_min: f64, fall_hold_min: f64) -> SpaceConfig {
        SpaceConfig {
            id: "bathroom".into(),
            long_lie_min,
            fall_hold_min,
            // The 30-minute inactivity alert stays on; the assertions
            // filter on kind.
            ..Default::default()
        }
    }

    fn sample(t: f64, occupied: Option<bool>, motion: f64) -> SpaceSample {
        SpaceSample {
            t_s: t,
            occupied,
            presence_index: if occupied == Some(true) { 3.0 } else { 0.5 },
            motion_index: motion,
            links_usable: if occupied.is_some() { 2 } else { 0 },
            links_total: 2,
            breathing: None,
            heart: None,
        }
    }

    /// Feed samples at 2 Hz for t in [t0, t1).
    fn run(
        m: &mut SpaceMonitor,
        t0: f64,
        t1: f64,
        f: impl Fn(f64) -> SpaceSample,
    ) -> Vec<HealthEvent> {
        let n = ((t1 - t0) / SAMPLE_INTERVAL_S).round() as usize;
        (0..n)
            .flat_map(|i| m.update(&f(t0 + i as f64 * SAMPLE_INTERVAL_S)))
            .collect()
    }

    fn present_still(t: f64) -> SpaceSample {
        sample(t, Some(true), 0.5)
    }

    fn walking(t: f64) -> SpaceSample {
        sample(t, Some(true), 4.0)
    }

    fn burst(t: f64) -> SpaceSample {
        sample(t, Some(true), 12.0)
    }

    fn empty(t: f64) -> SpaceSample {
        sample(t, Some(false), 0.2)
    }

    fn long_lies(ev: &[HealthEvent]) -> Vec<&HealthEvent> {
        ev.iter()
            .filter(|e| matches!(e.kind, EventKind::LongLie { .. }))
            .collect()
    }

    /// Someone walks in (60 s), goes down suddenly (1 s burst at t = 160)
    /// and lies still from t = 161. Returns the events up to `until`.
    fn down_at_160(m: &mut SpaceMonitor, until: f64) -> Vec<HealthEvent> {
        let mut ev = run(m, 0.0, 100.0, empty);
        ev.extend(run(m, 100.0, 160.0, walking));
        ev.extend(run(m, 160.0, 161.0, burst));
        ev.extend(run(m, 161.0, until, present_still));
        ev
    }

    #[test]
    fn off_by_default() {
        let mut m = SpaceMonitor::new(cfg(0.0, 0.0));
        let ev = down_at_160(&mut m, 161.0 + 40.0 * 60.0);
        assert!(long_lies(&ev).is_empty(), "{:?}", long_lies(&ev));
    }

    #[test]
    fn a_burst_then_n_minutes_still_is_a_check_without_vitals() {
        let mut m = SpaceMonitor::new(cfg(N_MIN, 0.0));
        assert!(!m.config().vitals, "vitals are off by default");
        let lie_end = 161.0 + N_MIN * 60.0;
        let mut ev = down_at_160(&mut m, lie_end - 5.0);
        assert!(long_lies(&ev).is_empty(), "not before N minutes");
        ev.extend(run(&mut m, lie_end - 5.0, lie_end + 5.0, present_still));
        let ll = long_lies(&ev);
        assert_eq!(ll.len(), 1, "{ll:?}");
        let e = ll[0];
        assert_eq!(e.severity, Severity::Check);
        assert!((e.t_s - lie_end).abs() <= 1.0, "fired at {}", e.t_s);
        assert!(
            (e.t_start_s.unwrap() - 160.5).abs() <= 1.0,
            "starts at the burst"
        );
        match e.kind {
            EventKind::LongLie {
                still_min,
                burst_index,
                movement_seen,
            } => {
                assert!(!movement_seen);
                assert!((still_min - N_MIN).abs() < 0.05, "{still_min}");
                assert!((burst_index - 12.0).abs() < 1e-9);
            }
            _ => unreachable!(),
        }
        assert!((e.confidence - 1.0).abs() < 1e-9, "present the whole time");
        let w = e.evidence.to_lowercase();
        assert!(
            w.contains("15.0 min") && w.contains("check on the space"),
            "{}",
            e.evidence
        );
        assert!(!w.contains("fall"), "wellness wording only: {}", e.evidence);
        assert!(m.state().active_alerts.contains(&"long_lie".to_string()));
    }

    #[test]
    fn it_fires_once_per_lie() {
        let mut m = SpaceMonitor::new(cfg(N_MIN, 0.0));
        let ev = down_at_160(&mut m, 161.0 + 3.0 * N_MIN * 60.0);
        assert_eq!(long_lies(&ev).len(), 1);
    }

    #[test]
    fn stillness_without_a_burst_is_not_a_long_lie() {
        let mut m = SpaceMonitor::new(cfg(N_MIN, 0.0));
        let mut ev = run(&mut m, 0.0, 100.0, empty);
        ev.extend(run(&mut m, 100.0, 160.0, walking));
        ev.extend(run(&mut m, 160.0, 160.0 + 40.0 * 60.0, present_still));
        assert!(long_lies(&ev).is_empty());
    }

    #[test]
    fn motion_that_carries_on_after_the_burst_disarms_it() {
        // A burst that does not stop within COLLAPSE_SETTLE_S is someone
        // moving about, not going down.
        let mut m = SpaceMonitor::new(cfg(N_MIN, 0.0));
        let mut ev = run(&mut m, 0.0, 100.0, empty);
        ev.extend(run(&mut m, 100.0, 160.0, walking));
        ev.extend(run(&mut m, 160.0, 161.0, burst));
        ev.extend(run(&mut m, 161.0, 175.0, walking));
        ev.extend(run(&mut m, 175.0, 175.0 + 30.0 * 60.0, present_still));
        assert!(long_lies(&ev).is_empty());
    }

    #[test]
    fn moving_before_n_minutes_cancels_it() {
        let mut m = SpaceMonitor::new(cfg(N_MIN, 0.0));
        let t = 161.0 + 10.0 * 60.0;
        let mut ev = down_at_160(&mut m, t);
        ev.extend(run(&mut m, t, t + 3.0, walking));
        ev.extend(run(&mut m, t + 3.0, t + 3.0 + 30.0 * 60.0, present_still));
        assert!(long_lies(&ev).is_empty(), "{:?}", long_lies(&ev));
    }

    #[test]
    fn motion_after_the_alert_is_reported_and_pages_nobody() {
        let mut m = SpaceMonitor::new(cfg(N_MIN, 0.0));
        let t = 161.0 + N_MIN * 60.0 + 60.0;
        let mut ev = down_at_160(&mut m, t);
        ev.extend(run(&mut m, t, t + 3.0, walking));
        let ll = long_lies(&ev);
        assert_eq!(ll.len(), 2, "{ll:?}");
        let moved = ll[1];
        assert_eq!(moved.severity, Severity::Info, "Info never pages");
        assert!(matches!(
            moved.kind,
            EventKind::LongLie {
                movement_seen: true,
                ..
            }
        ));
        assert!(moved.evidence.contains("stays open"), "{}", moved.evidence);
        assert!(!m.state().active_alerts.contains(&"long_lie".to_string()));
        // And it does not fire again for the same lie.
        ev.extend(run(&mut m, t + 3.0, t + 3.0 + 20.0 * 60.0, present_still));
        assert_eq!(long_lies(&ev).len(), 2);
    }

    #[test]
    fn a_room_that_reads_empty_after_the_burst_never_starts_it() {
        // Without the presence hold, someone who goes down out of the
        // links' sight reads as having left (docs/ELDER_CARE.md §3.3,
        // "The person who vanishes"). This pins that, so the cost of the
        // hold being off is visible in a test, not only in prose.
        let mut m = SpaceMonitor::new(cfg(N_MIN, 0.0));
        let mut ev = run(&mut m, 0.0, 100.0, empty);
        ev.extend(run(&mut m, 100.0, 160.0, walking));
        ev.extend(run(&mut m, 160.0, 161.0, burst));
        ev.extend(run(&mut m, 161.0, 161.0 + 30.0 * 60.0, empty));
        assert!(long_lies(&ev).is_empty());
    }

    #[test]
    fn with_the_presence_hold_on_it_fires_and_says_inferred() {
        let mut m = SpaceMonitor::new(cfg(N_MIN, N_MIN + 5.0));
        let mut ev = run(&mut m, 0.0, 100.0, empty);
        ev.extend(run(&mut m, 100.0, 160.0, walking));
        ev.extend(run(&mut m, 160.0, 161.0, burst));
        ev.extend(run(&mut m, 161.0, 170.0, present_still));
        // The links lose them on the floor.
        ev.extend(run(&mut m, 170.0, 161.0 + N_MIN * 60.0 + 10.0, empty));
        let ll = long_lies(&ev);
        assert_eq!(ll.len(), 1, "{ll:?}");
        let e = ll[0];
        assert_eq!(e.severity, Severity::Check);
        assert!(
            e.evidence.contains("inferred, not measured"),
            "{}",
            e.evidence
        );
        // Held presence is not seen presence: the confidence is the share
        // of the lie in which someone was measured (about 9 s of 15 min).
        assert!(e.confidence < 0.02, "{}", e.confidence);
    }

    #[test]
    fn a_gap_in_the_samples_forgets_a_lie_in_progress() {
        let mut m = SpaceMonitor::new(cfg(N_MIN, 0.0));
        let t = 161.0 + 5.0 * 60.0;
        let mut ev = down_at_160(&mut m, t);
        // 30 s with no samples at all, then the lie carries on.
        ev.extend(run(&mut m, t + 30.0, t + 30.0 + 20.0 * 60.0, present_still));
        assert!(long_lies(&ev).is_empty(), "{:?}", long_lies(&ev));
    }

    #[test]
    fn a_new_burst_after_the_alert_reports_movement_and_can_arm_a_new_lie() {
        let mut m = SpaceMonitor::new(cfg(N_MIN, 0.0));
        let t = 161.0 + N_MIN * 60.0 + 60.0;
        let mut ev = down_at_160(&mut m, t);
        ev.extend(run(&mut m, t, t + 1.0, burst));
        ev.extend(run(
            &mut m,
            t + 1.0,
            t + 1.0 + N_MIN * 60.0 + 5.0,
            present_still,
        ));
        let alerts = |moved: bool| {
            ev.iter()
                .filter(|e| {
                    matches!(e.kind, EventKind::LongLie { movement_seen, .. } if movement_seen == moved)
                })
                .count()
        };
        assert_eq!(alerts(true), 1, "the burst is movement after the alert");
        assert_eq!(alerts(false), 2, "and a new lie after it alerts again");
    }

    #[test]
    fn minutes_that_are_not_a_positive_number_turn_it_off() {
        for n in [f64::NAN, f64::INFINITY, -5.0] {
            let mut m = SpaceMonitor::new(cfg(n, 0.0));
            let ev = down_at_160(&mut m, 161.0 + 40.0 * 60.0);
            assert!(long_lies(&ev).is_empty(), "{n}");
        }
    }

    #[test]
    fn blind_samples_do_not_start_the_stillness_clock() {
        // No usable link right after the burst: nothing is known about
        // stillness, so the clock waits for a visible still sample, and a
        // burst that is not seen to settle in time is dropped.
        let mut m = SpaceMonitor::new(cfg(N_MIN, 0.0));
        let mut ev = run(&mut m, 0.0, 100.0, empty);
        ev.extend(run(&mut m, 100.0, 160.0, walking));
        ev.extend(run(&mut m, 160.0, 161.0, burst));
        ev.extend(run(&mut m, 161.0, 170.0, |t| sample(t, None, 0.0)));
        ev.extend(run(&mut m, 170.0, 170.0 + 30.0 * 60.0, present_still));
        assert!(long_lies(&ev).is_empty(), "{:?}", long_lies(&ev));
    }

    #[test]
    fn a_short_empty_reading_does_not_disarm_it() {
        let mut m = SpaceMonitor::new(cfg(N_MIN, 0.0));
        let t = 161.0 + 5.0 * 60.0;
        let mut ev = down_at_160(&mut m, t);
        ev.extend(run(&mut m, t, t + 10.0, empty));
        ev.extend(run(
            &mut m,
            t + 10.0,
            161.0 + N_MIN * 60.0 + 5.0,
            present_still,
        ));
        let ll = long_lies(&ev);
        assert_eq!(ll.len(), 1, "{ll:?}");
        assert!(
            ll[0].confidence < 1.0 && ll[0].confidence > 0.95,
            "{}",
            ll[0].confidence
        );
    }

    #[test]
    fn a_sustained_empty_reading_ends_the_lie() {
        // Empty for longer than the stillness grace: that lie is over. A
        // still person seen again afterwards has made no new burst, so
        // there is nothing to arm (the presence hold is the remedy, not a
        // lie that outlives its evidence).
        let mut m = SpaceMonitor::new(cfg(N_MIN, 0.0));
        let t = 161.0 + 5.0 * 60.0;
        let mut ev = down_at_160(&mut m, t);
        ev.extend(run(&mut m, t, t + 60.0, empty));
        ev.extend(run(&mut m, t + 60.0, t + 60.0 + 30.0 * 60.0, present_still));
        assert!(long_lies(&ev).is_empty(), "{:?}", long_lies(&ev));
    }

    #[test]
    fn it_waits_for_presence_before_firing() {
        // N minutes pass while the room briefly reads empty: the alert
        // waits until someone is seen, and does not fire on an empty room.
        let mut m = SpaceMonitor::new(cfg(N_MIN, 0.0));
        let lie_end = 161.0 + N_MIN * 60.0;
        let mut ev = down_at_160(&mut m, lie_end - 10.0);
        ev.extend(run(&mut m, lie_end - 10.0, lie_end + 10.0, empty));
        assert!(long_lies(&ev).is_empty(), "not while the room reads empty");
        ev.extend(run(&mut m, lie_end + 10.0, lie_end + 20.0, present_still));
        let ll = long_lies(&ev);
        assert_eq!(ll.len(), 1, "{ll:?}");
        assert!(ll[0].t_s >= lie_end + 10.0, "fired at {}", ll[0].t_s);
    }

    #[test]
    fn a_long_lie_does_not_change_the_other_detectors() {
        // Turning it on adds its own events and nothing else.
        let run_with = |n: f64| {
            let mut m = SpaceMonitor::new(cfg(n, 0.0));
            let ev = down_at_160(&mut m, 161.0 + 40.0 * 60.0);
            ev.into_iter()
                .filter(|e| !matches!(e.kind, EventKind::LongLie { .. }))
                .collect::<Vec<_>>()
        };
        assert_eq!(run_with(0.0), run_with(N_MIN));
    }
}
