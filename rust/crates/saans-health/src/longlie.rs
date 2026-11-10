//! Long lie (issue #38, docs/ELDER_CARE.md §3.3): one elder-care alert,
//! off by default. In one space, an urgent (`Check`) alert when
//!
//! 1. a sudden movement, motion index at least `BURST_INDEX`, stops
//!    within `COLLAPSE_SETTLE_S` (the possible-collapse arming, unchanged);
//! 2. then the space stays present and without gross motion for
//!    `SpaceConfig::long_lie_min`, with presence measured, or held if the
//!    space has the presence hold on (`fall_hold_min`);
//! 3. and nobody is seen moving in one of the spaces named in
//!    `long_lie_elsewhere` meanwhile (the gateway reports that). Only
//!    motion that starts there after the count began counts: motion
//!    already going on when it began is the person's own trail (they
//!    walked in from there) or the movement itself seen through a wall.
//!
//! It reads only presence and motion: no vitals, and it does not need
//! `collapse_detection`. Gross motion after the alert is reported once
//! (`LongLieMovement`, `Info`) and does not resolve it; the gateway's
//! alert keeps it open until a person does.
//!
//! What it cannot do, by construction: catch a person who reaches the
//! floor without a sudden movement (a slow slide), or one who keeps
//! moving a limb on the floor, or one the links lose for longer than the
//! grace below with the presence hold off ("the person who vanishes").
//! And it cannot tell a fall from lying down quickly for a nap, or from
//! sitting down heavily to read: WiFi amplitude does not see height.

use crate::detectors::{BLIND_END_S, BURST_INDEX, COLLAPSE_SETTLE_S, VACANCY_GRACE_S};
use crate::types::{EventKind, HealthEvent, LongLieStatus, Severity};

/// One sample, as the space monitor sees it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Input {
    pub t: f64,
    /// Some link can see the space.
    pub visible: bool,
    pub motion_index: f64,
    /// Gross motion (`SpaceMonitor`'s confirmed `moving`).
    pub moving: bool,
    /// Present, measured or held.
    pub present: bool,
    /// Present only because the presence hold is in force.
    pub held: bool,
    /// Gross motion was seen in a `long_lie_elsewhere` space.
    pub elsewhere: bool,
}

#[derive(Debug, Clone, Copy)]
enum Phase {
    Idle,
    /// A burst; waiting for the motion to stop.
    Burst {
        peak: f64,
        last_t: f64,
    },
    Counting(Count),
    /// Alert raised at `fired_t`; waiting for gross motion.
    Alerted {
        count: Count,
        fired_t: f64,
    },
}

#[derive(Debug, Clone, Copy)]
struct Count {
    peak: f64,
    burst_t: f64,
    still_since: f64,
    empty_since: Option<f64>,
    blind_since: Option<f64>,
    samples: u64,
    held_samples: u64,
    /// Samples in which the links measured someone present (not held).
    seen_samples: u64,
    held_now: bool,
    /// The `long_lie_elsewhere` spaces have been seen without motion
    /// since the count began: motion there from now on is new.
    elsewhere_quiet: bool,
}

/// The long-lie detector for one space.
#[derive(Debug, Clone)]
pub(crate) struct LongLie {
    limit_s: f64,
    phase: Phase,
}

impl LongLie {
    pub fn new(long_lie_min: f64) -> Self {
        Self {
            limit_s: long_lie_min * 60.0,
            phase: Phase::Idle,
        }
    }

    fn on(&self) -> bool {
        self.limit_s > 0.0 && self.limit_s.is_finite()
    }

    /// A data gap: anything could have happened unseen, so a count stops.
    /// An alert already raised stays raised until motion is seen.
    pub fn discontinuity(&mut self) {
        if !matches!(self.phase, Phase::Alerted { .. }) {
            self.phase = Phase::Idle;
        }
    }

    pub fn status(&self, t: f64) -> Option<LongLieStatus> {
        let (c, alerted) = match &self.phase {
            Phase::Counting(c) => (c, false),
            Phase::Alerted { count, .. } => (count, true),
            _ => return None,
        };
        Some(LongLieStatus {
            still_s: (t - c.still_since).max(0.0),
            limit_s: self.limit_s,
            held: c.held_now,
            alerted,
        })
    }

    pub fn alerted(&self) -> bool {
        matches!(self.phase, Phase::Alerted { .. })
    }

    /// Feed one sample; returns the alert, or the movement after it.
    pub fn step(&mut self, space: &str, x: &Input) -> Vec<HealthEvent> {
        let mut ev = Vec::new();
        if !self.on() {
            self.phase = Phase::Idle;
            return ev;
        }
        if let Phase::Alerted { fired_t, .. } = self.phase {
            if !x.moving {
                return ev;
            }
            ev.push(movement(space, x.t, fired_t));
            // The same sample can be the burst of a new count.
            self.phase = Phase::Idle;
        }
        if x.visible && x.motion_index >= BURST_INDEX {
            let peak = match self.phase {
                Phase::Burst { peak, last_t } if x.t - last_t <= COLLAPSE_SETTLE_S => {
                    peak.max(x.motion_index)
                }
                _ => x.motion_index,
            };
            self.phase = Phase::Burst { peak, last_t: x.t };
            return ev;
        }
        match self.phase {
            Phase::Idle | Phase::Alerted { .. } => {}
            Phase::Burst { peak, last_t } => {
                if x.t - last_t > COLLAPSE_SETTLE_S {
                    // Kept moving, or never seen to stop: they got up.
                    self.phase = Phase::Idle;
                } else if !x.moving && x.visible {
                    self.phase = Phase::Counting(Count {
                        peak,
                        burst_t: last_t,
                        still_since: x.t,
                        empty_since: None,
                        blind_since: None,
                        samples: 0,
                        held_samples: 0,
                        seen_samples: 0,
                        held_now: false,
                        elsewhere_quiet: false,
                    });
                    self.count(space, x, &mut ev);
                }
            }
            Phase::Counting(_) => self.count(space, x, &mut ev),
        }
        ev
    }

    fn count(&mut self, space: &str, x: &Input, ev: &mut Vec<HealthEvent>) {
        let Phase::Counting(mut c) = self.phase else {
            return;
        };
        if x.moving || (x.elsewhere && c.elsewhere_quiet) {
            self.phase = Phase::Idle;
            return;
        }
        c.elsewhere_quiet |= !x.elsewhere;
        if !x.visible {
            let since = *c.blind_since.get_or_insert(x.t);
            if x.t - since > BLIND_END_S {
                self.phase = Phase::Idle;
                return;
            }
        } else if !x.present {
            c.blind_since = None;
            let since = *c.empty_since.get_or_insert(x.t);
            if x.t - since > VACANCY_GRACE_S {
                // The person who vanishes: with the hold off, a person the
                // links lose reads as having left, and nothing is raised.
                self.phase = Phase::Idle;
                return;
            }
        } else {
            c.blind_since = None;
            c.empty_since = None;
        }
        c.samples += 1;
        if x.held {
            c.held_samples += 1;
        } else if x.present {
            c.seen_samples += 1;
        }
        c.held_now = x.held;
        let still_s = x.t - c.still_since;
        if !(x.present && still_s >= self.limit_s) {
            self.phase = Phase::Counting(c);
            return;
        }
        let held_fraction = c.held_samples as f64 / c.samples.max(1) as f64;
        let presence = if c.held_samples == 0 {
            "with presence measured throughout".to_string()
        } else {
            format!(
                "with presence held, inferred, not measured, for {:.0}% of that time (the links read empty \
                 after the movement and the presence hold kept the space occupied)",
                held_fraction * 100.0
            )
        };
        let evidence = format!(
            "no gross motion for {:.1} min after a sudden movement (motion index {:.1}, {:.0}x the stillness \
             limit) that stopped within {:.1} s, {presence}. Lying down quickly for a nap, or sitting down \
             heavily and staying still, looks the same to WiFi, so this is advisory: check on the space",
            still_s / 60.0,
            c.peak,
            c.peak / crate::detectors::STILL_INDEX,
            c.still_since - c.burst_t
        );
        ev.push(HealthEvent {
            space_id: space.to_string(),
            t_s: x.t,
            t_start_s: Some(c.burst_t),
            severity: Severity::Check,
            // The share of the lie in which the links measured someone:
            // held presence keeps it running but is not a sighting. Ported
            // from PR #59. It is not the chance that this is a fall; a nap
            // after lying down quickly scores the same.
            confidence: c.seen_samples as f64 / c.samples.max(1) as f64,
            evidence,
            kind: EventKind::LongLie {
                still_min: still_s / 60.0,
                burst_index: c.peak,
                held_fraction,
            },
        });
        self.phase = Phase::Alerted {
            count: c,
            fired_t: x.t,
        };
    }
}

fn movement(space: &str, t: f64, fired_t: f64) -> HealthEvent {
    HealthEvent {
        space_id: space.to_string(),
        t_s: t,
        t_start_s: Some(fired_t),
        severity: Severity::Info,
        confidence: 1.0,
        evidence: format!(
            "gross motion seen in the space {:.1} min after the long-lie alert; moving does not mean being \
             fine, so the alert stays open until someone resolves it",
            (t - fired_t) / 60.0
        ),
        kind: EventKind::LongLieMovement {
            after_alert_s: t - fired_t,
        },
    }
}

// ----------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use crate::types::{EventKind, HealthEvent, Severity, SpaceConfig, SpaceSample};
    use crate::SpaceMonitor;

    const DT: f64 = 0.5;

    fn cfg(long_lie_min: f64) -> SpaceConfig {
        SpaceConfig {
            id: "bathroom".into(),
            long_lie_min,
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

    fn still(t: f64) -> SpaceSample {
        sample(t, Some(true), 0.5)
    }

    /// Feed samples at 2 Hz for t in [t0, t1).
    fn run(
        m: &mut SpaceMonitor,
        t0: f64,
        t1: f64,
        f: impl Fn(f64) -> SpaceSample,
    ) -> Vec<HealthEvent> {
        let n = ((t1 - t0) / DT).round() as usize;
        (0..n)
            .flat_map(|i| m.update(&f(t0 + i as f64 * DT)))
            .collect()
    }

    /// Present and walking about until t = 60, a sudden movement at
    /// 60.0-61.0 (index 9.5), still from 61.0.
    fn arrive_and_drop(m: &mut SpaceMonitor) -> Vec<HealthEvent> {
        let mut ev = run(m, 0.0, 60.0, |t| sample(t, Some(true), 4.0));
        ev.extend(run(m, 60.0, 61.0, |t| sample(t, Some(true), 9.5)));
        ev
    }

    fn lies(ev: &[HealthEvent]) -> Vec<&HealthEvent> {
        ev.iter()
            .filter(|e| matches!(e.kind, EventKind::LongLie { .. }))
            .collect()
    }

    fn movements(ev: &[HealthEvent]) -> Vec<&HealthEvent> {
        ev.iter()
            .filter(|e| matches!(e.kind, EventKind::LongLieMovement { .. }))
            .collect()
    }

    #[test]
    fn off_by_default() {
        let mut m = SpaceMonitor::new(cfg(0.0));
        let mut ev = arrive_and_drop(&mut m);
        ev.extend(run(&mut m, 61.0, 3700.0, still));
        assert!(lies(&ev).is_empty());
        assert!(m.state().long_lie.is_none());
    }

    #[test]
    fn a_sudden_movement_then_n_minutes_still_is_a_check_without_vitals() {
        let mut m = SpaceMonitor::new(cfg(3.0));
        assert!(!m.config().vitals && !m.config().collapse_detection);
        let mut ev = arrive_and_drop(&mut m);
        ev.extend(run(&mut m, 61.0, 61.0 + 240.0, still));
        let l = lies(&ev);
        assert_eq!(l.len(), 1, "{ev:?}");
        let e = l[0];
        assert_eq!(e.severity, Severity::Check);
        assert!((e.t_s - (61.0 + 180.0)).abs() <= 1.0, "fired at {}", e.t_s);
        assert!((e.t_start_s.unwrap() - 60.5).abs() < 1e-9, "the burst");
        assert_eq!(e.confidence, 1.0, "measured present throughout");
        match e.kind {
            EventKind::LongLie {
                still_min,
                burst_index,
                held_fraction,
            } => {
                assert!((still_min - 3.0).abs() < 0.02, "{still_min}");
                assert_eq!(burst_index, 9.5);
                assert_eq!(held_fraction, 0.0);
            }
            _ => unreachable!(),
        }
        assert!(e.evidence.contains("sudden movement"), "{}", e.evidence);
        let words = e.evidence.to_lowercase();
        for banned in ["fall", "collapse", "unconscious", "injur"] {
            assert!(!words.contains(banned), "{banned}: {}", e.evidence);
        }
    }

    #[test]
    fn the_dashboard_counts_down_and_then_shows_the_alert() {
        let mut m = SpaceMonitor::new(cfg(3.0));
        arrive_and_drop(&mut m);
        run(&mut m, 61.0, 61.0 + 160.0, still);
        let s = m.state().long_lie.expect("counting");
        assert!((s.still_s - 159.5).abs() < 1e-9, "{}", s.still_s);
        assert_eq!(s.limit_s, 180.0);
        assert!(!s.alerted && !s.held);
        assert!(!m.state().active_alerts.contains(&"long_lie".to_string()));
        run(&mut m, 61.0 + 160.0, 61.0 + 200.0, still);
        let s = m.state().long_lie.expect("alerted");
        assert!(s.alerted);
        assert!(m.state().active_alerts.contains(&"long_lie".to_string()));
    }

    #[test]
    fn no_sudden_movement_no_alert() {
        // A slow slide to the floor, or a walk in and sitting down gently:
        // the motion never reaches BURST_INDEX.
        let mut m = SpaceMonitor::new(cfg(3.0));
        let mut ev = run(&mut m, 0.0, 60.0, |t| sample(t, Some(true), 7.9));
        ev.extend(run(&mut m, 60.0, 1000.0, still));
        assert!(lies(&ev).is_empty());
    }

    #[test]
    fn motion_that_does_not_stop_within_the_settle_time_does_not_arm() {
        let mut m = SpaceMonitor::new(cfg(3.0));
        let mut ev = arrive_and_drop(&mut m);
        // 8 s more of walking after the burst: they got up.
        ev.extend(run(&mut m, 61.0, 69.0, |t| sample(t, Some(true), 4.0)));
        ev.extend(run(&mut m, 69.0, 1000.0, still));
        assert!(lies(&ev).is_empty());
    }

    #[test]
    fn gross_motion_before_n_stops_the_count() {
        let mut m = SpaceMonitor::new(cfg(3.0));
        let mut ev = arrive_and_drop(&mut m);
        ev.extend(run(&mut m, 61.0, 181.0, still));
        ev.extend(run(&mut m, 181.0, 184.0, |t| sample(t, Some(true), 4.0)));
        ev.extend(run(&mut m, 184.0, 1000.0, still));
        assert!(lies(&ev).is_empty());
        assert!(m.state().long_lie.is_none());
    }

    #[test]
    fn a_short_dropout_is_tolerated_and_a_long_one_is_the_person_who_vanishes() {
        let short = |dropout_s: f64| {
            let mut m = SpaceMonitor::new(cfg(3.0));
            let mut ev = arrive_and_drop(&mut m);
            ev.extend(run(&mut m, 61.0, 120.0, still));
            ev.extend(run(&mut m, 120.0, 120.0 + dropout_s, |t| {
                sample(t, Some(false), 0.5)
            }));
            ev.extend(run(&mut m, 120.0 + dropout_s, 600.0, still));
            lies(&ev).len()
        };
        assert_eq!(short(10.0), 1, "10 s read empty: still counted");
        let mut m = SpaceMonitor::new(cfg(3.0));
        let mut ev = arrive_and_drop(&mut m);
        ev.extend(run(&mut m, 61.0, 120.0, still));
        ev.extend(run(&mut m, 120.0, 130.0, |t| sample(t, Some(false), 0.5)));
        ev.extend(run(&mut m, 130.0, 600.0, still));
        let c = lies(&ev)[0].confidence;
        // 20 of 360 samples read empty.
        assert!((c - 340.0 / 360.0).abs() < 0.01, "{c}");
        assert_eq!(short(60.0), 0, "60 s read empty: the count stops");
    }

    #[test]
    fn blindness_is_tolerated_for_a_minute() {
        let blind = |blind_s: f64| {
            let mut m = SpaceMonitor::new(cfg(3.0));
            let mut ev = arrive_and_drop(&mut m);
            ev.extend(run(&mut m, 61.0, 120.0, still));
            ev.extend(run(&mut m, 120.0, 120.0 + blind_s, |t| {
                sample(t, None, 0.0)
            }));
            ev.extend(run(&mut m, 120.0 + blind_s, 600.0, still));
            lies(&ev).len()
        };
        assert_eq!(blind(30.0), 1);
        assert_eq!(blind(90.0), 0);
    }

    #[test]
    fn with_the_presence_hold_on_a_person_the_links_lose_still_counts() {
        let held = |fall_hold_min: f64| {
            let mut m = SpaceMonitor::new(SpaceConfig {
                fall_hold_min,
                ..cfg(3.0)
            });
            let mut ev = arrive_and_drop(&mut m);
            ev.extend(run(&mut m, 61.0, 70.0, still));
            // On the floor, too weak a breather for the links.
            ev.extend(run(&mut m, 70.0, 400.0, |t| sample(t, Some(false), 0.5)));
            (ev, m)
        };
        let (ev, _) = held(0.0);
        assert!(lies(&ev).is_empty(), "hold off: the room reads empty");
        let (ev, m) = held(10.0);
        let l = lies(&ev);
        assert_eq!(l.len(), 1);
        match l[0].kind {
            EventKind::LongLie { held_fraction, .. } => {
                assert!(held_fraction > 0.9, "{held_fraction}")
            }
            _ => unreachable!(),
        }
        // Held presence keeps the lie running but is not a sighting.
        assert!(l[0].confidence < 0.1, "{}", l[0].confidence);
        assert!(
            l[0].evidence.contains("inferred, not measured"),
            "{}",
            l[0].evidence
        );
        assert!(m.state().long_lie.expect("alerted").held);
    }

    #[test]
    fn motion_in_another_room_stops_the_count() {
        let mut m = SpaceMonitor::new(cfg(3.0));
        let mut ev = arrive_and_drop(&mut m);
        ev.extend(run(&mut m, 61.0, 120.0, still));
        m.note_motion_elsewhere();
        ev.extend(run(&mut m, 120.0, 600.0, still));
        assert!(lies(&ev).is_empty());
    }

    #[test]
    fn motion_elsewhere_that_was_already_going_on_when_the_count_began_does_not_stop_it() {
        // They walked out of the living room and fell in the bathroom: the
        // living room still reads them moving for a few seconds after.
        let mut m = SpaceMonitor::new(cfg(3.0));
        let mut ev = run(&mut m, 0.0, 60.0, |t| sample(t, Some(true), 4.0));
        for i in 0..10 {
            let t = 60.0 + i as f64 * DT;
            m.note_motion_elsewhere();
            ev.extend(m.update(&sample(t, Some(true), if t < 61.0 { 9.5 } else { 0.5 })));
        }
        ev.extend(run(&mut m, 65.0, 400.0, still));
        assert_eq!(
            lies(&ev).len(),
            1,
            "trailing motion elsewhere is not a new sighting"
        );
    }

    #[test]
    fn movement_after_the_alert_is_reported_once_and_the_alert_is_not_resolved_by_it() {
        let mut m = SpaceMonitor::new(cfg(3.0));
        let mut ev = arrive_and_drop(&mut m);
        ev.extend(run(&mut m, 61.0, 300.0, still));
        assert_eq!(lies(&ev).len(), 1);
        // Still for a long time after the alert: no repeat.
        ev.extend(run(&mut m, 300.0, 1200.0, still));
        assert_eq!(lies(&ev).len(), 1);
        assert!(movements(&ev).is_empty());
        ev.extend(run(&mut m, 1200.0, 1205.0, |t| sample(t, Some(true), 4.0)));
        ev.extend(run(&mut m, 1205.0, 1210.0, still));
        ev.extend(run(&mut m, 1210.0, 1215.0, |t| sample(t, Some(true), 4.0)));
        let mv = movements(&ev);
        assert_eq!(mv.len(), 1, "{mv:?}");
        assert_eq!(mv[0].severity, Severity::Info);
        assert!(mv[0].evidence.contains("stays open"), "{}", mv[0].evidence);
        match mv[0].kind {
            EventKind::LongLieMovement { after_alert_s } => {
                assert!(
                    (after_alert_s - (1201.0 - 241.0)).abs() < 2.0,
                    "{after_alert_s}"
                )
            }
            _ => unreachable!(),
        }
        assert!(m.state().long_lie.is_none());
    }

    #[test]
    fn a_new_sudden_movement_after_the_movement_arms_a_new_count() {
        let mut m = SpaceMonitor::new(cfg(3.0));
        let mut ev = arrive_and_drop(&mut m);
        ev.extend(run(&mut m, 61.0, 300.0, still));
        ev.extend(run(&mut m, 300.0, 330.0, |t| sample(t, Some(true), 4.0)));
        ev.extend(run(&mut m, 330.0, 331.0, |t| sample(t, Some(true), 9.0)));
        ev.extend(run(&mut m, 331.0, 600.0, still));
        assert_eq!(lies(&ev).len(), 2);
        assert_eq!(movements(&ev).len(), 1);
    }

    #[test]
    fn a_data_gap_stops_a_count_but_not_an_alert() {
        let mut m = SpaceMonitor::new(cfg(3.0));
        let mut ev = arrive_and_drop(&mut m);
        ev.extend(run(&mut m, 61.0, 120.0, still));
        // 30 s with no samples at all.
        ev.extend(run(&mut m, 150.0, 400.0, still));
        assert!(lies(&ev).is_empty());

        let mut m = SpaceMonitor::new(cfg(3.0));
        let mut ev = arrive_and_drop(&mut m);
        ev.extend(run(&mut m, 61.0, 300.0, still));
        ev.extend(run(&mut m, 330.0, 400.0, still));
        assert_eq!(lies(&ev).len(), 1);
        assert!(m.state().long_lie.expect("still alerted").alerted);
    }
}
