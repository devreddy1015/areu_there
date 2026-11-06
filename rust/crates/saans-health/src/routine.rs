//! The daily routine across one home's rooms (issue #39,
//! `docs/ELDER_CARE.md` §3.2). Advisory: it describes a day and diagnoses
//! nothing.
//!
//! Per local day (`floor(t / DAY_S)`, the same index `DailySummary` uses):
//!
//! - **got up:** the end of a bedroom rest period in the morning window,
//!   confirmed by a motion bout somewhere in the home between 10 and 30
//!   minutes later. Getting up for the bathroom and going back to bed is not
//!   getting up.
//! - **active hours:** hours of the day with a motion bout in any room
//!   (`BOUT_INDEX`, as the per-space bout counter).
//! - **night bathroom visits:** bathroom occupancy sessions that start in
//!   the night window. A night belongs to the morning it ends in.
//! - **compared with the person's own baseline:** when a day closes, each
//!   value is set against the median of up to `ROUTINE_BASELINE_DAYS`
//!   previous complete days, flagged beyond the larger of
//!   `TREND_MAD_K * MAD_TO_SIGMA * MAD` and a fixed floor, the rule the
//!   breathing trend uses. Fewer than `ROUTINE_MIN_DAYS` previous days: no
//!   comparison.
//!
//! **Unusual quiet** is not computed. The spec's MVP cut (§7.2) leaves it
//! until there are 14 days of data, and it needs per-time-of-day baselines
//! this first version does not keep.
//!
//! Every threshold here is a starting value [proposal] for pilots to test,
//! not a measured one.

use crate::detectors::{BOUT_INDEX, DAY_S, HISTORY_MAX_DAYS, MAD_TO_SIGMA, TREND_MAD_K};
use crate::types::{EventKind, HealthEvent, Room, SpaceSample};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Rest periods that end in [MORNING_START_H, MORNING_END_H) local can be
/// getting up.
pub const MORNING_START_H: f64 = 3.0;
pub const MORNING_END_H: f64 = 12.0;
/// A rest end is getting up if a motion bout follows it after at least
/// this long...
pub const GOT_UP_ACTIVE_S: f64 = 600.0;
/// ...and no later than this.
pub const GOT_UP_WINDOW_S: f64 = 1800.0;
/// Previous complete days needed before a day is compared with them.
pub const ROUTINE_MIN_DAYS: usize = 7;
/// At most this many previous days form the baseline.
pub const ROUTINE_BASELINE_DAYS: usize = 28;
/// A day with fewer observed hours (some room visible) is incomplete: it is
/// neither compared nor used as a baseline.
pub const ROUTINE_MIN_OBSERVED_H: u32 = 20;
/// Smallest difference from the median that is ever called unusual.
pub const FLOOR_GOT_UP_MIN: f64 = 45.0;
pub const FLOOR_ACTIVE_HOURS: f64 = 2.0;
pub const FLOOR_NIGHT_BATHROOM: f64 = 1.0;

/// The home's settings for the routine.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RoutineConfig {
    /// Night window, local hours: bathroom visits from `night_start_h` to
    /// `night_end_h` (the next morning) are night visits.
    pub night_start_h: f64,
    pub night_end_h: f64,
}

impl Default for RoutineConfig {
    fn default() -> Self {
        Self {
            night_start_h: 22.0,
            night_end_h: 6.0,
        }
    }
}

/// One value set against the person's own previous days.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Comparison {
    /// `got_up_min`, `active_hours` or `night_bathroom_visits`.
    pub metric: String,
    pub value: f64,
    pub median: f64,
    pub mad: f64,
    /// Previous complete days in the baseline.
    pub days: usize,
    pub unusual: bool,
}

/// One home's day.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DailyRoutine {
    pub day: i64,
    /// Minutes after local midnight; `None` if no getting-up was seen.
    pub got_up_min: Option<f64>,
    /// Bit h: a motion bout somewhere in the home during local hour h.
    pub active_hours_mask: u32,
    pub night_bathroom_visits: u32,
    /// Bit h: some room was visible during local hour h.
    pub observed_hours_mask: u32,
    /// Filled when the day closes; empty for the day in progress.
    pub comparisons: Vec<Comparison>,
}

impl DailyRoutine {
    fn new(day: i64) -> Self {
        Self {
            day,
            got_up_min: None,
            active_hours_mask: 0,
            night_bathroom_visits: 0,
            observed_hours_mask: 0,
            comparisons: Vec::new(),
        }
    }

    pub fn active_hours(&self) -> u32 {
        self.active_hours_mask.count_ones()
    }

    pub fn observed_hours(&self) -> u32 {
        self.observed_hours_mask.count_ones()
    }

    pub fn complete(&self) -> bool {
        self.observed_hours() >= ROUTINE_MIN_OBSERVED_H
    }
}

/// The routine of one home: every space's samples and events, by room.
#[derive(Debug, Clone)]
pub struct HomeRoutine {
    cfg: RoutineConfig,
    today: Option<DailyRoutine>,
    /// Night bathroom visits after `night_start_h`: they belong to the
    /// next day.
    carry_bathroom: u32,
    /// A bedroom rest period that ended in the morning, awaiting a bout.
    got_up_candidate: Option<f64>,
    history: BTreeMap<i64, DailyRoutine>,
}

/// Local hour of day, fractional, for an anchored gateway time.
fn hour_of(t: f64) -> f64 {
    t.rem_euclid(DAY_S) / 3600.0
}

fn bit(t: f64) -> u32 {
    1 << (hour_of(t).floor() as u32).min(23)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let m = v.len() / 2;
    if v.len().is_multiple_of(2) {
        (v[m - 1] + v[m]) / 2.0
    } else {
        v[m]
    }
}

impl HomeRoutine {
    pub fn new(cfg: RoutineConfig) -> Self {
        Self {
            cfg,
            today: None,
            carry_bathroom: 0,
            got_up_candidate: None,
            history: BTreeMap::new(),
        }
    }

    /// The day in progress, without comparisons.
    pub fn today(&self) -> Option<&DailyRoutine> {
        self.today.as_ref()
    }

    /// Closed days, oldest first.
    pub fn history(&self) -> Vec<DailyRoutine> {
        self.history.values().cloned().collect()
    }

    /// Restore persisted days. Call once, at startup.
    pub fn import_history(&mut self, days: Vec<DailyRoutine>) {
        for d in days {
            self.history.insert(d.day, d);
        }
        self.prune();
    }

    /// One space's fused sample and the events it produced. Returns the day
    /// that closed, if this sample began a new one.
    pub fn observe(
        &mut self,
        room: Room,
        s: &SpaceSample,
        events: &[HealthEvent],
    ) -> Option<DailyRoutine> {
        let t = s.t_s;
        if !t.is_finite() {
            return None;
        }
        let d = (t / DAY_S).floor() as i64;
        let closed = if self.today.as_ref().is_some_and(|r| r.day >= d) {
            None
        } else {
            self.roll(d)
        };
        let today = self.today.as_mut()?;
        if s.occupied.is_some() {
            today.observed_hours_mask |= bit(t);
        }
        if self
            .got_up_candidate
            .is_some_and(|end| t - end > GOT_UP_WINDOW_S)
        {
            self.got_up_candidate = None;
        }
        if s.occupied.is_some() && s.motion_index > BOUT_INDEX {
            today.active_hours_mask |= bit(t);
            if let Some(end) = self.got_up_candidate {
                if t - end >= GOT_UP_ACTIVE_S && today.got_up_min.is_none() {
                    today.got_up_min = Some(hour_of(end) * 60.0);
                    self.got_up_candidate = None;
                }
            }
        }
        for ev in events {
            match ev.kind {
                EventKind::RestEnded { duration_min, .. } if room == Room::Bedroom => {
                    let end = ev.t_start_s.unwrap_or(ev.t_s) + duration_min * 60.0;
                    if (MORNING_START_H..MORNING_END_H).contains(&hour_of(end))
                        && (end / DAY_S).floor() as i64 == today.day
                        && today.got_up_min.is_none()
                    {
                        self.got_up_candidate = Some(end);
                    }
                }
                EventKind::OccupancyStarted if room == Room::Bathroom => {
                    let h = hour_of(ev.t_s);
                    if h >= self.cfg.night_start_h {
                        self.carry_bathroom += 1;
                    } else if h < self.cfg.night_end_h {
                        today.night_bathroom_visits += 1;
                    }
                }
                _ => {}
            }
        }
        closed
    }

    /// Close the day in progress (if any) and start day `d`.
    fn roll(&mut self, d: i64) -> Option<DailyRoutine> {
        let closed = self.today.take().map(|mut r| {
            r.comparisons = self.compare(&r);
            self.history.insert(r.day, r.clone());
            self.prune();
            r
        });
        let mut today = DailyRoutine::new(d);
        // Visits after the night began belong to the morning after; a gap
        // of more than a day drops them with the day they belonged to.
        if closed.as_ref().is_some_and(|c| c.day == d - 1) {
            today.night_bathroom_visits = self.carry_bathroom;
        }
        self.carry_bathroom = 0;
        self.got_up_candidate = None;
        self.today = Some(today);
        closed
    }

    fn prune(&mut self) {
        while self.history.len() > HISTORY_MAX_DAYS {
            self.history.pop_first();
        }
    }

    /// Each value against up to `ROUTINE_BASELINE_DAYS` previous complete
    /// days that have it.
    fn compare(&self, r: &DailyRoutine) -> Vec<Comparison> {
        if !r.complete() {
            return Vec::new();
        }
        type Get = fn(&DailyRoutine) -> Option<f64>;
        let metrics: [(&str, Get, f64); 3] = [
            ("got_up_min", |d| d.got_up_min, FLOOR_GOT_UP_MIN),
            (
                "active_hours",
                |d| Some(d.active_hours() as f64),
                FLOOR_ACTIVE_HOURS,
            ),
            (
                "night_bathroom_visits",
                |d| Some(d.night_bathroom_visits as f64),
                FLOOR_NIGHT_BATHROOM,
            ),
        ];
        let mut out = Vec::new();
        for (name, get, floor) in metrics {
            let Some(value) = get(r) else { continue };
            let mut prev: Vec<f64> = self
                .history
                .range(..r.day)
                .rev()
                .filter(|(_, d)| d.complete())
                .filter_map(|(_, d)| get(d))
                .take(ROUTINE_BASELINE_DAYS)
                .collect();
            if prev.len() < ROUTINE_MIN_DAYS {
                continue;
            }
            let med = median(&mut prev);
            let mut dev: Vec<f64> = prev.iter().map(|x| (x - med).abs()).collect();
            let mad = median(&mut dev);
            let limit = (TREND_MAD_K * MAD_TO_SIGMA * mad).max(floor);
            out.push(Comparison {
                metric: name.to_string(),
                value,
                median: med,
                mad,
                days: prev.len(),
                unusual: (value - med).abs() > limit,
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Severity;

    const H: f64 = 3600.0;

    fn sample(t: f64, motion: f64) -> SpaceSample {
        SpaceSample {
            t_s: t,
            occupied: Some(true),
            presence_index: 3.0,
            motion_index: motion,
            links_usable: 2,
            links_total: 2,
            breathing: None,
            heart: None,
        }
    }

    fn rest_ended(end: f64, minutes: f64) -> HealthEvent {
        HealthEvent {
            space_id: "bedroom".into(),
            t_s: end + 120.0,
            t_start_s: Some(end - minutes * 60.0),
            severity: Severity::Info,
            confidence: 0.8,
            evidence: String::new(),
            kind: EventKind::RestEnded {
                duration_min: minutes,
                motion_bouts: 0,
                mean_bpm: None,
                bpm_iqr: None,
                pauses: 0,
            },
        }
    }

    fn arrived(t: f64) -> HealthEvent {
        HealthEvent {
            space_id: "bathroom".into(),
            t_s: t,
            t_start_s: None,
            severity: Severity::Info,
            confidence: 0.9,
            evidence: String::new(),
            kind: EventKind::OccupancyStarted,
        }
    }

    fn still(r: &mut HomeRoutine, t: f64) -> Option<DailyRoutine> {
        r.observe(Room::Other, &sample(t, 0.5), &[])
    }

    fn bout(r: &mut HomeRoutine, t: f64) -> Option<DailyRoutine> {
        r.observe(Room::Other, &sample(t, 5.0), &[])
    }

    fn bedroom_rest_ends(r: &mut HomeRoutine, end: f64, minutes: f64) {
        r.observe(
            Room::Bedroom,
            &sample(end + 130.0, 0.5),
            &[rest_ended(end, minutes)],
        );
    }

    /// A whole observed day `d`: a sample every 10 minutes, the bedroom rest
    /// ending at `got_up_h` with a bout 15 minutes later, and a bout at each
    /// of `bouts` (hours). Returns any day that closed.
    fn day(r: &mut HomeRoutine, d: i64, got_up_h: Option<f64>, bouts: &[f64]) -> Vec<DailyRoutine> {
        let base = d as f64 * DAY_S;
        let mut closed: Vec<DailyRoutine> = (0..144)
            .filter_map(|i| still(r, base + i as f64 * 600.0))
            .collect();
        if let Some(h) = got_up_h {
            let end = base + h * H;
            bedroom_rest_ends(r, end, 420.0);
            closed.extend(bout(r, end + 15.0 * 60.0));
        }
        for &h in bouts {
            closed.extend(bout(r, base + h * H));
        }
        closed
    }

    #[test]
    fn active_hours_are_the_hours_with_a_bout_in_any_room() {
        let mut r = HomeRoutine::new(RoutineConfig::default());
        still(&mut r, 0.0);
        bout(&mut r, 7.2 * H);
        bout(&mut r, 7.9 * H);
        bout(&mut r, 13.5 * H);
        // Motion below BOUT_INDEX is not a bout.
        r.observe(Room::Other, &sample(15.0 * H, 2.0), &[]);
        let t = r.today().unwrap();
        assert_eq!(t.active_hours_mask, (1 << 7) | (1 << 13));
        assert_eq!(t.active_hours(), 2);
    }

    #[test]
    fn got_up_is_a_morning_rest_end_followed_by_activity() {
        let mut r = HomeRoutine::new(RoutineConfig::default());
        day(&mut r, 0, Some(6.5), &[]);
        assert_eq!(r.today().unwrap().got_up_min, Some(390.0));
    }

    #[test]
    fn back_to_bed_after_the_bathroom_is_not_getting_up() {
        let mut r = HomeRoutine::new(RoutineConfig::default());
        let end = 5.0 * H;
        bedroom_rest_ends(&mut r, end, 300.0);
        bout(&mut r, end + 3.0 * 60.0);
        for i in 0..40 {
            still(&mut r, end + 300.0 + i as f64 * 60.0);
        }
        // Activity long after the window does not count either.
        bout(&mut r, end + 45.0 * 60.0);
        assert_eq!(r.today().unwrap().got_up_min, None);
    }

    #[test]
    fn an_afternoon_nap_or_another_rooms_rest_is_not_getting_up() {
        let mut r = HomeRoutine::new(RoutineConfig::default());
        let nap = 14.0 * H;
        bedroom_rest_ends(&mut r, nap, 60.0);
        bout(&mut r, nap + 15.0 * 60.0);
        let morning = 7.0 * H;
        r.observe(
            Room::Other,
            &sample(morning + 130.0, 0.5),
            &[rest_ended(morning, 420.0)],
        );
        bout(&mut r, morning + 15.0 * 60.0);
        assert_eq!(r.today().unwrap().got_up_min, None);
    }

    #[test]
    fn a_night_belongs_to_the_morning_it_ends_in() {
        let mut r = HomeRoutine::new(RoutineConfig::default());
        still(&mut r, 0.0);
        let bath = |r: &mut HomeRoutine, t: f64| {
            r.observe(Room::Bathroom, &sample(t, 0.5), &[arrived(t)]);
        };
        bath(&mut r, 23.5 * H); // day 0 after 22:00: day 1's night
        bath(&mut r, DAY_S + 2.0 * H); // day 1 before 06:00
        bath(&mut r, DAY_S + 10.0 * H); // day 1 daytime: not a night visit
                                        // An arrival in a room that is not the bathroom does not count.
        let t = DAY_S + 3.0 * H;
        r.observe(Room::Other, &sample(t, 0.5), &[arrived(t)]);
        assert_eq!(r.today().unwrap().day, 1);
        assert_eq!(r.today().unwrap().night_bathroom_visits, 2);
        assert_eq!(r.history()[0].night_bathroom_visits, 0);
    }

    #[test]
    fn a_day_is_compared_with_the_persons_own_previous_days() {
        let mut r = HomeRoutine::new(RoutineConfig::default());
        let active = [8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5];
        for d in 0..8 {
            // Up at 06:30 give or take a few minutes.
            let up = 6.5 + (d % 3) as f64 * 0.05;
            day(&mut r, d, Some(up), &active);
        }
        // Day 8: up at 09:30, and one active hour fewer.
        day(&mut r, 8, Some(9.5), &active[1..]);
        let closed = day(&mut r, 9, None, &[]);
        assert_eq!(closed.len(), 1);
        let c = &closed[0];
        assert_eq!(c.day, 8);
        let get = |m: &str| c.comparisons.iter().find(|x| x.metric == m).cloned();
        let up = get("got_up_min").expect("compared");
        assert!(up.unusual, "{up:?}");
        assert_eq!(up.days, 8);
        let act = get("active_hours").expect("compared");
        assert!(!act.unusual, "one hour fewer is within the floor: {act:?}");
    }

    #[test]
    fn fewer_than_seven_previous_days_are_not_a_baseline() {
        let mut r = HomeRoutine::new(RoutineConfig::default());
        for d in 0..6 {
            day(&mut r, d, Some(6.5), &[9.5]);
        }
        day(&mut r, 6, Some(11.0), &[9.5]);
        let closed = day(&mut r, 7, None, &[]);
        assert!(
            closed[0].comparisons.is_empty(),
            "{:?}",
            closed[0].comparisons
        );
    }

    #[test]
    fn an_incomplete_day_is_neither_compared_nor_a_baseline() {
        let mut r = HomeRoutine::new(RoutineConfig::default());
        for d in 0..8 {
            day(&mut r, d, Some(6.5), &[9.5]);
        }
        // Day 8: the gateway saw only the first five hours.
        let base = 8.0 * DAY_S;
        for i in 0..30 {
            still(&mut r, base + i as f64 * 600.0);
        }
        let closed = day(&mut r, 9, Some(6.5), &[9.5]);
        assert!(!closed[0].complete());
        assert!(closed[0].comparisons.is_empty());
        // Day 9 is compared with the 8 complete days only.
        let closed = day(&mut r, 10, None, &[]);
        let up = closed[0]
            .comparisons
            .iter()
            .find(|x| x.metric == "got_up_min")
            .unwrap();
        assert_eq!(up.days, 8);
    }

    #[test]
    fn a_day_closes_once() {
        let mut r = HomeRoutine::new(RoutineConfig::default());
        let mut closed = day(&mut r, 0, None, &[9.5]);
        closed.extend(day(&mut r, 1, None, &[9.5]));
        closed.extend(day(&mut r, 2, None, &[9.5]));
        assert_eq!(closed.iter().map(|c| c.day).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(r.history().len(), 2);
    }
}
