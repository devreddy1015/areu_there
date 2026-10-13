//! Timeline -> plan. The whole timeline is compiled once, up front, into
//! pieces that are analytic in time:
//!
//! - motion legs (hold, walk at constant speed, fast "drop" for falls and
//!   sitting down) that cover `[0, duration]` without gaps, so the body
//!   position is continuous;
//! - per-segment breathing rate (linear drift) with the phase integrated in
//!   closed form, so rate changes never make the chest jump;
//! - breathing-amplitude ramps around pauses;
//! - arm motions (fidget bursts, fall flailing).
//!
//! Because everything is a function of time, each link can be evaluated at
//! its own jittered sampling instant, and the truth intervals are known
//! before the first tick.

use crate::physics::dist;
use crate::rng::Rng;
use crate::scenario::{Activity, Point, Scenario, Truth, TruthEvent, TruthKind};
use std::f64::consts::{PI, TAU};

/// The body centre stays this far from every wall, which keeps the torso
/// (±0.15 m) and the swinging arm (0.3 + 0.05 m) strictly inside.
pub const MARGIN_M: f64 = 0.4;
/// Speed of the implicit walk to a still activity's `at`, and of the
/// approach before sitting down.
pub const RELOCATE_SPEED_MPS: f64 = 1.0;
pub const FALL_S: f64 = 0.8;
pub const SIT_DOWN_S: f64 = 0.6;
pub const SIT_DOWN_DROP_M: f64 = 0.4;
/// Breathing amplitude ramps to zero (and back) over this long.
pub const PAUSE_RAMP_S: f64 = 1.0;
/// Breathing rate before any segment has set one (walking in).
pub const DEFAULT_BPM: f64 = 16.0;
pub const BREATHING_AMPLITUDE_M: f64 = 0.005;
pub const HEART_AMPLITUDE_M: f64 = 0.0004;
pub const HEART_BPM: f64 = 68.0;
pub const ARM_OFFSET_M: f64 = 0.3;
pub const ARM_SWING_M: f64 = 0.05;
/// Raised-cosine taper at each end of an arm motion, so the arm never
/// jumps.
pub const ARM_TAPER_S: f64 = 0.25;
const WAYPOINT_MIN_STEP_M: f64 = 1.0;
const WAYPOINT_NODE_CLEARANCE_M: f64 = 0.5;
const MIN_SPEED_MPS: f64 = 0.05;
const MAX_SPEED_MPS: f64 = 10.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LegKind {
    Hold,
    Walk,
    Drop,
}

#[derive(Debug, Clone, Copy)]
struct Leg {
    t0: f64,
    t1: f64,
    from: Point,
    to: Point,
    kind: LegKind,
}

#[derive(Debug, Clone, Copy)]
struct SegPlan {
    t0: f64,
    t1: f64,
    present: bool,
    pause: bool,
    bpm0: f64,
    bpm1: f64,
    /// Breathing phase at `t0`.
    phase0: f64,
}

#[derive(Debug, Clone, Copy)]
struct ArmMove {
    t0: f64,
    t1: f64,
    freq_hz: f64,
    dir: Point,
    burst: bool,
}

#[derive(Debug, Clone, Copy)]
struct Pause {
    t0: f64,
    t1: f64,
    /// Breathing gain at t0 and t1.
    a0: f64,
    a1: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    pub x0: f64,
    pub x1: f64,
    pub y0: f64,
    pub y1: f64,
}

impl Bounds {
    pub fn new(size: Point) -> Self {
        let (w, h) = size;
        Self { x0: MARGIN_M.min(w / 2.0), x1: (w - MARGIN_M).max(w / 2.0), y0: MARGIN_M.min(h / 2.0), y1: (h - MARGIN_M).max(h / 2.0) }
    }
    pub fn clamp(&self, p: Point) -> Point {
        let cx = if p.0.is_finite() { p.0 } else { (self.x0 + self.x1) / 2.0 };
        let cy = if p.1.is_finite() { p.1 } else { (self.y0 + self.y1) / 2.0 };
        (cx.clamp(self.x0, self.x1), cy.clamp(self.y0, self.y1))
    }
    #[cfg(test)]
    pub fn contains(&self, p: Point) -> bool {
        p.0 >= self.x0 - 1e-9 && p.0 <= self.x1 + 1e-9 && p.1 >= self.y0 - 1e-9 && p.1 <= self.y1 + 1e-9
    }
    fn diag(&self) -> f64 {
        (self.x1 - self.x0).hypot(self.y1 - self.y0)
    }
}

/// Where the body is and what its chest is doing at one instant.
#[derive(Debug, Clone, Copy)]
pub struct BodyState {
    pub present: bool,
    pub centre: Point,
    /// Chest displacement along the bisector (m): breathing + heart.
    pub disp: f64,
    pub arm: Point,
}

#[derive(Debug, Clone)]
pub struct Plan {
    pub duration_s: f64,
    segs: Vec<SegPlan>,
    legs: Vec<Leg>,
    arms: Vec<ArmMove>,
    pauses: Vec<Pause>,
    events: Vec<TruthEvent>,
    arm_side: Point,
    rest: Point,
}

fn lerp(a: Point, b: Point, u: f64) -> Point {
    (a.0 + (b.0 - a.0) * u, a.1 + (b.1 - a.1) * u)
}

fn smoothstep(u: f64) -> f64 {
    u * u * (3.0 - 2.0 * u)
}

fn sane_speed(s: f64) -> f64 {
    if s.is_finite() {
        s.clamp(MIN_SPEED_MPS, MAX_SPEED_MPS)
    } else {
        RELOCATE_SPEED_MPS
    }
}

fn sane_bpm(b: f64) -> f64 {
    if b.is_finite() {
        b.max(0.0)
    } else {
        DEFAULT_BPM
    }
}

/// Breathing phase accumulated `tau` seconds into a segment of length `d`
/// whose rate drifts linearly from `b0` to `b1` BPM. With `b0 == b1` this
/// is exactly the reference's `2 * pi * (bpm / 60) * t`.
fn phase_delta(b0: f64, b1: f64, d: f64, tau: f64) -> f64 {
    let lin = 2.0 * PI * (b0 / 60.0) * tau;
    let quad = if d > 0.0 { 2.0 * PI * ((b1 - b0) / 60.0) * (tau * tau) / (2.0 * d) } else { 0.0 };
    lin + quad
}

/// Raised-cosine taper: 0 at both ends of `[0, d]`, 1 in between.
fn taper(tau: f64, d: f64) -> f64 {
    let r = ARM_TAPER_S.min(d / 2.0);
    if r <= 0.0 || tau <= 0.0 || tau >= d {
        return 0.0;
    }
    let edge = tau.min(d - tau);
    if edge >= r {
        1.0
    } else {
        0.5 * (1.0 - (PI * edge / r).cos())
    }
}

fn merge(mut iv: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    iv.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out: Vec<(f64, f64)> = Vec::with_capacity(iv.len());
    for (a, b) in iv {
        match out.last_mut() {
            Some(last) if a <= last.1 + 1e-9 => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
    out
}

struct Builder<'a> {
    bx: Bounds,
    nodes: Vec<Point>,
    rng: &'a mut Rng,
    legs: Vec<Leg>,
    arms: Vec<ArmMove>,
    events: Vec<TruthEvent>,
    pos: Point,
}

impl Builder<'_> {
    fn hold(&mut self, t0: f64, t1: f64) {
        if t1 > t0 {
            self.legs.push(Leg { t0, t1, from: self.pos, to: self.pos, kind: LegKind::Hold });
        }
    }

    /// Walk straight towards `target`; returns when it was reached (or
    /// `t1`, wherever the person then is).
    fn walk_to(&mut self, t0: f64, t1: f64, target: Point, speed: f64) -> f64 {
        let d = dist(self.pos, target);
        if d <= 1e-9 {
            self.pos = target;
            return t0;
        }
        if t1 <= t0 {
            return t0;
        }
        let dur = d / speed;
        if t0 + dur <= t1 {
            self.legs.push(Leg { t0, t1: t0 + dur, from: self.pos, to: target, kind: LegKind::Walk });
            self.pos = target;
            t0 + dur
        } else {
            let end = lerp(self.pos, target, (t1 - t0) / dur);
            self.legs.push(Leg { t0, t1, from: self.pos, to: end, kind: LegKind::Walk });
            self.pos = end;
            t1
        }
    }

    /// A fast move to `target` taking `dur` (compressed if the segment is
    /// shorter). Returns its end time.
    fn drop_to(&mut self, t0: f64, t1: f64, dur: f64, target: Point) -> f64 {
        let te = (t0 + dur).min(t1);
        if te > t0 {
            self.legs.push(Leg { t0, t1: te, from: self.pos, to: target, kind: LegKind::Drop });
        }
        self.pos = target;
        te
    }

    fn waypoint(&mut self) -> Point {
        let min_step = WAYPOINT_MIN_STEP_M.min(0.5 * self.bx.diag());
        let mut p = self.pos;
        for _ in 0..64 {
            p = (self.rng.range(self.bx.x0, self.bx.x1), self.rng.range(self.bx.y0, self.bx.y1));
            if dist(p, self.pos) >= min_step && self.nodes.iter().all(|&n| dist(p, n) >= WAYPOINT_NODE_CLEARANCE_M) {
                break;
            }
        }
        p
    }

    /// Random-waypoint walking for the whole interval.
    fn wander(&mut self, t0: f64, t1: f64, speed: f64) {
        let mut t = t0;
        while t < t1 {
            let wp = self.waypoint();
            let tr = self.walk_to(t, t1, wp, speed);
            if tr <= t {
                break;
            }
            t = tr;
        }
        self.hold(t, t1);
    }

    /// Get to a still activity's position; returns when stillness begins.
    fn settle(&mut self, present: bool, t0: f64, t1: f64, at: Point) -> f64 {
        let target = self.bx.clamp(at);
        if !present {
            self.pos = target;
            self.hold(t0, t1);
            return t0;
        }
        let tr = self.walk_to(t0, t1, target, RELOCATE_SPEED_MPS);
        self.hold(tr, t1);
        tr
    }

    /// Poisson burst starts (mean spacing `every_s`), dead time while a
    /// burst runs; only bursts that end inside the interval.
    fn bursts(&mut self, ts: f64, t1: f64, every_s: f64, burst_s: f64) {
        if !(every_s > 0.0 && burst_s > 0.0 && every_s.is_finite() && burst_s.is_finite()) {
            return;
        }
        let mut start = ts + self.rng.exp(every_s);
        while start + burst_s <= t1 {
            let end = start + burst_s;
            let freq_hz = self.rng.range(1.0, 2.0);
            let dir = self.rng.unit_vector();
            self.arms.push(ArmMove { t0: start, t1: end, freq_hz, dir, burst: true });
            self.events.push(TruthEvent { kind: TruthKind::Burst, t_start_s: start, t_end_s: end });
            let mut next = start + self.rng.exp(every_s);
            while next < end {
                next += self.rng.exp(every_s);
            }
            start = next;
        }
    }
}

impl Plan {
    pub fn new(sc: &Scenario, rng: &mut Rng) -> Self {
        let bx = Bounds::new(sc.room.size_m);
        let door = bx.clamp(sc.room.door_m);
        let arm_side = rng.unit_vector();
        let mut b = Builder { bx, nodes: sc.nodes.iter().map(|n| n.pos).collect(), rng, legs: Vec::new(), arms: Vec::new(), events: Vec::new(), pos: door };
        let mut segs: Vec<SegPlan> = Vec::with_capacity(sc.timeline.len());
        let (mut t, mut present, mut rate, mut phase) = (0.0f64, false, DEFAULT_BPM, 0.0f64);

        for seg in &sc.timeline {
            let d = seg.duration_s;
            if !(d > 0.0 && d.is_finite()) {
                continue;
            }
            let (t0, t1) = (t, t + d);
            let mut pause = false;
            let (bpm0, bpm1) = match &seg.activity {
                Activity::Absent => {
                    b.hold(t0, t1);
                    (rate, rate)
                }
                Activity::Walk { speed_mps } => {
                    if !present {
                        b.pos = door;
                    }
                    b.wander(t0, t1, sane_speed(*speed_mps));
                    (rate, rate)
                }
                Activity::WalkTo { to, speed_mps } => {
                    if !present {
                        b.pos = door;
                    }
                    let tr = b.walk_to(t0, t1, bx.clamp(*to), sane_speed(*speed_mps));
                    b.hold(tr, t1);
                    (rate, rate)
                }
                Activity::Seated { at, bpm, bpm_end } => {
                    b.settle(present, t0, t1, *at);
                    (sane_bpm(*bpm), sane_bpm(bpm_end.unwrap_or(*bpm)))
                }
                Activity::BreathingPause { at } => {
                    pause = true;
                    b.settle(present, t0, t1, *at);
                    (rate, rate)
                }
                Activity::Fidget { at, bpm, every_s, burst_s } => {
                    let ts = b.settle(present, t0, t1, *at);
                    b.bursts(ts, t1, *every_s, *burst_s);
                    (sane_bpm(*bpm), sane_bpm(*bpm))
                }
                Activity::Fall { to, bpm } => {
                    if !present {
                        b.pos = door;
                    }
                    let te = b.drop_to(t0, t1, FALL_S, bx.clamp(*to));
                    if te > t0 {
                        let freq_hz = b.rng.range(1.5, 2.5);
                        let dir = b.rng.unit_vector();
                        b.arms.push(ArmMove { t0, t1: te, freq_hz, dir, burst: false });
                    }
                    b.events.push(TruthEvent { kind: TruthKind::Fall, t_start_s: t0, t_end_s: te });
                    b.hold(te, t1);
                    (sane_bpm(*bpm), sane_bpm(*bpm))
                }
                Activity::SitDown { at, bpm } => {
                    if !present {
                        b.pos = door;
                    }
                    let target = bx.clamp(*at);
                    let d = dist(b.pos, target);
                    let mut ts = t0;
                    if d > SIT_DOWN_DROP_M {
                        let f = SIT_DOWN_DROP_M / d;
                        let approach = (target.0 + (b.pos.0 - target.0) * f, target.1 + (b.pos.1 - target.1) * f);
                        ts = b.walk_to(t0, t1, approach, RELOCATE_SPEED_MPS);
                    }
                    if ts < t1 {
                        let te = b.drop_to(ts, t1, SIT_DOWN_S, target);
                        b.events.push(TruthEvent { kind: TruthKind::SitDown, t_start_s: ts, t_end_s: te });
                        b.hold(te, t1);
                    }
                    (sane_bpm(*bpm), sane_bpm(*bpm))
                }
            };
            let now_present = !matches!(seg.activity, Activity::Absent);
            segs.push(SegPlan { t0, t1, present: now_present, pause, bpm0, bpm1, phase0: phase });
            phase = (phase + phase_delta(bpm0, bpm1, d, d)).rem_euclid(TAU);
            rate = bpm1;
            present = now_present;
            t = t1;
        }

        let Builder { legs, arms, mut events, .. } = b;
        for (a, z) in merge(segs.iter().filter(|s| s.present).map(|s| (s.t0, s.t1)).collect()) {
            events.push(TruthEvent { kind: TruthKind::Occupied, t_start_s: a, t_end_s: z });
        }
        let pause_iv = merge(segs.iter().filter(|s| s.pause).map(|s| (s.t0, s.t1)).collect());
        for &(a, z) in &pause_iv {
            events.push(TruthEvent { kind: TruthKind::BreathingPause, t_start_s: a, t_end_s: z });
        }
        for (a, z) in merge(legs.iter().filter(|l| l.kind == LegKind::Walk).map(|l| (l.t0, l.t1)).collect()) {
            events.push(TruthEvent { kind: TruthKind::Walking, t_start_s: a, t_end_s: z });
        }
        events.sort_by(|a, b| a.t_start_s.total_cmp(&b.t_start_s).then(a.kind.cmp(&b.kind)));

        let mut pauses = Vec::with_capacity(pause_iv.len());
        let mut prev: Option<(f64, f64)> = None;
        for (p0, p1) in pause_iv {
            let a0 = match prev {
                None => 1.0,
                Some((e, a)) => (a + (p0 - e) / PAUSE_RAMP_S).min(1.0),
            };
            let a1 = (a0 - (p1 - p0) / PAUSE_RAMP_S).max(0.0);
            pauses.push(Pause { t0: p0, t1: p1, a0, a1 });
            prev = Some((p1, a1));
        }

        Self { duration_s: t, segs, legs, arms, pauses, events, arm_side, rest: door }
    }

    pub fn events(&self) -> &[TruthEvent] {
        &self.events
    }

    fn seg_at(&self, t: f64) -> Option<&SegPlan> {
        if self.segs.is_empty() {
            return None;
        }
        let i = self.segs.partition_point(|s| s.t1 <= t).min(self.segs.len() - 1);
        Some(&self.segs[i])
    }

    fn leg_at(&self, t: f64) -> Option<&Leg> {
        if self.legs.is_empty() {
            return None;
        }
        let i = self.legs.partition_point(|l| l.t1 <= t).min(self.legs.len() - 1);
        Some(&self.legs[i])
    }

    fn arm_at(&self, t: f64) -> Option<&ArmMove> {
        let i = self.arms.partition_point(|a| a.t1 <= t);
        self.arms.get(i).filter(|a| a.t0 <= t)
    }

    /// Body centre at time t.
    pub fn position(&self, t: f64) -> Point {
        let Some(l) = self.leg_at(t) else { return self.rest };
        let span = l.t1 - l.t0;
        let u = if span > 0.0 { ((t - l.t0) / span).clamp(0.0, 1.0) } else { 1.0 };
        match l.kind {
            LegKind::Hold => l.from,
            LegKind::Walk => lerp(l.from, l.to, u),
            LegKind::Drop => lerp(l.from, l.to, smoothstep(u)),
        }
    }

    fn moving(&self, t: f64) -> bool {
        self.leg_at(t).is_some_and(|l| l.kind != LegKind::Hold && t >= l.t0 && t < l.t1)
    }

    /// Breathing amplitude gain in [0, 1] (pause ramps).
    fn breath_gain(&self, t: f64) -> f64 {
        let i = self.pauses.partition_point(|p| p.t0 <= t);
        if i == 0 {
            return 1.0;
        }
        let p = &self.pauses[i - 1];
        if t < p.t1 {
            (p.a0 - (t - p.t0) / PAUSE_RAMP_S).max(0.0)
        } else {
            (p.a1 + (t - p.t1) / PAUSE_RAMP_S).min(1.0)
        }
    }

    fn rate(s: &SegPlan, t: f64) -> f64 {
        let d = s.t1 - s.t0;
        let tau = (t - s.t0).clamp(0.0, d);
        if d > 0.0 {
            s.bpm0 + (s.bpm1 - s.bpm0) * tau / d
        } else {
            s.bpm0
        }
    }

    /// Rest position of the arm for a body centred at `centre`.
    pub fn arm_rest(&self, centre: Point) -> Point {
        (centre.0 + ARM_OFFSET_M * self.arm_side.0, centre.1 + ARM_OFFSET_M * self.arm_side.1)
    }

    pub fn body_at(&self, t: f64) -> BodyState {
        let centre = self.position(t);
        let (present, disp) = match self.seg_at(t) {
            None => (false, 0.0),
            Some(s) => {
                let phase_b = s.phase0 + phase_delta(s.bpm0, s.bpm1, s.t1 - s.t0, t - s.t0);
                let breath = BREATHING_AMPLITUDE_M * self.breath_gain(t);
                let disp = breath * phase_b.sin() + HEART_AMPLITUDE_M * (2.0 * PI * (HEART_BPM / 60.0) * t).sin();
                (s.present, disp)
            }
        };
        let mut arm = self.arm_rest(centre);
        if let Some(a) = self.arm_at(t) {
            let tau = t - a.t0;
            let swing = ARM_SWING_M * taper(tau, a.t1 - a.t0) * (2.0 * PI * a.freq_hz * tau).sin();
            arm = (arm.0 + swing * a.dir.0, arm.1 + swing * a.dir.1);
        }
        BodyState { present, centre, disp, arm }
    }

    pub fn truth(&self, t: f64) -> Truth {
        let seg = self.seg_at(t);
        let present = seg.is_some_and(|s| s.present);
        let in_pause = present && seg.is_some_and(|s| s.pause);
        let in_burst = present && self.arm_at(t).is_some_and(|a| a.burst);
        let still = present && !self.moving(t) && !in_burst;
        let breathing = present && !in_pause;
        let bpm = match seg {
            Some(s) if breathing => Self::rate(s, t),
            _ => 0.0,
        };
        Truth { t_s: t, present, still, breathing, bpm, in_pause, in_burst, position: self.position(t) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenarios;

    #[test]
    fn taper_is_continuous() {
        assert_eq!(taper(0.0, 3.0), 0.0);
        assert_eq!(taper(3.0, 3.0), 0.0);
        assert_eq!(taper(1.5, 3.0), 1.0);
        assert!((taper(0.125, 3.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn pause_gain_ramps_down_and_back() {
        let sc = scenarios::scenario(
            "pause",
            scenarios::room_clear_los(),
            scenarios::one_link(),
            vec![
                crate::scenario::Segment { duration_s: 10.0, activity: Activity::Seated { at: (3.0, 3.5), bpm: 15.0, bpm_end: None } },
                crate::scenario::Segment { duration_s: 5.0, activity: Activity::BreathingPause { at: (3.0, 3.5) } },
                crate::scenario::Segment { duration_s: 10.0, activity: Activity::Seated { at: (3.0, 3.5), bpm: 15.0, bpm_end: None } },
            ],
            1,
        );
        let plan = Plan::new(&sc, &mut Rng::new(1));
        assert_eq!(plan.breath_gain(9.99), 1.0);
        assert!((plan.breath_gain(10.5) - 0.5).abs() < 1e-12);
        assert_eq!(plan.breath_gain(12.0), 0.0);
        assert!((plan.breath_gain(15.5) - 0.5).abs() < 1e-12);
        assert_eq!(plan.breath_gain(16.0), 1.0);
    }

    #[test]
    fn arm_and_torso_stay_inside_through_falls_and_fidgets() {
        let room = scenarios::room_clear_los();
        let (w, h) = room.size_m;
        let mut tl = scenarios::collapse();
        tl.extend(scenarios::lone_worker(5.0));
        // A fall aimed outside the room is clamped.
        tl.push(crate::scenario::Segment { duration_s: 20.0, activity: Activity::Fall { to: (-3.0, 9.0), bpm: 15.0 } });
        let sc = scenarios::scenario("inside", room, scenarios::three_links(), tl, 3);
        let plan = Plan::new(&sc, &mut Rng::new(3));
        let mut t = 0.0;
        while t < plan.duration_s {
            let b = plan.body_at(t);
            for p in [b.centre, b.arm] {
                assert!(p.0 > 0.0 && p.0 < w && p.1 > 0.0 && p.1 < h, "t={t}: {p:?} outside");
            }
            assert!(Bounds::new((w, h)).contains(b.centre));
            t += 0.01;
        }
    }
}
