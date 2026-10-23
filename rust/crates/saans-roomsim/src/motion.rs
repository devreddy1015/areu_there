//! Motion of spec/roomsim.md §9: walkable points and legs, where a person
//! starts, and the walk/pause trajectory of a walking person.

use crate::config::Config;
use crate::geometry::{dist2, edges, inside, inside_with_margin, seg_seg_dist, slab_meets, P2};
use crate::numerics::pymax;
use crate::room::RoomModel;
use saans_scene::ObjectKind;
use saans_sim::rng::Rng;

/// The walkability rules of §9 for one room and configuration.
#[derive(Debug, Clone)]
pub struct Walkability<'a> {
    room: &'a RoomModel,
    margin: f64,
    /// `rg = body_radius_m + object_clearance_m`.
    rg: f64,
}

impl<'a> Walkability<'a> {
    pub fn new(room: &'a RoomModel, cfg: &Config) -> Self {
        Self {
            room,
            margin: cfg.walk_margin_m,
            rg: cfg.body_radius_m + cfg.object_clearance_m,
        }
    }

    /// Walkable point: inside the footprint with `walk_margin_m`, and
    /// outside every grown non-opening object footprint.
    pub fn point(&self, p: P2) -> bool {
        if !inside_with_margin(&self.room.footprint, p, self.margin) {
            return false;
        }
        self.room
            .objects
            .iter()
            .filter(|o| !o.is_opening())
            .all(|o| {
                let (lx, ly) = o.to_local_xy(p);
                lx.abs() > o.sl / 2.0 + self.rg || ly.abs() > o.sb / 2.0 + self.rg
            })
    }

    /// Walkable leg `a -> b`: both ends walkable, at least `walk_margin_m`
    /// from every footprint edge, and missing every grown rectangle.
    pub fn leg(&self, a: P2, b: P2) -> bool {
        if !(self.point(a) && self.point(b)) {
            return false;
        }
        if !edges(&self.room.footprint).all(|(e0, e1)| seg_seg_dist(a, b, e0, e1) >= self.margin) {
            return false;
        }
        self.room
            .objects
            .iter()
            .filter(|o| !o.is_opening())
            .all(|o| {
                let la = o.to_local_xy(a);
                let lb = o.to_local_xy(b);
                !slab_meets(
                    &[la.0, la.1],
                    &[lb.0, lb.1],
                    &[o.sl / 2.0 + self.rg, o.sb / 2.0 + self.rg],
                )
            })
    }

    /// One random try of §9: `x = range(x0 + m, x1 - m)` then `y = range(y0 + m, y1 - m)`.
    fn draw(&self, rng: &mut Rng) -> P2 {
        let (x0, y0, x1, y1) = self.room.bbox;
        let m = self.margin;
        let x = rng.range(x0 + m, x1 - m);
        let y = rng.range(y0 + m, y1 - m);
        (x, y)
    }
}

/// The start of a person with no scene start (§9): inside the first door,
/// else a random walkable point from `rng` (stream `200 + i`). `None` when
/// 1000 tries find none.
pub fn entry_start(w: &Walkability<'_>, rng: &mut Rng) -> Option<P2> {
    if let Some(door) = w.room.objects.iter().find(|o| o.kind == ObjectKind::Door) {
        let along_x = door.sl < door.sb;
        let local = |d: f64| if along_x { (d, 0.0) } else { (0.0, d) };
        let into = [1.0, -1.0].into_iter().find(|&sgn| {
            let (lx, ly) = local(sgn * 0.5);
            let p = door.to_world(lx, ly, 0.0);
            inside(&w.room.footprint, (p.x, p.y))
        });
        if let Some(sgn) = into {
            for n in 1..=40 {
                let (lx, ly) = local(sgn * (0.05 * f64::from(n)));
                let p = door.to_world(lx, ly, 0.0);
                if w.point((p.x, p.y)) {
                    return Some((p.x, p.y));
                }
            }
        }
    }
    for _ in 0..1000 {
        let p = w.draw(rng);
        if w.point(p) {
            return Some(p);
        }
    }
    None
}

/// One trajectory segment (§9).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SegmentKind {
    Walk { from: P2, to: P2, speed: f64 },
    Pause { at: P2 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment {
    pub t0: f64,
    pub t1: f64,
    pub kind: SegmentKind,
    /// Heading during the segment: of the walk itself, or of the last walk
    /// before a pause (0.0 if none).
    pub heading: f64,
}

impl Segment {
    pub fn end(&self) -> P2 {
        match self.kind {
            SegmentKind::Walk { to, .. } => to,
            SegmentKind::Pause { at } => at,
        }
    }
}

/// Where a walking person is at one instant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionState {
    pub x: f64,
    pub y: f64,
    pub heading: f64,
    /// `s = 1`: on a walk segment.
    pub walking: bool,
    /// The walk's speed `v`, m/s; 0 on a pause.
    pub speed: f64,
}

/// The walk/pause trajectory of one walking person (§9), extended lazily
/// from stream `200 + i`. Times must be evaluated in nondecreasing order;
/// segments that end before the last evaluated time are dropped, so an
/// hours-long run holds a handful of segments.
#[derive(Debug, Clone)]
pub struct Trajectory {
    rng: Rng,
    start: P2,
    segments: std::collections::VecDeque<Segment>,
    last_walk_heading: f64,
    last_t: f64,
    /// Segments appended so far (kept count, for inspection).
    appended: u64,
}

impl Trajectory {
    /// `rng` is stream `200 + i` after any start draws.
    pub fn new(start: P2, rng: Rng) -> Self {
        Self {
            rng,
            start,
            segments: Default::default(),
            last_walk_heading: 0.0,
            last_t: f64::NEG_INFINITY,
            appended: 0,
        }
    }

    pub fn start(&self) -> P2 {
        self.start
    }

    /// Number of segments appended so far.
    pub fn segments_appended(&self) -> u64 {
        self.appended
    }

    fn push(&mut self, s: Segment) {
        self.segments.push_back(s);
        self.appended += 1;
    }

    fn extend(&mut self, w: &Walkability<'_>, cfg: &Config) {
        let (pos, t0) = match self.segments.back() {
            Some(s) => (s.end(), s.t1),
            None => (self.start, 0.0),
        };
        let mut accepted = None;
        for _ in 0..64 {
            let p = w.draw(&mut self.rng);
            if !w.point(p) {
                continue;
            }
            let d = dist2(pos, p);
            if d >= 0.5 && w.leg(pos, p) {
                accepted = Some((p, d));
                break;
            }
        }
        match accepted {
            Some((to, d)) => {
                let v = self.rng.range(cfg.speed_mps.0, cfg.speed_mps.1);
                let pause = self.rng.range(cfg.pause_s.0, cfg.pause_s.1);
                let heading = (to.1 - pos.1).atan2(to.0 - pos.0);
                let t1 = t0 + d / v;
                self.push(Segment {
                    t0,
                    t1,
                    kind: SegmentKind::Walk {
                        from: pos,
                        to,
                        speed: v,
                    },
                    heading,
                });
                self.last_walk_heading = heading;
                let p1 = t1 + pause;
                if p1 > t1 {
                    self.push(Segment {
                        t0: t1,
                        t1: p1,
                        kind: SegmentKind::Pause { at: to },
                        heading,
                    });
                }
            }
            None => {
                let pause = self.rng.range(cfg.pause_s.0, cfg.pause_s.1);
                let t1 = t0 + pymax(pause, 0.5);
                self.push(Segment {
                    t0,
                    t1,
                    kind: SegmentKind::Pause { at: pos },
                    heading: self.last_walk_heading,
                });
            }
        }
    }

    /// The segment containing `t` (`t0 <= t < t1`), extending first.
    ///
    /// # Panics
    /// If `t` is earlier than a time already evaluated.
    pub fn segment_at(&mut self, w: &Walkability<'_>, cfg: &Config, t: f64) -> Segment {
        assert!(
            t >= self.last_t,
            "trajectory times must be nondecreasing ({t} after {})",
            self.last_t
        );
        self.last_t = t;
        loop {
            match self.segments.back() {
                Some(s) if s.t1 > t => break,
                _ => self.extend(w, cfg),
            }
        }
        while self.segments.front().is_some_and(|s| s.t1 <= t) {
            self.segments.pop_front();
        }
        *self.segments.front().expect("a segment covers t")
    }

    /// Position, heading and walk flag at `t` (§9).
    pub fn at(&mut self, w: &Walkability<'_>, cfg: &Config, t: f64) -> MotionState {
        let s = self.segment_at(w, cfg, t);
        match s.kind {
            SegmentKind::Walk { from, to, speed } => {
                let f = (t - s.t0) / (s.t1 - s.t0);
                MotionState {
                    x: from.0 + (to.0 - from.0) * f,
                    y: from.1 + (to.1 - from.1) * f,
                    heading: s.heading,
                    walking: true,
                    speed,
                }
            }
            SegmentKind::Pause { at } => MotionState {
                x: at.0,
                y: at.1,
                heading: s.heading,
                walking: false,
                speed: 0.0,
            },
        }
    }
}
