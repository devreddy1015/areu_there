//! The simulator: a compiled [`Plan`] rendered through per-link geometric
//! channels, one tick at a time.

use crate::physics::{BodyPoint, LinkChannel, BODY_RCS_M2};
use crate::plan::{BodyState, Bounds, Plan};
use crate::rng::Rng;
use crate::scenario::{
    Point, Point3, Scenario, SimOptions, Truth, TruthEvent, Vertical, FLOOR_CHEST_Z_M,
};
use saans_dsp::CsiFrame;

/// RSSI of a link whose mean |h| is the unattenuated-LoS 40.
pub const RSSI_REF_DBM: f64 = -45.0;

/// Upper bound on the cells in one [`Coverage`] grid.
///
/// The map exists to show Fresnel structure, whose finest feature is a
/// half wavelength: 6.2 cm at 2.437 GHz, so ~3 cm cells already sample it
/// at Nyquist. 100k cells is 1.7 cm resolution in a 6 m x 5 m room —
/// twice oversampled — and costs ~100k x n_subcarriers x body_points
/// flops, a fraction of a second. Anything finer buys no physics and
/// turns a GET into a stall, so it is rejected rather than served slowly.
pub const MAX_COVERAGE_CELLS: usize = 100_000;

/// A grid of ground-truth breathing sensitivity over the room for one
/// link, row-major with `index = iy * w + ix` and `iy` ascending with y.
/// SYNTHETIC, and an upper bound no receiver can measure: it is computed
/// from the scenario's own geometry, not from any CSI.
#[derive(Debug, Clone, PartialEq)]
pub struct Coverage {
    pub res_m: f64,
    pub w: usize,
    pub h: usize,
    /// Centre of cell (0, 0), i.e. `(res_m / 2, res_m / 2)`. Cell
    /// `(ix, iy)` is centred at `origin_m + (ix, iy) * res_m`.
    pub origin_m: Point,
    /// The raw (un-normalised) median this grid was divided by, in the
    /// [0, 1] units of [`Simulator::sensitivity`]. Each link's grid is
    /// normalised by its *own* median, so comparing levels across links
    /// means re-multiplying by this first.
    pub median_raw: f64,
    /// Relative breathing sensitivity; 1.0 is this room's median for this
    /// link. `values[iy * w + ix]`.
    pub values: Vec<f64>,
}

/// Everything produced at one tick: the truth at the tick time and the
/// frames that survived packet loss (in link order).
#[derive(Debug, Clone, PartialEq)]
pub struct Tick {
    pub truth: Truth,
    pub frames: Vec<LinkFrame>,
}

/// One received CSI packet. `seq` is the link's own u32 counter; it
/// advances for lost packets too, which is how loss is visible
/// downstream. `frame.sample_index` carries the same sequence number and
/// `frame.node_id` is the receiving node.
#[derive(Debug, Clone, PartialEq)]
pub struct LinkFrame {
    pub tx: u8,
    pub rx: u8,
    pub seq: u32,
    /// Sampling instant: tick time plus this packet's jitter.
    pub t_s: f64,
    pub frame: CsiFrame,
}

#[derive(Debug, Clone)]
struct LinkSim {
    tx: u8,
    rx: u8,
    ch: LinkChannel,
    rng: Rng,
    seq: u32,
}

/// Seeded, deterministic multi-link CSI simulator. SYNTHETIC by
/// construction.
#[derive(Debug, Clone)]
pub struct Simulator {
    sc: Scenario,
    opts: SimOptions,
    plan: Plan,
    links: Vec<LinkSim>,
    scatterers: Vec<(Point, f64)>,
    /// Their heights, in the same order; empty in the 2-D slice.
    scatterer_z: Vec<f64>,
    n_ticks: u64,
    tick: u64,
    /// Time of the last tick produced (0 before the first).
    now_s: f64,
    /// Walkable area (MARGIN_M from every wall); clamps a manual position.
    bounds: Bounds,
    /// When set, the body sits here instead of where the plan puts it.
    pos_override: Option<Point>,
    /// When false, there is no body at all: an empty room.
    present: bool,
    pts: Vec<BodyPoint>,
    re: Vec<f64>,
    im: Vec<f64>,
}

/// Apply the manual overrides to one instant of the plan's body.
///
/// The override moves the body *rigidly*: the arm keeps its offset from
/// the centre and whatever swing the plan gave it, so dragging the person
/// never fabricates limb motion. When not present there is no body at
/// all, and no chest displacement to go with it.
fn overridden(mut b: BodyState, pos_override: Option<Point>, present: bool) -> BodyState {
    if let Some(p) = pos_override {
        b.arm = (b.arm.0 + p.0 - b.centre.0, b.arm.1 + p.1 - b.centre.1);
        b.centre = p;
    }
    b.present &= present;
    if !b.present {
        b.disp = 0.0;
    }
    b
}

/// Chest height for a posture `lying` (0 upright .. 1 on the floor), and
/// whether the chest breathes vertically: only once fully on the floor.
fn posture(v: &Vertical, lying: f64) -> (f64, bool) {
    (
        v.chest_z_m + (FLOOR_CHEST_Z_M - v.chest_z_m) * lying,
        lying >= 1.0,
    )
}

/// The torso in its posture: in a box, at the posture's height and axis;
/// in the slice, as it always was.
fn posed_torso(
    ch: &LinkChannel,
    sc: &Scenario,
    centre: Point,
    lying: f64,
    pts: &mut Vec<BodyPoint>,
) {
    match &sc.room.vertical {
        None => ch.torso(centre, sc.body_points, sc.body_width_m, BODY_RCS_M2, pts),
        Some(v) => {
            let (z, flat) = posture(v, lying);
            ch.torso_posed(
                centre,
                z,
                flat,
                sc.body_points,
                sc.body_width_m,
                BODY_RCS_M2,
                pts,
            );
        }
    }
}

/// The arm at the chest's height in its posture.
fn posed_arm(ch: &LinkChannel, sc: &Scenario, arm: Point, lying: f64, rcs: f64) -> (f64, f64) {
    match &sc.room.vertical {
        None => ch.body_point_path(arm, rcs),
        Some(v) => ch.body_point_path_at(arm, posture(v, lying).0, rcs),
    }
}

// Stream ids for the seed-derived generators.
const STREAM_ROOM: u64 = 1;
const STREAM_PLAN: u64 = 2;
const STREAM_LINK0: u64 = 1000;

impl Simulator {
    /// Build a simulator with default [`SimOptions`]. Panics on an invalid
    /// scenario (see [`Simulator::try_new`]).
    pub fn new(s: &Scenario) -> Self {
        Self::with_options(s, SimOptions::default())
    }

    /// Build a simulator with explicit options. Panics on an invalid scenario.
    pub fn with_options(s: &Scenario, opts: SimOptions) -> Self {
        Self::try_with_options(s, opts)
            .unwrap_or_else(|e| panic!("invalid scenario {:?}: {e}", s.name))
    }

    /// Build a simulator, or say what is wrong with the scenario.
    pub fn try_new(s: &Scenario) -> Result<Self, String> {
        Self::try_with_options(s, SimOptions::default())
    }

    /// [`Simulator::try_new`] with explicit options.
    pub fn try_with_options(s: &Scenario, opts: SimOptions) -> Result<Self, String> {
        let (w, h) = s.room.size_m;
        if !(w > 0.0 && h > 0.0) {
            return Err(format!("room size {:?} must be positive", s.room.size_m));
        }
        if !(s.sample_rate_hz > 0.0 && s.sample_rate_hz.is_finite()) {
            return Err(format!(
                "sample_rate_hz {} must be positive",
                s.sample_rate_hz
            ));
        }
        if s.n_subcarriers == 0 {
            return Err("n_subcarriers must be at least 1".into());
        }
        if !(0.0..=1.0).contains(&s.loss_prob) {
            return Err(format!("loss_prob {} must be in [0, 1]", s.loss_prob));
        }
        if !(s.noise_std >= 0.0 && s.noise_std.is_finite()) {
            return Err(format!("noise_std {} must be >= 0", s.noise_std));
        }
        if !(opts.jitter_s >= 0.0 && opts.jitter_s.is_finite()) {
            return Err(format!("jitter_s {} must be >= 0", opts.jitter_s));
        }
        if !(opts.arm_rcs_m2 >= 0.0 && opts.carrier_hz > 0.0 && opts.subcarrier_spacing_hz > 0.0) {
            return Err(format!("invalid options {opts:?}"));
        }
        for n in &s.nodes {
            if !(n.pos.0 > 0.0 && n.pos.0 < w && n.pos.1 > 0.0 && n.pos.1 < h) {
                return Err(format!(
                    "node {} at {:?} is not inside room {:?}",
                    n.id, n.pos, s.room.size_m
                ));
            }
        }
        if let Some(v) = &s.room.vertical {
            let height = v.height_m;
            if !(height > 0.0 && height.is_finite()) {
                return Err(format!("room height {height} must be positive"));
            }
            let inside = |z: f64| z > 0.0 && z < height;
            if ![v.floor_reflection, v.ceiling_reflection]
                .iter()
                .all(|g| (0.0..=1.0).contains(g))
            {
                return Err(format!(
                    "floor and ceiling reflections {} and {} must be in [0, 1]",
                    v.floor_reflection, v.ceiling_reflection
                ));
            }
            if !inside(v.chest_z_m) {
                return Err(format!(
                    "chest height {} is not inside the room's {height} m",
                    v.chest_z_m
                ));
            }
            let (lo, hi) = v.scatterer_z_m;
            if !(lo >= 0.0 && lo <= hi && hi <= height) {
                return Err(format!(
                    "furniture heights {:?} must satisfy 0 <= low <= high <= {height} m",
                    v.scatterer_z_m
                ));
            }
            for n in &s.nodes {
                if !inside(n.z_m) {
                    return Err(format!(
                        "node {} at height {} is not inside the room's {height} m",
                        n.id, n.z_m
                    ));
                }
            }
            for sc in &s.room.scatterers {
                match sc.z_m {
                    Some(z) if (0.0..=height).contains(&z) => {}
                    Some(z) => {
                        return Err(format!(
                            "scatterer at {:?}: height {z} is not inside the room's {height} m",
                            sc.pos
                        ))
                    }
                    None => {
                        return Err(format!(
                            "scatterer at {:?} has no height, and this room has one",
                            sc.pos
                        ))
                    }
                }
            }
        }
        let node = |id: u8| -> Result<Point3, String> {
            s.nodes
                .iter()
                .find(|n| n.id == id)
                .map(|n| (n.pos.0, n.pos.1, n.z_m))
                .ok_or(format!("link references unknown node {id}"))
        };

        // Furniture: explicit first, then Python's rule for random ones.
        let mut scatterers: Vec<(Point, f64)> = s
            .room
            .scatterers
            .iter()
            .map(|sc| (sc.pos, sc.rcs_m2))
            .collect();
        let mut room_rng = Rng::stream(s.seed, STREAM_ROOM);
        let (lo, hi) = s.room.scatterer_rcs_m2;
        for _ in 0..s.room.n_scatterers {
            let p = (
                room_rng.range(0.05 * w, 0.95 * w),
                room_rng.range(0.05 * h, 0.95 * h),
            );
            let rcs = room_rng.range(lo, hi);
            scatterers.push((p, rcs));
        }
        // Heights after every horizontal draw, so a seed puts the same
        // furniture footprint in the slice and the box.
        let mut scatterer_z = Vec::new();
        if let Some(v) = &s.room.vertical {
            scatterer_z.extend(s.room.scatterers.iter().filter_map(|sc| sc.z_m));
            let (zlo, zhi) = v.scatterer_z_m;
            for _ in 0..s.room.n_scatterers {
                scatterer_z.push(room_rng.range(zlo, zhi));
            }
        }
        let placed: Vec<(Point3, f64)> = scatterers
            .iter()
            .enumerate()
            .map(|(i, &(p, rcs))| ((p.0, p.1, scatterer_z.get(i).copied().unwrap_or(0.0)), rcs))
            .collect();

        let mut links = Vec::with_capacity(s.links.len());
        for (i, &(tx, rx)) in s.links.iter().enumerate() {
            let (ptx, prx) = (node(tx)?, node(rx)?);
            let apart = match s.room.vertical {
                None => crate::physics::dist((ptx.0, ptx.1), (prx.0, prx.1)),
                Some(_) => crate::physics::dist3(ptx, prx),
            };
            if apart <= 0.0 {
                return Err(format!("link ({tx}, {rx}): nodes coincide"));
            }
            let ch = LinkChannel::new(
                &s.room,
                &placed,
                ptx,
                prx,
                opts.carrier_hz,
                opts.subcarrier_spacing_hz,
                s.n_subcarriers,
            );
            links.push(LinkSim {
                tx,
                rx,
                ch,
                rng: Rng::stream(s.seed, STREAM_LINK0 + i as u64),
                seq: 0,
            });
        }

        let plan = Plan::new(s, &mut Rng::stream(s.seed, STREAM_PLAN));
        let n_ticks = (plan.duration_s * s.sample_rate_hz - 1e-9).ceil().max(0.0) as u64;
        Ok(Self {
            sc: s.clone(),
            opts,
            plan,
            links,
            scatterers,
            scatterer_z,
            n_ticks,
            tick: 0,
            now_s: 0.0,
            bounds: Bounds::new(s.room.size_m),
            pos_override: None,
            present: true,
            pts: Vec::with_capacity(s.body_points.max(1)),
            re: vec![0.0; s.n_subcarriers],
            im: vec![0.0; s.n_subcarriers],
        })
    }

    /// Length of the timeline in seconds.
    pub fn duration_s(&self) -> f64 {
        self.plan.duration_s
    }

    /// Total ticks this run will produce (`ceil(duration * fs)`).
    pub fn n_ticks(&self) -> u64 {
        self.n_ticks
    }

    /// Ticks produced so far.
    pub fn ticks_done(&self) -> u64 {
        self.tick
    }

    pub fn scenario(&self) -> &Scenario {
        &self.sc
    }

    pub fn options(&self) -> &SimOptions {
        &self.opts
    }

    /// All furniture scatterers actually used, (position, RCS).
    pub fn scatterers(&self) -> &[(Point, f64)] {
        &self.scatterers
    }

    /// Their heights, in the same order as [`Simulator::scatterers`]; empty
    /// when the room is the 2-D slice.
    pub fn scatterer_z(&self) -> &[f64] {
        &self.scatterer_z
    }

    /// Advance one tick. `None` after the last tick.
    pub fn step(&mut self) -> Option<Tick> {
        if self.tick >= self.n_ticks {
            return None;
        }
        let fs = self.sc.sample_rate_hz;
        let t = self.tick as f64 / fs;
        let truth = self.truth_at(t);
        let Self {
            sc,
            opts,
            plan,
            links,
            pts,
            re,
            im,
            pos_override,
            present,
            ..
        } = self;
        let (pos_override, present) = (*pos_override, *present);
        let mut frames = Vec::with_capacity(links.len());
        for link in links.iter_mut() {
            // Always draw jitter and loss, so noise realisations do not
            // depend on either setting.
            let jitter = link.rng.uniform() * opts.jitter_s;
            let lost = link.rng.uniform() < sc.loss_prob;
            let seq = link.seq;
            link.seq = seq.wrapping_add(1);
            if lost {
                continue;
            }
            let tl = t + jitter;
            let body = overridden(plan.body_at(tl), pos_override, present);
            re.copy_from_slice(&link.ch.static_re);
            im.copy_from_slice(&link.ch.static_im);
            if body.present {
                posed_torso(&link.ch, sc, body.centre, body.lying, pts);
                for p in pts.iter() {
                    // Chest motion shortens/lengthens the path by gain * disp:
                    // exp(-j k L) * exp(+j k g disp) = exp(-j k (L - g disp)).
                    link.ch.add_path(re, im, p.amp, p.len - p.gain * body.disp);
                }
                if opts.arm_rcs_m2 > 0.0 {
                    let (amp, len) = posed_arm(&link.ch, sc, body.arm, body.lying, opts.arm_rcs_m2);
                    link.ch.add_path(re, im, amp, len);
                }
            }
            let n = re.len();
            let mut amplitudes = Vec::with_capacity(n);
            let mut phases = Vec::with_capacity(n);
            let sigma = sc.noise_std;
            let mut sum_amp = 0.0;
            for k in 0..n {
                let (mut hr, mut hi) = (re[k], im[k]);
                if sigma > 0.0 {
                    hr += sigma * link.rng.gauss();
                    hi += sigma * link.rng.gauss();
                }
                let a = (hr * hr + hi * hi).sqrt();
                sum_amp += a;
                amplitudes.push(a);
                phases.push(hi.atan2(hr));
            }
            let mean = (sum_amp / n as f64).max(1e-9);
            let rssi = (RSSI_REF_DBM + 20.0 * (mean / 40.0).log10() + link.rng.range(-1.0, 1.0))
                .round() as i32;
            frames.push(LinkFrame {
                tx: link.tx,
                rx: link.rx,
                seq,
                t_s: tl,
                frame: CsiFrame {
                    amplitudes,
                    phases,
                    sample_index: seq as u64,
                    sample_rate_hz: fs,
                    rssi_dbm: rssi,
                    node_id: link.rx,
                },
            });
        }
        self.now_s = t;
        self.tick += 1;
        Some(Tick { truth, frames })
    }

    /// Interval truths from the timeline: Occupied, BreathingPause, Fall,
    /// SitDown, Walking (merged) and Burst, sorted by start time. Known
    /// before the first tick.
    pub fn truth_events(&self) -> Vec<TruthEvent> {
        self.plan.events().to_vec()
    }

    /// Ground truth at an arbitrary time (what a tick at `t_s` would
    /// report), with any manual overrides applied.
    pub fn truth_at(&self, t_s: f64) -> Truth {
        let mut tr = self.plan.truth(t_s);
        if let Some(p) = self.pos_override {
            tr.position = p;
        }
        if !self.present {
            // An empty room: nothing is still, breathing, paused or
            // bursting, because there is nobody to be any of those.
            // `position` stays, like the plan's own `Absent` truth, as
            // where the person last was.
            tr.present = false;
            tr.still = false;
            tr.breathing = false;
            tr.bpm = 0.0;
            tr.in_pause = false;
            tr.in_burst = false;
        }
        tr
    }

    /// Pin the body at `p` for every following tick, ignoring the plan's
    /// scripted path; `None` hands it back to the plan. The point is
    /// clamped into the walkable area (`MARGIN_M` from every wall), which
    /// is what [`Simulator::position_override`] then reports.
    ///
    /// This does not touch the sequence counters, the RNG streams or the
    /// channel, so the CSI stays continuous across the call: a position
    /// that is still before and after is still *through* the change. It
    /// is a teleport, though — set it between ticks, or a single-tick jump
    /// will read downstream as motion.
    pub fn set_position_override(&mut self, p: Option<Point>) {
        self.pos_override = p.map(|p| self.bounds.clamp(p));
    }

    /// The manual position in force (already clamped), or `None` when the
    /// plan is driving.
    pub fn position_override(&self) -> Option<Point> {
        self.pos_override
    }

    /// Put a body in the room, or take it out entirely. With `false` the
    /// tick is the static channel plus noise — no torso, no arm, no chest
    /// motion — and the truth says `present: false`. Jitter, loss and
    /// noise draws are unchanged, so toggling this does not shift the
    /// random streams.
    pub fn set_present(&mut self, present: bool) {
        self.present = present;
    }

    /// Whether a body is in the room at all. False does not mean the plan
    /// says "absent": it means this simulator has been told there is
    /// nobody, which overrides the plan.
    pub fn present(&self) -> bool {
        self.present
    }

    /// Body centre in force at `t_s`: the manual override if set, else the
    /// plan's. Never derived from CSI.
    pub fn position_at(&self, t_s: f64) -> Point {
        self.pos_override.unwrap_or_else(|| self.plan.position(t_s))
    }

    /// Ground-truth first-order breathing sensitivity |d|h|/dx| per
    /// subcarrier, normalised to [0, 1] like the reference's
    /// `sensitivity()`, for the body at rest at its current position (the
    /// last tick's; t = 0 before the first). Computed whether or not the
    /// person is present. A receiver cannot measure this.
    ///
    /// # Panics
    /// If `link` is not one of the scenario's (tx, rx) links.
    pub fn sensitivity(&self, link: (u8, u8)) -> Vec<f64> {
        self.sensitivity_at(link, self.position_at(self.now_s))
    }

    /// [`Simulator::sensitivity`] for a body standing at `centre` instead
    /// of where it actually is: what breathing *would* do to this link
    /// from that point. The one physics path both methods use, so the two
    /// agree bit for bit when `centre` is the current position. The body
    /// keeps its current posture: in a box after a fall, this is a map for
    /// a chest on the floor.
    ///
    /// Independent of [`Simulator::present`]: this is a what-if map, not a
    /// statement that anyone is there.
    ///
    /// # Panics
    /// If `link` is not one of the scenario's (tx, rx) links.
    pub fn sensitivity_at(&self, link: (u8, u8), centre: Point) -> Vec<f64> {
        let l = self
            .links
            .iter()
            .find(|l| (l.tx, l.rx) == link)
            .unwrap_or_else(|| {
                panic!(
                    "sensitivity: link {link:?} is not in scenario {:?}",
                    self.sc.name
                )
            });
        let mut pts = Vec::with_capacity(self.sc.body_points.max(1));
        let lying = self.plan.lying(self.now_s);
        posed_torso(&l.ch, &self.sc, centre, lying, &mut pts);
        let extra: Vec<(f64, f64)> = if self.opts.arm_rcs_m2 > 0.0 {
            vec![posed_arm(
                &l.ch,
                &self.sc,
                self.plan.arm_rest(centre),
                lying,
                self.opts.arm_rcs_m2,
            )]
        } else {
            Vec::new()
        };
        l.ch.sensitivity(&pts, &extra)
    }

    /// Breathing sensitivity over the whole room for one link, on a
    /// `res_m` grid. See [`Coverage`]. Panics on a link that is not in the
    /// scenario or a `res_m` [`Simulator::try_coverage`] rejects.
    pub fn coverage(&self, link: (u8, u8), res_m: f64) -> Coverage {
        self.try_coverage(link, res_m)
            .unwrap_or_else(|e| panic!("coverage: {e}"))
    }

    /// [`Simulator::coverage`], saying what is wrong with `res_m` instead
    /// of panicking (a caller serving this over HTTP wants a 400, not an
    /// unwound thread). Still panics on an unknown link, like
    /// [`Simulator::sensitivity`].
    pub fn try_coverage(&self, link: (u8, u8), res_m: f64) -> Result<Coverage, String> {
        if !(res_m > 0.0 && res_m.is_finite()) {
            return Err(format!("res_m {res_m} must be finite and positive"));
        }
        let (room_w, room_h) = self.sc.room.size_m;
        let w = (room_w / res_m).ceil().max(1.0) as usize;
        let h = (room_h / res_m).ceil().max(1.0) as usize;
        let cells = w.saturating_mul(h);
        if cells > MAX_COVERAGE_CELLS {
            return Err(format!(
                "res_m {res_m} gives {w}x{h} = {cells} cells, over the {MAX_COVERAGE_CELLS} cap"
            ));
        }
        let origin_m = (res_m / 2.0, res_m / 2.0);

        // Summary across subcarriers: the MAXIMUM, not the mean.
        //
        // The DSP does not average the band, it picks the best of it:
        // `subcarrier_bnr` scores every subcarrier and `best_bnr` keeps
        // the largest (saans-dsp features.rs), and ADR-006 then fuses
        // links by taking the most confident one. A mean would call a
        // link blind when it is blind on 55 subcarriers and excellent on
        // one -- but that one is the subcarrier the pipeline would use,
        // so the person IS detectable there. The max matches what the
        // system can actually do with the cell, and it is the same
        // summary `examples/scan_seats.rs` already uses to call a seat a
        // whole-band blind spot.
        let mut raw: Vec<f64> = Vec::with_capacity(cells);
        for iy in 0..h {
            for ix in 0..w {
                let c = (
                    origin_m.0 + ix as f64 * res_m,
                    origin_m.1 + iy as f64 * res_m,
                );
                raw.push(self.sensitivity_at(link, c).into_iter().fold(0.0, f64::max));
            }
        }

        // Normalise by this room's median so the units are the relative
        // ones the UI advertises ("1.0 = this room's median"): the raw
        // [0, 1] number is a ratio to a best case nothing in a real room
        // reaches, which would paint every map uniformly dark. The median
        // is the robust choice -- a null or a hot fringe is a few percent
        // of cells and must not move the reference.
        let mut sorted: Vec<f64> = raw.iter().copied().filter(|v| v.is_finite()).collect();
        sorted.sort_by(f64::total_cmp);
        let median_raw = match sorted.len() {
            0 => 0.0,
            n if n % 2 == 1 => sorted[n / 2],
            n => 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]),
        };
        // A room whose median sensitivity is zero has no scale to be
        // relative to; serve the raw [0, 1] values rather than 0/0.
        let values = if median_raw > 0.0 {
            raw.iter().map(|v| v / median_raw).collect()
        } else {
            raw
        };
        Ok(Coverage {
            res_m,
            w,
            h,
            origin_m,
            median_raw,
            values,
        })
    }
}

impl Iterator for Simulator {
    type Item = Tick;
    fn next(&mut self) -> Option<Tick> {
        self.step()
    }
}
