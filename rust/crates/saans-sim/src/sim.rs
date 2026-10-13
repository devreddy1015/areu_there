//! The simulator: a compiled [`Plan`] rendered through per-link geometric
//! channels, one tick at a time.

use crate::physics::{point_path, BodyPoint, LinkChannel, BODY_RCS_M2};
use crate::plan::Plan;
use crate::rng::Rng;
use crate::scenario::{Point, Scenario, SimOptions, Truth, TruthEvent};
use saans_dsp::CsiFrame;

/// RSSI of a link whose mean |h| is the unattenuated-LoS 40.
pub const RSSI_REF_DBM: f64 = -45.0;

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
    n_ticks: u64,
    tick: u64,
    /// Time of the last tick produced (0 before the first).
    now_s: f64,
    pts: Vec<BodyPoint>,
    re: Vec<f64>,
    im: Vec<f64>,
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
        Self::try_with_options(s, opts).unwrap_or_else(|e| panic!("invalid scenario {:?}: {e}", s.name))
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
            return Err(format!("sample_rate_hz {} must be positive", s.sample_rate_hz));
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
                return Err(format!("node {} at {:?} is not inside room {:?}", n.id, n.pos, s.room.size_m));
            }
        }
        let node = |id: u8| s.nodes.iter().find(|n| n.id == id).map(|n| n.pos).ok_or(format!("link references unknown node {id}"));

        // Furniture: explicit first, then Python's rule for random ones.
        let mut scatterers: Vec<(Point, f64)> = s.room.scatterers.iter().map(|sc| (sc.pos, sc.rcs_m2)).collect();
        let mut room_rng = Rng::stream(s.seed, STREAM_ROOM);
        let (lo, hi) = s.room.scatterer_rcs_m2;
        for _ in 0..s.room.n_scatterers {
            let p = (room_rng.range(0.05 * w, 0.95 * w), room_rng.range(0.05 * h, 0.95 * h));
            let rcs = room_rng.range(lo, hi);
            scatterers.push((p, rcs));
        }

        let mut links = Vec::with_capacity(s.links.len());
        for (i, &(tx, rx)) in s.links.iter().enumerate() {
            let (ptx, prx) = (node(tx)?, node(rx)?);
            if crate::physics::dist(ptx, prx) <= 0.0 {
                return Err(format!("link ({tx}, {rx}): nodes coincide"));
            }
            let ch = LinkChannel::new(&s.room, &scatterers, ptx, prx, opts.carrier_hz, opts.subcarrier_spacing_hz, s.n_subcarriers);
            links.push(LinkSim { tx, rx, ch, rng: Rng::stream(s.seed, STREAM_LINK0 + i as u64), seq: 0 });
        }

        let plan = Plan::new(s, &mut Rng::stream(s.seed, STREAM_PLAN));
        let n_ticks = (plan.duration_s * s.sample_rate_hz - 1e-9).ceil().max(0.0) as u64;
        Ok(Self {
            sc: s.clone(),
            opts,
            plan,
            links,
            scatterers,
            n_ticks,
            tick: 0,
            now_s: 0.0,
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

    /// Advance one tick. `None` after the last tick.
    pub fn step(&mut self) -> Option<Tick> {
        if self.tick >= self.n_ticks {
            return None;
        }
        let fs = self.sc.sample_rate_hz;
        let t = self.tick as f64 / fs;
        let truth = self.plan.truth(t);
        let Self { sc, opts, plan, links, pts, re, im, .. } = self;
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
            let body = plan.body_at(tl);
            re.copy_from_slice(&link.ch.static_re);
            im.copy_from_slice(&link.ch.static_im);
            if body.present {
                link.ch.torso(body.centre, sc.body_points, sc.body_width_m, BODY_RCS_M2, pts);
                for p in pts.iter() {
                    // Chest motion shortens/lengthens the path by gain * disp:
                    // exp(-j k L) * exp(+j k g disp) = exp(-j k (L - g disp)).
                    link.ch.add_path(re, im, p.amp, p.len - p.gain * body.disp);
                }
                if opts.arm_rcs_m2 > 0.0 {
                    let (amp, len) = point_path(link.ch.tx, link.ch.rx, body.arm, opts.arm_rcs_m2);
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
            let rssi = (RSSI_REF_DBM + 20.0 * (mean / 40.0).log10() + link.rng.range(-1.0, 1.0)).round() as i32;
            frames.push(LinkFrame {
                tx: link.tx,
                rx: link.rx,
                seq,
                t_s: tl,
                frame: CsiFrame { amplitudes, phases, sample_index: seq as u64, sample_rate_hz: fs, rssi_dbm: rssi, node_id: link.rx },
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

    /// Ground truth at an arbitrary time (what a tick at `t_s` would report).
    pub fn truth_at(&self, t_s: f64) -> Truth {
        self.plan.truth(t_s)
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
        let l = self
            .links
            .iter()
            .find(|l| (l.tx, l.rx) == link)
            .unwrap_or_else(|| panic!("sensitivity: link {link:?} is not in scenario {:?}", self.sc.name));
        let centre = self.plan.position(self.now_s);
        let mut pts = Vec::with_capacity(self.sc.body_points.max(1));
        l.ch.torso(centre, self.sc.body_points, self.sc.body_width_m, BODY_RCS_M2, &mut pts);
        let extra: Vec<(f64, f64)> =
            if self.opts.arm_rcs_m2 > 0.0 { vec![point_path(l.ch.tx, l.ch.rx, self.plan.arm_rest(centre), self.opts.arm_rcs_m2)] } else { Vec::new() };
        l.ch.sensitivity(&pts, &extra)
    }
}

impl Iterator for Simulator {
    type Item = Tick;
    fn next(&mut self) -> Option<Tick> {
        self.step()
    }
}
