//! The simulator of spec/roomsim.md: static paths per link at
//! construction, then a streaming iterator over transmissions in §10
//! order, each with its truth record and one packet per receiver.

use crate::config::{Config, Mode, Posture};
use crate::datagram;
use crate::error::Error;
use crate::geometry::{P2, V3};
use crate::impair::{self, Device, PacketDraws, BUFFER_LEN};
use crate::motion::{MotionState, Segment, Trajectory, Walkability};
use crate::numerics::{bin_freq, bins, cexpj, cmul, cscale, theta, Cx, Neumaier};
use crate::paths::{draw_clutter, draw_tail, ClutterPoint, PathKind, StaticPath, World};
use crate::people::{
    body_point_rcs, body_point_world, cylinder_top, shadow_factor, BodyParams, Cylinder, Pose,
    BODY_POINTS,
};
use crate::room::{body_to_world, RoomModel, MAX_PEOPLE};
use saans_scene::{ObjectKind, Scene};
use saans_sim::rng::Rng;
use std::collections::BTreeMap;

/// Every output of this crate is labelled with this.
pub const LABEL: &str = "SIM";
/// The spec this crate implements.
pub const SPEC: &str = "roomsim v1";

/// A link `(tx, rx)` (§10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Link {
    /// Position in the list of links sorted by `(tx, rx)`.
    pub index: usize,
    pub tx: u8,
    pub rx: u8,
    /// Position of `{tx, rx}` in the sorted list of all node-id pairs (§6.8).
    pub pair_index: usize,
}

#[derive(Debug, Clone)]
struct LinkData {
    link: Link,
    tp: V3,
    rp: V3,
    /// Kinds 0..5 in §6 order, then the tail terms.
    paths: Vec<StaticPath>,
    n_static: usize,
    /// `cexpj(theta(f_k, L_p))` of the static paths, `[k * n_static + p]`.
    phasor: Vec<Cx>,
    /// `cmul(A_n, cexpj(theta(f_k, L_n)))` of the tail terms, `[k * n_tail + n]`.
    tail_terms: Vec<Cx>,
}

/// A person as the run sees them (§8.2, §9).
#[derive(Debug, Clone)]
pub struct PersonModel {
    pub index: usize,
    pub start: P2,
    /// Whether the start came from the scene (otherwise from §9's door or random draw).
    pub scene_start: bool,
    pub mode: Mode,
    pub posture: Posture,
    /// The first object whose (not grown) footprint contains the start, for
    /// a stationary person: what they sit or lie on.
    pub on_object: Option<usize>,
    /// Heading of a stationary person (`start_heading_rad`).
    pub heading0: f64,
    pub body: BodyParams,
    /// Stream `200 + i` after the start draws.
    rng: Rng,
}

impl PersonModel {
    /// The object excluded from this person's body-leg blockage (§6.9): the
    /// one a stationary sitting or lying person is on.
    pub fn resting_on(&self) -> Option<usize> {
        match (self.mode, self.posture) {
            (Mode::Stationary, Posture::Sit | Posture::Lie) => self.on_object,
            _ => None,
        }
    }
}

/// Every person at one instant: pose, cylinder and body-point positions.
#[derive(Debug, Clone, PartialEq)]
pub struct PeopleAt {
    pub t: f64,
    pub poses: Vec<Pose>,
    pub cylinders: Vec<Cylinder>,
    /// World position of each used body point, per person.
    pub points: Vec<Vec<V3>>,
}

/// Truth for one person at one transmission (§13).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PersonTruth {
    pub x: f64,
    pub y: f64,
    pub heading: f64,
    /// On a walk segment (`s = 1`).
    pub walking: bool,
    pub posture: Posture,
}

/// Truth record of one transmission `(m, i)`, lost or not (§13).
#[derive(Debug, Clone, PartialEq)]
pub struct Truth {
    pub t: f64,
    /// People in index order (named `people[i]`, never by a scene id).
    pub people: Vec<PersonTruth>,
    /// The phone `(x, y, z)`, or `None` when the scene has no phone.
    pub phone: Option<V3>,
}

/// The ESP32 buffer after step 9, or the floats when `quantize` is off.
#[derive(Debug, Clone, PartialEq)]
pub enum Buffer {
    /// `(re, im)` per entry, buffer order.
    Int8(Vec<(i8, i8)>),
    Float(Vec<Cx>),
}

/// What a receiver reports for a packet it received (§13).
#[derive(Debug, Clone, PartialEq)]
pub struct Received {
    /// Per receiving node, 0 first, `(seq + 1) mod 2^32` (§12.4).
    pub seq: u32,
    /// Clean channel `H` (§11), 64 bins ascending `k`.
    pub h: Vec<Cx>,
    pub rssi: i32,
    pub rssi_true: f64,
    /// `nf_dbm(rx)`.
    pub noise_floor: i32,
    /// 128 entries, buffer order, before quantisation.
    pub buffer_pre: Vec<Cx>,
    pub buffer: Buffer,
    /// The raw CSI datagram (§12.4) when `quantize` is on.
    pub datagram: Option<Vec<u8>>,
}

/// One packet record (§13), lost or not.
#[derive(Debug, Clone, PartialEq)]
pub struct Packet {
    pub t: f64,
    /// Superframe and slot of the transmission.
    pub m: u64,
    pub slot: usize,
    pub tx: u8,
    pub rx: u8,
    pub link_index: usize,
    pub lost: bool,
    /// The link's AGC state after this packet's update (§12.2).
    pub agc_db: f64,
    /// `None` when lost: a lost packet has no `H`, `seq`, RSSI or buffer (§13).
    pub received: Option<Received>,
}

/// One transmission `(m, i)`: its truth and a packet per receiver, in
/// receiver id order.
#[derive(Debug, Clone, PartialEq)]
pub struct Transmission {
    pub m: u64,
    pub slot: usize,
    pub t: f64,
    pub tx: u8,
    pub truth: Truth,
    pub packets: Vec<Packet>,
}

/// Room Sim v1 for one scene and configuration. Everything it produces is SIM.
#[derive(Debug, Clone)]
pub struct Simulator {
    cfg: Config,
    world: World,
    clutter: Vec<ClutterPoint>,
    freqs: Vec<f64>,
    transmitters: Vec<u8>,
    receivers: Vec<u8>,
    tx_period: Vec<u64>,
    links: Vec<LinkData>,
    link_lookup: BTreeMap<(u8, u8), usize>,
    devices: Vec<Device>,
    people: Vec<PersonModel>,
    rcs: Vec<f64>,
}

fn cfg_err(param: &str, reason: String) -> Error {
    Error::Config {
        param: param.to_string(),
        reason,
    }
}

impl Simulator {
    /// Validate the configuration and the scene (§3) and build every static
    /// path. The room comes from `scene` and nothing else.
    pub fn new(scene: &Scene, cfg: Config) -> Result<Self, Error> {
        cfg.validate()?;
        let room = RoomModel::from_scene(scene, &cfg)?;
        if room.people.len() > MAX_PEOPLE {
            return Err(cfg_err(
                "n_people",
                format!("{} people; at most {MAX_PEOPLE}", room.people.len()),
            ));
        }
        let ids: Vec<u8> = room.nodes.iter().map(|n| n.id).collect();
        cfg.validate_for_scene(&ids, room.people.len())?;
        let role = |id: u8| cfg.roles.get(&u32::from(id)).copied().unwrap_or_default();
        let transmitters: Vec<u8> = ids
            .iter()
            .copied()
            .filter(|&i| role(i).transmits())
            .collect();
        let receivers: Vec<u8> = ids
            .iter()
            .copied()
            .filter(|&i| role(i).receives())
            .collect();
        let tx_period = transmitters
            .iter()
            .map(|&i| cfg.tx_period.get(&u32::from(i)).copied().unwrap_or(1))
            .collect();

        let people = build_people(&room, &cfg)?;
        let world = World::new(room, &cfg);
        let clutter = draw_clutter(&cfg, &world.room);
        let freqs: Vec<f64> = bins()
            .map(|k| bin_freq(world.carrier_hz, cfg.spacing_hz, k))
            .collect();

        // Pairs of all node ids, a < b, sorted (§6.8).
        let mut pair_index = BTreeMap::new();
        for (i, &a) in ids.iter().enumerate() {
            for &b in &ids[i + 1..] {
                let n = pair_index.len();
                pair_index.insert((a, b), n);
            }
        }
        let tails: BTreeMap<usize, Vec<(f64, f64)>> = if cfg.tail {
            pair_index
                .values()
                .map(|&p| (p, draw_tail(&cfg, p)))
                .collect()
        } else {
            BTreeMap::new()
        };

        let mut links = Vec::new();
        let mut link_lookup = BTreeMap::new();
        for &tx in &transmitters {
            for &rx in &receivers {
                if tx == rx {
                    continue;
                }
                let index = links.len();
                let pair = pair_index[&(tx.min(rx), tx.max(rx))];
                let tp = world.room.nodes[world.room.node_index(tx).expect("known node")].p;
                let rp = world.room.nodes[world.room.node_index(rx).expect("known node")].p;
                let paths =
                    world.static_paths(&cfg, tp, rp, &clutter, tails.get(&pair).map(Vec::as_slice));
                let n_static = paths.iter().filter(|p| p.kind != PathKind::Tail).count();
                let n_tail = paths.len() - n_static;
                let mut phasor = Vec::with_capacity(64 * n_static);
                let mut tail_terms = Vec::with_capacity(64 * n_tail);
                for &f in &freqs {
                    for p in &paths[..n_static] {
                        phasor.push(cexpj(theta(f, p.length)));
                    }
                    for p in &paths[n_static..] {
                        tail_terms.push(cmul(p.coeff, cexpj(theta(f, p.length))));
                    }
                }
                link_lookup.insert((tx, rx), index);
                links.push(LinkData {
                    link: Link {
                        index,
                        tx,
                        rx,
                        pair_index: pair,
                    },
                    tp,
                    rp,
                    paths,
                    n_static,
                    phasor,
                    tail_terms,
                });
            }
        }

        let devices = world
            .room
            .nodes
            .iter()
            .map(|n| Device::draw(&cfg, n.id))
            .collect();
        let rcs = body_point_rcs(cfg.body_rcs_m2, cfg.body_points);
        Ok(Self {
            cfg,
            world,
            clutter,
            freqs,
            transmitters,
            receivers,
            tx_period,
            links,
            link_lookup,
            devices,
            people,
            rcs,
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// The scene as the simulator reads it (§3.1).
    pub fn room(&self) -> &RoomModel {
        &self.world.room
    }

    /// Surfaces, materials and the path formulas.
    pub fn world(&self) -> &World {
        &self.world
    }

    /// Kept clutter points, in draw order (§6.7).
    pub fn clutter(&self) -> &[ClutterPoint] {
        &self.clutter
    }

    /// Transmitters `T[0..nT)`, id order (§10).
    pub fn transmitters(&self) -> &[u8] {
        &self.transmitters
    }

    /// Receivers, id order (§10).
    pub fn receivers(&self) -> &[u8] {
        &self.receivers
    }

    /// Every link, in `link_index` order.
    pub fn links(&self) -> Vec<Link> {
        self.links.iter().map(|l| l.link).collect()
    }

    /// `link_index` of `(tx, rx)`.
    pub fn link_index(&self, tx: u8, rx: u8) -> Option<usize> {
        self.link_lookup.get(&(tx, rx)).copied()
    }

    /// The static paths of a link, in §6 order: kinds 0..5 with their §6.9
    /// coefficients, then the tail terms (kind 6).
    pub fn paths(&self, link_index: usize) -> &[StaticPath] {
        &self.links[link_index].paths
    }

    /// Per-link packet rate `1 / ((nT * slot_s) * tx_period[tx])`, Hz.
    pub fn per_link_rate_hz(&self, link_index: usize) -> f64 {
        let tx = self.links[link_index].link.tx;
        let i = self
            .transmitters
            .iter()
            .position(|&t| t == tx)
            .expect("a link's tx transmits");
        1.0 / ((self.transmitters.len() as f64 * self.cfg.slot_s) * self.tx_period[i] as f64)
    }

    /// Node `id`'s device draws (§12.1).
    pub fn device(&self, id: u8) -> Option<&Device> {
        self.world.room.node_index(id).map(|i| &self.devices[i])
    }

    /// People after §8.2 and §9's start rules.
    pub fn people(&self) -> &[PersonModel] {
        &self.people
    }

    /// The walkability rules of §9 for this room.
    pub fn walkability(&self) -> Walkability<'_> {
        Walkability::new(&self.world.room, &self.cfg)
    }

    /// A fresh trajectory for walking person `i` (§9), or `None` if they are
    /// stationary. Evaluate it with [`PersonTrack::at`] in nondecreasing time.
    pub fn track(&self, i: usize) -> Option<PersonTrack<'_>> {
        let p = self.people.get(i)?;
        if p.mode != Mode::Walk {
            return None;
        }
        Some(PersonTrack {
            sim: self,
            traj: Trajectory::new(p.start, p.rng.clone()),
        })
    }

    /// `buffer_k`: 128 ints `0..31, -32..-1, 0..31, -32..-1` (§13).
    pub fn buffer_k(&self) -> Vec<i32> {
        impair::buffer_k_all()
    }

    /// `buffer_active`: 128 bools (§13). Mask by index, never by value.
    pub fn buffer_active(&self) -> Vec<bool> {
        impair::buffer_active_all()
    }

    /// Every person's pose, cylinder and body points at `t` for `poses`
    /// (one per person, index order).
    pub fn people_at(&self, t: f64, poses: &[Pose]) -> PeopleAt {
        let room = &self.world.room;
        let mut cylinders = Vec::with_capacity(poses.len());
        let mut points = Vec::with_capacity(poses.len());
        for (p, pose) in self.people.iter().zip(poses) {
            let bed = match pose.posture {
                Posture::Lie => p.on_object.map(|o| &room.objects[o]),
                _ => None,
            };
            cylinders.push(Cylinder {
                c: (pose.x, pose.y),
                radius: self.cfg.body_radius_m,
                top: cylinder_top(&p.body, pose.posture, bed),
            });
            points.push(
                BODY_POINTS[..self.cfg.body_points]
                    .iter()
                    .map(|row| body_point_world(row, &p.body, pose, bed, t))
                    .collect(),
            );
        }
        PeopleAt {
            t,
            poses: poses.to_vec(),
            cylinders,
            points,
        }
    }

    /// The clean channel `H` of a link (§11), 64 bins ascending `k`.
    pub fn channel(&self, link_index: usize, people: &PeopleAt) -> Vec<Cx> {
        let cfg = &self.cfg;
        let ld = &self.links[link_index];
        let lambda = self.world.lambda;
        let shadow = cfg.shadowing && !people.cylinders.is_empty();
        let a_static: Vec<Cx> = ld.paths[..ld.n_static]
            .iter()
            .map(|p| {
                let s = if shadow {
                    shadow_factor(
                        &p.points,
                        &people.cylinders,
                        None,
                        lambda,
                        cfg.shadow_cap_db,
                    )
                } else {
                    1.0
                };
                cscale(p.coeff, s)
            })
            .collect();
        let mut body: Vec<(Cx, f64)> = Vec::new();
        for (q, pts) in people.points.iter().enumerate() {
            let skip_obj = self.people[q].resting_on();
            for (j, &bp) in pts.iter().enumerate() {
                let (a, l) = self.world.point_scatter(ld.tp, bp, ld.rp, self.rcs[j]);
                let a = self.world.leg_losses(a, ld.tp, bp, &[], skip_obj);
                let a = self.world.leg_losses(a, bp, ld.rp, &[], skip_obj);
                let s = if shadow {
                    shadow_factor(
                        &[ld.tp, bp, ld.rp],
                        &people.cylinders,
                        Some(q),
                        lambda,
                        cfg.shadow_cap_db,
                    )
                } else {
                    1.0
                };
                body.push((cscale(a, s), l));
            }
        }
        let n_tail = ld.paths.len() - ld.n_static;
        let mut h = Vec::with_capacity(64);
        for (ki, &f) in self.freqs.iter().enumerate() {
            let mut re = Neumaier::new();
            let mut im = Neumaier::new();
            let ph = &ld.phasor[ki * ld.n_static..(ki + 1) * ld.n_static];
            for (a, e) in a_static.iter().zip(ph) {
                let t = cmul(*a, *e);
                re.add(t.re);
                im.add(t.im);
            }
            for t in &ld.tail_terms[ki * n_tail..(ki + 1) * n_tail] {
                re.add(t.re);
                im.add(t.im);
            }
            for &(a, l) in &body {
                let t = cmul(a, cexpj(theta(f, l)));
                re.add(t.re);
                im.add(t.im);
            }
            h.push(Cx {
                re: re.value(),
                im: im.value(),
            });
        }
        h
    }

    /// The pose of a stationary person (§8.2): at the start, heading
    /// `start_heading_rad`, not walking.
    pub fn stationary_pose(&self, p: &PersonModel) -> Pose {
        Pose {
            x: p.start.0,
            y: p.start.1,
            heading: p.heading0,
            walking: false,
            speed: 0.0,
            posture: p.posture,
        }
    }

    /// The phone at one instant (§9): carried by `phone_carrier`, or at its
    /// scene start, z = 1.0. Truth only; it does not radiate.
    fn phone_at(&self, poses: &[Pose]) -> Option<V3> {
        let start = self.world.room.phone?;
        let Some(pose) = poses.get(self.cfg.phone_carrier) else {
            return Some(V3 {
                x: start.0,
                y: start.1,
                z: 1.0,
            });
        };
        let p = &self.people[self.cfg.phone_carrier];
        let (bx, by, bz) = self.cfg.phone_offset_m;
        Some(match pose.posture {
            Posture::Stand => body_to_world(pose.x, pose.y, pose.heading, bx, by, bz),
            Posture::Sit => body_to_world(pose.x, pose.y, pose.heading, bx, by, 0.72 * bz),
            Posture::Lie => {
                let w = body_to_world(pose.x, pose.y, pose.heading, bx, by, bz);
                let top = p
                    .on_object
                    .map(|o| self.world.room.objects[o].top())
                    .expect("a lying person lies on an object");
                V3 {
                    x: w.x,
                    y: w.y,
                    z: top + 0.05,
                }
            }
        })
    }

    /// A streaming run: transmissions in §10 order while `t < duration_s`.
    pub fn run(&self) -> Run<'_> {
        Run {
            sim: self,
            m: 0,
            slot: 0,
            done: false,
            tracks: self
                .people
                .iter()
                .map(|p| (p.mode == Mode::Walk).then(|| Trajectory::new(p.start, p.rng.clone())))
                .collect(),
            link_rng: (0..self.links.len())
                .map(|i| Rng::stream(self.cfg.seed, 1000 + i as u64))
                .collect(),
            agc: vec![self.cfg.agc_initial_db; self.links.len()],
            seq: vec![0; self.world.room.nodes.len()],
        }
    }

    /// The packets of a run, flattened, in §10 order.
    pub fn packets(&self) -> impl Iterator<Item = Packet> + '_ {
        self.run().flat_map(|t| t.packets)
    }
}

/// A walking person's trajectory with the rules it needs (§9).
#[derive(Debug, Clone)]
pub struct PersonTrack<'a> {
    sim: &'a Simulator,
    traj: Trajectory,
}

impl PersonTrack<'_> {
    /// Position, heading and walk flag at `t` (nondecreasing).
    pub fn at(&mut self, t: f64) -> MotionState {
        let w = self.sim.walkability();
        self.traj.at(&w, &self.sim.cfg, t)
    }

    /// The segment containing `t` (nondecreasing), extending as needed.
    pub fn segment(&mut self, t: f64) -> Segment {
        let w = self.sim.walkability();
        self.traj.segment_at(&w, &self.sim.cfg, t)
    }

    /// The trajectory itself.
    pub fn trajectory(&self) -> &Trajectory {
        &self.traj
    }
}

fn build_people(room: &RoomModel, cfg: &Config) -> Result<Vec<PersonModel>, Error> {
    let w = Walkability::new(room, cfg);
    let mut out = Vec::with_capacity(room.people.len());
    for (i, &start) in room.people.iter().enumerate() {
        let perr = |reason: String| Error::Person { index: i, reason };
        let body = BodyParams::draw(cfg, i);
        let mut rng = Rng::stream(cfg.seed, 200 + i as u64);
        let (start, scene_start) = match start {
            Some(p) => (p, true),
            None => (
                crate::motion::entry_start(&w, &mut rng)
                    .ok_or_else(|| perr("no walkable start found in 1000 tries".into()))?,
                false,
            ),
        };
        let walkable = w.point(start);
        let default_mode = if scene_start && !walkable {
            Mode::Stationary
        } else {
            Mode::Walk
        };
        let mode = cfg.people_mode.get(&i).copied().unwrap_or(default_mode);
        if mode == Mode::Walk && !walkable {
            return Err(perr(format!(
                "people_mode: set to walk but the start ({}, {}) is not walkable",
                start.0, start.1
            )));
        }
        let (posture, on_object) = match mode {
            Mode::Walk => {
                // A walking person stands; `stand` is accepted and changes nothing.
                if cfg
                    .people_posture
                    .get(&i)
                    .is_some_and(|&p| p != Posture::Stand)
                {
                    return Err(perr(
                        "people_posture: a walking person stands; sit or lie needs people_mode stationary"
                            .into(),
                    ));
                }
                (Posture::Stand, None)
            }
            Mode::Stationary => {
                let on = room
                    .objects
                    .iter()
                    .position(|o| o.footprint_contains(start));
                let on_kind = on.map(|o| room.objects[o].kind);
                let from_object = match on_kind {
                    Some(ObjectKind::Bed) => Posture::Lie,
                    Some(ObjectKind::Chair | ObjectKind::Sofa) => Posture::Sit,
                    _ => Posture::Stand,
                };
                let posture = cfg.people_posture.get(&i).copied().unwrap_or(from_object);
                if posture == Posture::Lie && on_kind != Some(ObjectKind::Bed) {
                    return Err(perr(
                        "people_posture: set to lie, but the first object whose footprint holds the start is not a bed"
                            .into(),
                    ));
                }
                (posture, on)
            }
        };
        let heading0 = cfg.start_heading_rad.get(&i).copied().unwrap_or(0.0);
        out.push(PersonModel {
            index: i,
            start,
            scene_start,
            mode,
            posture,
            on_object,
            heading0,
            body,
            rng,
        });
    }
    Ok(out)
}

/// The streaming iterator of a run: one [`Transmission`] per `(m, i)`
/// with `t < duration_s`. State is per run (trajectories, per-link RNG and
/// AGC, per-node sequence counters), so memory does not grow with time.
#[derive(Debug, Clone)]
pub struct Run<'a> {
    sim: &'a Simulator,
    m: u64,
    slot: usize,
    done: bool,
    tracks: Vec<Option<Trajectory>>,
    link_rng: Vec<Rng>,
    agc: Vec<f64>,
    seq: Vec<u32>,
}

impl Run<'_> {
    /// Every person's pose at `t` (advancing walkers' trajectories).
    fn poses_at(&mut self, t: f64) -> Vec<Pose> {
        let sim = self.sim;
        let w = sim.walkability();
        sim.people
            .iter()
            .zip(self.tracks.iter_mut())
            .map(|(p, tr)| match tr {
                None => sim.stationary_pose(p),
                Some(tr) => {
                    let s = tr.at(&w, &sim.cfg, t);
                    Pose {
                        x: s.x,
                        y: s.y,
                        heading: s.heading,
                        walking: s.walking,
                        speed: s.speed,
                        posture: Posture::Stand,
                    }
                }
            })
            .collect()
    }

    fn transmit(&mut self, m: u64, slot: usize, t: f64) -> Transmission {
        let sim = self.sim;
        let cfg = &sim.cfg;
        let tx = sim.transmitters[slot];
        let poses = self.poses_at(t);
        let truth = Truth {
            t,
            people: poses
                .iter()
                .map(|p| PersonTruth {
                    x: p.x,
                    y: p.y,
                    heading: p.heading,
                    walking: p.walking,
                    posture: p.posture,
                })
                .collect(),
            phone: sim.phone_at(&poses),
        };
        let people = sim.people_at(t, &poses);
        let tx_dev = sim.device(tx).expect("tx is a node");
        let mut packets = Vec::new();
        for &rx in &sim.receivers {
            if rx == tx {
                continue;
            }
            let li = sim.link_lookup[&(tx, rx)];
            let draws = PacketDraws::draw(&mut self.link_rng[li], cfg);
            self.agc[li] = impair::agc_step(self.agc[li], &draws, cfg);
            let p_loss = cfg
                .loss_prob_link
                .get(&format!("{tx},{rx}"))
                .copied()
                .unwrap_or(cfg.loss_prob);
            let lost = cfg.impair && draws.u_loss < p_loss;
            let received = if lost {
                None
            } else {
                let ri = sim.world.room.node_index(rx).expect("rx is a node");
                let seq = self.seq[ri];
                self.seq[ri] = seq.wrapping_add(1);
                let rx_dev = &sim.devices[ri];
                let h = sim.channel(li, &people);
                let pr = impair::process(&h, rx_dev, tx_dev, &draws, self.agc[li], cfg);
                let nf = rx_dev.nf_dbm as i32;
                let (buffer, datagram) = if cfg.quantize {
                    let q: Vec<(i8, i8)> = pr
                        .buffer_pre
                        .iter()
                        .map(|c| (impair::quantize(c.re), impair::quantize(c.im)))
                        .collect();
                    let d = datagram::serialize(
                        &q,
                        cfg.fw_subcarriers,
                        rx,
                        cfg.channel,
                        seq,
                        pr.rssi as i8,
                        nf as i8,
                    );
                    (Buffer::Int8(q), Some(d))
                } else {
                    (Buffer::Float(pr.buffer_pre.clone()), None)
                };
                debug_assert_eq!(pr.buffer_pre.len(), BUFFER_LEN);
                Some(Received {
                    seq,
                    h,
                    rssi: pr.rssi,
                    rssi_true: pr.rssi_true,
                    noise_floor: nf,
                    buffer_pre: pr.buffer_pre,
                    buffer,
                    datagram,
                })
            };
            packets.push(Packet {
                t,
                m,
                slot,
                tx,
                rx,
                link_index: li,
                lost,
                agc_db: self.agc[li],
                received,
            });
        }
        Transmission {
            m,
            slot,
            t,
            tx,
            truth,
            packets,
        }
    }
}

impl Iterator for Run<'_> {
    type Item = Transmission;

    fn next(&mut self) -> Option<Transmission> {
        let n_t = self.sim.transmitters.len() as u64;
        while !self.done {
            let (m, slot) = (self.m, self.slot);
            let t = (m * n_t + slot as u64) as f64 * self.sim.cfg.slot_s;
            // "packets with t < duration_s" (§3.2), written so that it reads so.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            let past_the_end = !(t < self.sim.cfg.duration_s);
            if past_the_end {
                self.done = true;
                break;
            }
            self.slot += 1;
            if self.slot as u64 == n_t {
                self.slot = 0;
                self.m += 1;
            }
            if m % self.sim.tx_period[slot] == 0 {
                return Some(self.transmit(m, slot, t));
            }
        }
        None
    }
}
