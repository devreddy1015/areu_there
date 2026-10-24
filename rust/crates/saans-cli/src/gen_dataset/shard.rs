//! One scene's shard (spec/dataset.md §4-§6, §7.2) from its three Room
//! Sim runs (§3), all deterministic:
//! - the **occupied run** (`Simulator::new` on the scene and its config,
//!   `duration_s = S + 0.5`) fills the CSI stream (§4) and the labels (§5);
//!   it stops at the first transmission `n >= 80 S`, which is not stored;
//! - Locate's own **[`RoomSimSetup`]** (its calibration run, then the same
//!   occupied run again through `stream()`, advanced to the end: exactly
//!   `run_roomsim`) gives the baseline at Locate's ticks (§6);
//! - the **calibration run** once more (`calibration_config`, the same
//!   scene, config and seed with nobody in the room) gives the packets
//!   Locate calibrated from, `S + 0.5 <= t < S + 0.5 + calib_s`, on the
//!   §4.1 grid from superframe `calib_m0`; their count must equal the
//!   count Locate reports.
//!
//! Every index is an integer: transmission `n = m*N + slot` (Room Sim's
//! `Packet.m`, `Packet.slot`), column `m`, label tick `k` at transmission
//! `n = 40(k+1) - 1`. No floating-point time decides a column; the only
//! float comparisons are Room Sim's and Locate's own (`t < duration_s`,
//! `t >= duration_s`), made the same way here so the calibration packets
//! are the ones Locate counted. Everything here is SIM.

use saans_locate::phone::Config as LocateConfig;
use saans_locate::pipeline::{calibration_config, TickRecord};
use saans_locate::radius::{Outcome, Reason};
use saans_locate::stream::RoomSimSetup;
use saans_locate::LABEL;
use saans_roomsim::impair::{buffer_active_all, buffer_k_all};
use saans_roomsim::{Buffer, Config as RoomSimConfig, Posture, Simulator};
use saans_scene::Scene;
use serde_json::json;

use super::npz::{Array, Data};

/// Buffer entries stored per packet (§4.2): the first 56, what the
/// firmware's datagram carries at `fw_subcarriers = 56`.
pub const SUBCARRIERS: usize = 56;
/// People labelled per tick (§5), padded.
pub const MAX_PEOPLE: usize = 3;
/// Label ticks per second (§5, the manifest's `label_hz`).
pub const LABEL_HZ: u32 = 2;
/// The 3 x 3 zone grid (§5).
pub const ZONES: usize = 9;

/// §4.1's grid: the transmissions per label tick (40 at the default
/// `slot_s` 0.0125). The spec refuses a config where `2 / slot_s` is not
/// an integer; the label ticks every 0.5 s (§5) also need `0.5 / slot_s`
/// to be one, which implies it, so that is the check made.
pub fn tx_per_tick(rs: &RoomSimConfig) -> Result<u64, String> {
    let per_tick = 0.5 / rs.slot_s;
    if !(per_tick >= 1.0 && per_tick.fract() == 0.0 && per_tick <= 1e6) {
        return Err(format!(
            "slot_s {}: 0.5 / slot_s must be a whole number of transmissions (spec/dataset.md §4.1)",
            rs.slot_s
        ));
    }
    Ok(per_tick as u64)
}

/// A label tick's zone: `3*row + col`, `col = min(floor(3x/w), 2)`, `row =
/// min(floor(3y/d), 2)` (§5).
pub fn zone(x: f64, y: f64, w: f64, d: f64) -> usize {
    let cell = |v: f64, extent: f64| ((3.0 * v) / extent).floor().clamp(0.0, 2.0) as usize;
    3 * cell(y, d) + cell(x, w)
}

/// The rectangle an L's footprint cuts from its bounding box: from the
/// reflex vertex (the one strictly inside the box) to the box corner that
/// is not a vertex. `None` for the rectangle; an error for anything else.
fn cut_rect(scene: &Scene) -> Result<Option<(f64, f64, f64, f64)>, String> {
    let (w, d) = (scene.room.w_m, scene.room.d_m);
    let fp = scene.footprint();
    if fp.len() == 4 && scene.room.polygon.is_none() {
        return Ok(None);
    }
    let interior: Vec<_> = fp
        .iter()
        .filter(|p| p.0 > 0.0 && p.0 < w && p.1 > 0.0 && p.1 < d)
        .collect();
    let missing: Vec<_> = [(0.0, 0.0), (w, 0.0), (w, d), (0.0, d)]
        .into_iter()
        .filter(|c| !fp.contains(c))
        .collect();
    match (fp.len(), interior.as_slice(), missing.as_slice()) {
        (4, [], []) => Ok(None),
        (6, [r], [c]) => Ok(Some((
            r.0.min(c.0),
            r.1.min(c.1),
            r.0.max(c.0),
            r.1.max(c.1),
        ))),
        _ => Err(
            "the footprint is neither the room's rectangle nor an L (spec/dataset.md §2.1)".into(),
        ),
    }
}

/// §5 `zone_floor_frac`: per cell of the 3 x 3 grid over `(0,0)..(w,d)`,
/// the fraction of its area that is floor; for an L, one minus the cut
/// rectangle's share of the cell (exactly 0 for a cell the cut covers).
pub fn zone_floor_frac(scene: &Scene) -> Result<[f32; ZONES], String> {
    let (w, d) = (scene.room.w_m, scene.room.d_m);
    let mut out = [1.0f32; ZONES];
    let Some((x0, y0, x1, y1)) = cut_rect(scene)? else {
        return Ok(out);
    };
    let edge = |i: usize, extent: f64| {
        if i == 3 {
            extent
        } else {
            i as f64 * extent / 3.0
        }
    };
    for row in 0..3 {
        for col in 0..3 {
            let (cx0, cx1) = (edge(col, w), edge(col + 1, w));
            let (cy0, cy1) = (edge(row, d), edge(row + 1, d));
            let ox = (cx1.min(x1) - cx0.max(x0)).max(0.0);
            let oy = (cy1.min(y1) - cy0.max(y0)).max(0.0);
            out[3 * row + col] = (1.0 - (ox * oy) / ((cx1 - cx0) * (cy1 - cy0))) as f32;
        }
    }
    Ok(out)
}

/// §6's reason code: 0 estimate, 1 `no_disturbance`, 2 `not_significant`,
/// 3 `implausible_radius` (4, `warming_up`, is the tick before Locate's
/// first).
fn reason_code(o: &Outcome) -> Result<u8, String> {
    match o {
        Outcome::Estimate(_) => Ok(0),
        Outcome::Abstain(Reason::NoDisturbance) => Ok(1),
        Outcome::Abstain(Reason::NotSignificant) => Ok(2),
        Outcome::Abstain(Reason::ImplausibleRadius) => Ok(3),
        Outcome::Abstain(r) => Err(format!(
            "a person estimator abstained with the phone's reason {r}"
        )),
    }
}

/// `warming_up`, the label tick before Locate's first (§6).
pub const WARMING_UP: u8 = 4;

/// One scene's arrays and what the summary counts from them. SIM.
#[derive(Debug)]
pub struct Built {
    /// The §7.2 members, in order.
    pub arrays: Vec<Array>,
    /// Locate's config resolved for the run (the `.locate.json`).
    pub locate_config: LocateConfig,
    /// Label ticks, `K = 2S`.
    pub labels: usize,
    /// 2 s windows, `S/2`, and those whose end tick has a published WLM
    /// estimate.
    pub windows: usize,
    pub windows_wlm: usize,
    pub n_links: usize,
}

/// The labels of §5 at one tick, from Room Sim's truth.
struct Labels {
    xy: Vec<f32>,
    person_mask: Vec<u8>,
    walking: Vec<u8>,
    posture: Vec<u8>,
    zone: Vec<i8>,
    zone_occupied: Vec<u8>,
    zone_single: Vec<i8>,
    seen: Vec<bool>,
}

impl Labels {
    fn new(k: usize) -> Labels {
        Labels {
            xy: vec![f32::NAN; k * MAX_PEOPLE * 2],
            person_mask: vec![0; k * MAX_PEOPLE],
            walking: vec![0; k * MAX_PEOPLE],
            posture: vec![255; k * MAX_PEOPLE],
            zone: vec![-1; k * MAX_PEOPLE],
            zone_occupied: vec![0; k * ZONES],
            zone_single: vec![-1; k],
            seen: vec![false; k],
        }
    }

    fn set(&mut self, k: usize, people: &[saans_roomsim::PersonTruth], w: f64, d: f64) {
        for (i, p) in people.iter().enumerate() {
            let at = k * MAX_PEOPLE + i;
            self.xy[2 * at] = p.x as f32;
            self.xy[2 * at + 1] = p.y as f32;
            self.person_mask[at] = 1;
            self.walking[at] = u8::from(p.walking);
            self.posture[at] = match p.posture {
                Posture::Stand => 0,
                Posture::Sit => 1,
                Posture::Lie => 2,
            };
            let z = zone(p.x, p.y, w, d);
            self.zone[at] = z as i8;
            self.zone_occupied[k * ZONES + z] = 1;
        }
        if people.len() == 1 {
            self.zone_single[k] = self.zone[k * MAX_PEOPLE];
        }
        self.seen[k] = true;
    }
}

/// `v` as int8, which §4.2 says it is (an RSSI in -100..0).
fn as_i8(v: i32, what: &str) -> Result<i8, String> {
    i8::try_from(v).map_err(|_| format!("{what} {v} does not fit an int8"))
}

/// The occupied run (§4, §5): the stream and the labels.
struct Stream {
    csi_iq: Vec<i8>,
    csi_mask: Vec<u8>,
    rssi: Vec<i8>,
    columns: usize,
    labels: Labels,
}

/// Fill `[L, 56, M, 2]`, `[L, M]`, `[L, M]` at `(l, col)` from a received
/// packet's int8 buffer: `(imag, real)` as the datagram carries them.
fn put(
    iq: &mut [i8],
    mask: &mut [u8],
    rssi: &mut [i8],
    columns: usize,
    (l, col): (usize, usize),
    r: &saans_roomsim::Received,
) -> Result<(), String> {
    let Buffer::Int8(entries) = &r.buffer else {
        return Err("quantize is off: no int8 buffer (spec/dataset.md §3)".into());
    };
    if entries.len() < SUBCARRIERS {
        return Err(format!(
            "a buffer of {} entries, fewer than {SUBCARRIERS}",
            entries.len()
        ));
    }
    let cell = l * columns + col;
    if mask[cell] != 0 {
        return Err(format!("two packets for link {l} in superframe {col}"));
    }
    mask[cell] = 1;
    rssi[cell] = as_i8(r.rssi, "RSSI")?;
    for (s, &(re, im)) in entries[..SUBCARRIERS].iter().enumerate() {
        let at = ((l * SUBCARRIERS + s) * columns + col) * 2;
        iq[at] = im;
        iq[at + 1] = re;
    }
    Ok(())
}

fn occupied(
    sim: &Simulator,
    seconds: u32,
    per_tick: u64,
    w: f64,
    d: f64,
) -> Result<Stream, String> {
    let nt = sim.transmitters().len() as u64;
    let n_links = sim.links().len();
    let n_store = 2 * per_tick * u64::from(seconds);
    let columns = n_store.div_ceil(nt) as usize;
    let k_labels = 2 * seconds as usize;
    let mut s = Stream {
        csi_iq: vec![0; n_links * SUBCARRIERS * columns * 2],
        csi_mask: vec![0; n_links * columns],
        rssi: vec![0; n_links * columns],
        columns,
        labels: Labels::new(k_labels),
    };
    let slot_s = sim.config().slot_s;
    for tr in sim.run() {
        let n = tr.m * nt + tr.slot as u64;
        if n >= n_store {
            break;
        }
        if tr.t != n as f64 * slot_s {
            return Err(format!(
                "transmission {n} is at t = {}, not float(n) * slot_s",
                tr.t
            ));
        }
        // `% == 0`, not `is_multiple_of`: the workspace's rust-version is 1.80.
        #[allow(clippy::manual_is_multiple_of)]
        if (n + 1) % per_tick == 0 {
            let k = ((n + 1) / per_tick - 1) as usize;
            if tr.truth.people.len() > MAX_PEOPLE {
                return Err(format!(
                    "{} people; the labels hold at most {MAX_PEOPLE}",
                    tr.truth.people.len()
                ));
            }
            s.labels.set(k, &tr.truth.people, w, d);
        }
        for p in &tr.packets {
            if let Some(r) = &p.received {
                put(
                    &mut s.csi_iq,
                    &mut s.csi_mask,
                    &mut s.rssi,
                    columns,
                    (p.link_index, tr.m as usize),
                    r,
                )?;
            }
        }
    }
    if let Some(k) = s.labels.seen.iter().position(|&seen| !seen) {
        return Err(format!("no transmission for label tick {k}"));
    }
    Ok(s)
}

/// The first transmission `n` with `float(n) * slot_s >= t`, as Room Sim
/// times a transmission (§10) and as Locate and Room Sim compare it.
fn first_at_or_after(t: f64, slot_s: f64) -> u64 {
    let mut n = ((t / slot_s).floor() as u64).saturating_sub(2);
    while (n as f64 * slot_s) < t {
        n += 1;
    }
    n
}

/// The calibration packets Locate calibrated from (§6), on the §4.1 grid
/// from superframe `calib_m0`.
struct Calibration {
    iq: Vec<i8>,
    mask: Vec<u8>,
    rssi: Vec<i8>,
    m0: u64,
    columns: usize,
    packets: usize,
}

fn calibration(scene: &Scene, rs: &RoomSimConfig, calib_s: f64) -> Result<Calibration, String> {
    let cfg = calibration_config(rs, calib_s);
    let sim =
        Simulator::new(scene, cfg).map_err(|e| format!("the calibration run: Room Sim: {e}"))?;
    let nt = sim.transmitters().len() as u64;
    let slot_s = sim.config().slot_s;
    let n_first = first_at_or_after(rs.duration_s, slot_s);
    let n_end = first_at_or_after(sim.config().duration_s, slot_s);
    if n_end <= n_first {
        return Err("the calibration run has no transmission after duration_s".into());
    }
    let m0 = n_first / nt;
    let columns = ((n_end - 1) / nt - m0 + 1) as usize;
    let n_links = sim.links().len();
    let mut c = Calibration {
        iq: vec![0; n_links * SUBCARRIERS * columns * 2],
        mask: vec![0; n_links * columns],
        rssi: vec![0; n_links * columns],
        m0,
        columns,
        packets: 0,
    };
    // Locate feeds the packets with t >= duration_s (spec/locate.md §10.2
    // step 1); the same comparison here.
    for tr in sim.run().filter(|tr| tr.t >= rs.duration_s) {
        let col = (tr.m - m0) as usize;
        for p in &tr.packets {
            if let Some(r) = &p.received {
                put(
                    &mut c.iq,
                    &mut c.mask,
                    &mut c.rssi,
                    columns,
                    (p.link_index, col),
                    r,
                )?;
                c.packets += 1;
            }
        }
    }
    Ok(c)
}

/// §6 on the label grid: Locate's tick `j` (`t = 1 + j/2`) is label tick
/// `k = j + 1`; `k = 0` is `warming_up`.
struct Baseline {
    wlm_xy: Vec<f32>,
    rti_xy: Vec<f32>,
    wlm_radius: Vec<f32>,
    rti_radius: Vec<f32>,
    wlm_reason: Vec<u8>,
    rti_reason: Vec<u8>,
}

fn baseline(ticks: &[TickRecord], k_labels: usize) -> Result<Baseline, String> {
    if ticks.len() + 1 != k_labels {
        return Err(format!(
            "Locate gave {} ticks, not {}",
            ticks.len(),
            k_labels - 1
        ));
    }
    let mut b = Baseline {
        wlm_xy: vec![f32::NAN; 2 * k_labels],
        rti_xy: vec![f32::NAN; 2 * k_labels],
        wlm_radius: vec![f32::NAN; k_labels],
        rti_radius: vec![f32::NAN; k_labels],
        wlm_reason: vec![WARMING_UP; k_labels],
        rti_reason: vec![WARMING_UP; k_labels],
    };
    for (j, tick) in ticks.iter().enumerate() {
        let k = j + 1;
        if tick.t != 0.5 * (k + 1) as f64 {
            return Err(format!(
                "Locate's tick {j} is at t = {}, not label tick {k}'s",
                tick.t
            ));
        }
        for (o, xy, radius, reason) in [
            (
                &tick.wlm,
                &mut b.wlm_xy,
                &mut b.wlm_radius,
                &mut b.wlm_reason,
            ),
            (
                &tick.rti,
                &mut b.rti_xy,
                &mut b.rti_radius,
                &mut b.rti_reason,
            ),
        ] {
            reason[k] = reason_code(o)?;
            if let Some(e) = o.estimate() {
                xy[2 * k] = e.x as f32;
                xy[2 * k + 1] = e.y as f32;
                radius[k] = e.radius_m as f32;
            }
        }
    }
    Ok(b)
}

/// Build scene `index`'s §7.2 members from its three runs. `scene_sha256`
/// is the digest of the scene file's bytes (for `meta_json`). An error
/// names what failed; the caller names the scene.
pub fn build(
    scene: &Scene,
    rs: &RoomSimConfig,
    seconds: u32,
    index: usize,
    scene_id: &str,
    scene_sha256: &str,
) -> Result<Built, String> {
    let per_tick = tx_per_tick(rs)?;
    let (w, d) = (scene.room.w_m, scene.room.d_m);
    let k_labels = (LABEL_HZ * seconds) as usize;

    // The occupied run: §4 and §5.
    let sim = Simulator::new(scene, rs.clone()).map_err(|e| format!("Room Sim: {e}"))?;
    let links = sim.links();
    let n_links = links.len();
    let transmitters = sim.transmitters().to_vec();
    let stream = occupied(&sim, seconds, per_tick, w, d)?;
    drop(sim);

    // Locate, exactly run_roomsim: §6.
    let refused = |e: saans_locate::pipeline::PipelineError| format!("Locate: {e}");
    let Some(setup) = RoomSimSetup::new(scene, rs, &LocateConfig::default(), LABEL, &mut || true)
        .map_err(refused)?
    else {
        unreachable!("a calibration whose keep_going is always true is never abandoned");
    };
    let canonical: Vec<(i64, i64)> = links
        .iter()
        .map(|l| (i64::from(l.tx), i64::from(l.rx)))
        .collect();
    if setup.links() != canonical.as_slice() {
        return Err("Locate's canonical link order is not Room Sim's link_index order".into());
    }
    let ticks = setup
        .stream()
        .and_then(|mut s| s.advance_to(f64::INFINITY))
        .map_err(refused)?
        .ticks;
    let base = baseline(&ticks, k_labels)?;
    let summary = setup.calibration().clone();
    let locate_config = setup.config().clone();
    drop(setup);

    // The calibration run once more: the packets Locate calibrated from.
    let calib = calibration(scene, rs, locate_config.calib_s)?;
    if calib.packets != summary.packets {
        return Err(format!(
            "{} calibration packets stored, but Locate calibrated from {}",
            calib.packets, summary.packets
        ));
    }

    let n_nodes = scene.nodes.len();
    let n_people = scene.people.len();
    let columns = stream.columns;
    let windows = seconds as usize / 2;
    let window_end_tick: Vec<i32> = (0..windows).map(|w| 4 * w as i32 + 3).collect();
    let windows_wlm = window_end_tick
        .iter()
        .filter(|&&k| base.wlm_reason[k as usize] == 0)
        .count();
    let floor = zone_floor_frac(scene)?;
    let mut nodes = scene.nodes.clone();
    nodes.sort_by_key(|n| n.id);
    let footprint = scene.footprint();
    let meta = json!({
        "label": LABEL,
        "spec": super::SPEC,
        "scene_index": index,
        "scene_id": scene_id,
        "scene_sha256": scene_sha256,
        "roomsim_seed": rs.seed,
        "n_nodes": n_nodes,
        "n_people": n_people,
        "slot_s": rs.slot_s,
        "n_tx": transmitters.len(),
        "columns": columns,
        "calib_m0": calib.m0,
        "impair": rs.impair,
    });
    let k = k_labels;
    let p = MAX_PEOPLE;
    let l = n_links;
    let s = SUBCARRIERS;
    let buffer_k = buffer_k_all();
    let buffer_active = buffer_active_all();
    let labels = stream.labels;
    let arrays = vec![
        // §4.2
        Array::new("csi_iq", &[l, s, columns, 2], Data::I8(stream.csi_iq)),
        Array::new("csi_mask", &[l, columns], Data::U8(stream.csi_mask)),
        Array::new("rssi", &[l, columns], Data::I8(stream.rssi)),
        Array::new(
            "tx_slot",
            &[l],
            Data::U8(
                links
                    .iter()
                    .map(|lk| {
                        transmitters
                            .iter()
                            .position(|&t| t == lk.tx)
                            .expect("a link's tx transmits") as u8
                    })
                    .collect(),
            ),
        ),
        Array::new(
            "subcarrier_k",
            &[s],
            Data::I8(buffer_k[..s].iter().map(|&b| b as i8).collect()),
        ),
        Array::new(
            "subcarrier_active",
            &[s],
            Data::U8(buffer_active[..s].iter().map(|&b| u8::from(b)).collect()),
        ),
        // §5
        Array::new("count", &[k], Data::U8(vec![n_people as u8; k])),
        Array::new("presence", &[k], Data::U8(vec![u8::from(n_people > 0); k])),
        Array::new("xy", &[k, p, 2], Data::F32(labels.xy)),
        Array::new("person_mask", &[k, p], Data::U8(labels.person_mask)),
        Array::new("walking", &[k, p], Data::U8(labels.walking)),
        Array::new("posture", &[k, p], Data::U8(labels.posture)),
        Array::new("zone", &[k, p], Data::I8(labels.zone)),
        Array::new("zone_occupied", &[k, ZONES], Data::U8(labels.zone_occupied)),
        Array::new("zone_single", &[k], Data::I8(labels.zone_single)),
        Array::new(
            "t_label",
            &[k],
            Data::F64((0..k).map(|i| 0.5 * (i + 1) as f64).collect()),
        ),
        Array::new("zone_floor_frac", &[ZONES], Data::F32(floor.to_vec())),
        Array::new(
            "zone_reachable",
            &[ZONES],
            Data::U8(floor.iter().map(|&f| u8::from(f > 0.0)).collect()),
        ),
        // §6
        Array::new("wlm_xy", &[k, 2], Data::F32(base.wlm_xy)),
        Array::new("rti_xy", &[k, 2], Data::F32(base.rti_xy)),
        Array::new("wlm_radius", &[k], Data::F32(base.wlm_radius)),
        Array::new("rti_radius", &[k], Data::F32(base.rti_radius)),
        Array::new("wlm_reason", &[k], Data::U8(base.wlm_reason)),
        Array::new("rti_reason", &[k], Data::U8(base.rti_reason)),
        Array::new("calib_baseline", &[l], Data::F64(summary.baseline)),
        Array::new("calib_null_sigma", &[l], Data::F64(summary.null_sigma)),
        Array::new(
            "calib_mask",
            &[l],
            Data::U8(
                summary
                    .calibrated_mask
                    .iter()
                    .map(|&b| u8::from(b))
                    .collect(),
            ),
        ),
        Array::new("calib_iq", &[l, s, calib.columns, 2], Data::I8(calib.iq)),
        Array::new("calib_mask_m", &[l, calib.columns], Data::U8(calib.mask)),
        Array::new("calib_rssi", &[l, calib.columns], Data::I8(calib.rssi)),
        // §7.2's tail
        Array::new(
            "links",
            &[l, 2],
            Data::U8(links.iter().flat_map(|lk| [lk.tx, lk.rx]).collect()),
        ),
        Array::new(
            "node_ids",
            &[n_nodes],
            Data::U8(nodes.iter().map(|n| n.id).collect()),
        ),
        Array::new(
            "node_pos",
            &[n_nodes, 3],
            Data::F32(
                nodes
                    .iter()
                    .flat_map(|n| [n.pos.0 as f32, n.pos.1 as f32, n.pos.2 as f32])
                    .collect(),
            ),
        ),
        Array::new(
            "room_size",
            &[3],
            Data::F32(vec![
                scene.room.w_m as f32,
                scene.room.d_m as f32,
                scene.room.h_m as f32,
            ]),
        ),
        Array::new(
            "footprint",
            &[footprint.len(), 2],
            Data::F32(
                footprint
                    .iter()
                    .flat_map(|&(x, y)| [x as f32, y as f32])
                    .collect(),
            ),
        ),
        Array::new("window_end_tick", &[windows], Data::I32(window_end_tick)),
        {
            let bytes = meta.to_string().into_bytes();
            Array::new("meta_json", &[bytes.len()], Data::U8(bytes))
        },
    ];
    Ok(Built {
        arrays,
        locate_config,
        labels: k,
        windows,
        windows_wlm,
        n_links,
    })
}

/// The bytes a scene with `n_nodes` nodes takes at `seconds` (§8's size
/// estimate): its shard, laid out as [`build`] lays it out (the
/// calibration grid's columns bounded above), and its three small JSON
/// files.
pub fn estimate_bytes(n_nodes: usize, seconds: u32, calib_s: f64) -> u64 {
    let n = n_nodes as u64;
    let l = n * n.saturating_sub(1);
    let s = SUBCARRIERS as u64;
    let k = u64::from(LABEL_HZ * seconds);
    let m = (80 * u64::from(seconds)).div_ceil(n.max(1));
    let mc = ((80.0 * calib_s).ceil() as u64).div_ceil(n.max(1)) + 2;
    let p = MAX_PEOPLE as u64;
    let z = ZONES as u64;
    let data = l * s * m * 2
        + 2 * l * m
        + l
        + 2 * s
        + k * (1 + 1 + 8 * p + 4 * p + 1 + 8 + z)
        + 5 * z
        + k * (8 + 8 + 4 + 4 + 1 + 1)
        + l * (8 + 8 + 1)
        + l * s * mc * 2
        + 2 * l * mc
        + 2 * l
        + 13 * n
        + 12
        + 48
        + 4 * u64::from(seconds / 2)
        + 400;
    // 37 members: a padded npy header (at most 128 bytes) and the zip's
    // headers each; the end record; the scene and its two configs.
    data + 37 * (128 + 30 + 46 + 2 * 24) + 22 + 24_000
}

#[cfg(test)]
mod tests {
    use super::*;
    use saans_scene::{Node, Provenance, Room};

    fn room(w: f64, d: f64, polygon: Option<Vec<(f64, f64)>>) -> Scene {
        Scene {
            schema_version: 1,
            name: "t".into(),
            provenance: Provenance::Generated,
            room: Room {
                w_m: w,
                d_m: d,
                h_m: 3.0,
                polygon,
            },
            walls: vec![],
            objects: vec![],
            nodes: vec![Node {
                id: 0,
                pos: (1.0, 1.0, 1.0),
            }],
            people: vec![],
            phone: None,
        }
    }

    #[test]
    fn zones_are_three_by_three_with_the_far_edges_inside() {
        let (w, d) = (6.0, 3.0);
        assert_eq!(zone(0.0, 0.0, w, d), 0);
        assert_eq!(zone(1.99, 0.5, w, d), 0);
        assert_eq!(zone(2.0, 0.5, w, d), 1);
        assert_eq!(zone(5.0, 0.5, w, d), 2);
        assert_eq!(zone(6.0, 0.5, w, d), 2, "x = w is the last column");
        assert_eq!(zone(0.5, 1.0, w, d), 3);
        assert_eq!(zone(3.0, 1.5, w, d), 4);
        assert_eq!(zone(6.0, 3.0, w, d), 8, "the far corner is cell 8");
        assert_eq!(zone(0.1, 2.9, w, d), 6);
    }

    #[test]
    fn a_rectangle_is_all_floor() {
        assert_eq!(zone_floor_frac(&room(6.0, 3.0, None)).unwrap(), [1.0; 9]);
        // A polygon that is the rectangle is the rectangle.
        let rect = vec![(0.0, 0.0), (6.0, 0.0), (6.0, 3.0), (0.0, 3.0)];
        assert_eq!(
            zone_floor_frac(&room(6.0, 3.0, Some(rect))).unwrap(),
            [1.0; 9]
        );
    }

    /// Each L corner, worked by hand: on a 6 x 3 room (cells 2 x 1 m) a
    /// 3 x 1.5 m cut covers the corner cell whole (0 floor), half of the
    /// next cell along its row (1 of 2 m) and of the next along its column
    /// (0.5 of 1 m), and a quarter of the centre cell (0.75 floor).
    #[test]
    fn each_l_corner_cuts_its_cells_by_hand() {
        let (w, d) = (6.0, 3.0);
        let cut = |corner: usize| {
            let poly = crate::gen_dataset::generator::l_polygon(w, d, corner, 3.0, 1.5);
            zone_floor_frac(&room(w, d, Some(poly))).unwrap()
        };
        // Corner (0,0): cells 0 (gone), 1 (half), 3 (half), 4 (three quarters).
        assert_eq!(cut(0), [0.0, 0.5, 1.0, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]);
        // Corner (w,0): cells 2, 1, 5, 4.
        assert_eq!(cut(1), [1.0, 0.5, 0.0, 1.0, 0.75, 0.5, 1.0, 1.0, 1.0]);
        // Corner (w,d): cells 8, 7, 5, 4.
        assert_eq!(cut(2), [1.0, 1.0, 1.0, 1.0, 0.75, 0.5, 1.0, 0.5, 0.0]);
        // Corner (0,d): cells 6, 7, 3, 4.
        assert_eq!(cut(3), [1.0, 1.0, 1.0, 0.5, 0.75, 1.0, 0.0, 0.5, 1.0]);
        // A cut under a third each way touches only the corner cell.
        let poly = crate::gen_dataset::generator::l_polygon(w, d, 0, 1.0, 0.5);
        let f = zone_floor_frac(&room(w, d, Some(poly))).unwrap();
        assert_eq!(f, [0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn a_footprint_that_is_not_a_rectangle_or_an_l_is_refused() {
        let tri = vec![(0.0, 0.0), (6.0, 0.0), (0.0, 3.0), (3.0, 1.0)];
        assert!(zone_floor_frac(&room(6.0, 3.0, Some(tri))).is_err());
    }

    #[test]
    fn the_grid_is_forty_transmissions_per_label_tick() {
        assert_eq!(tx_per_tick(&RoomSimConfig::default()), Ok(40));
        let odd = RoomSimConfig {
            slot_s: 0.03,
            ..RoomSimConfig::default()
        };
        assert!(tx_per_tick(&odd).is_err());
    }

    /// §11 "no collision for N = 4..12 over 120 s": a generated scene of
    /// each node count, its full 120 s grid filled by [`occupied`], which
    /// refuses a second packet in any `(l, m)`. The physics is cut to the
    /// line of sight with nobody in the room (the grid does not depend on
    /// it), so this stays quick in the dev profile.
    #[test]
    fn every_node_count_fills_its_120_s_grid_without_a_collision() {
        use crate::gen_dataset::generator::generate;
        use rayon::prelude::*;
        let mut by_n: Vec<Option<(Scene, RoomSimConfig)>> = vec![None; 13];
        for i in 0.. {
            let g = generate(0, i, 120).unwrap();
            let n = g.scene.nodes.len();
            if by_n[n].is_none() {
                by_n[n] = Some((g.scene, g.rs));
            }
            if (4..=12).all(|n| by_n[n].is_some()) {
                break;
            }
        }
        (4..=12usize).into_par_iter().for_each(|n| {
            let (scene, rs) = by_n[n].clone().unwrap();
            let mut rs = calibration_config(&rs, 0.0);
            rs.reflection_order = 0;
            rs.floor_ceiling = false;
            rs.object_specular = false;
            rs.object_scatter = false;
            rs.object_blockage = false;
            rs.clutter_count = 0;
            rs.tail = false;
            let sim = Simulator::new(&scene, rs).unwrap();
            let s = occupied(&sim, 120, 40, scene.room.w_m, scene.room.d_m).unwrap();
            let columns = 9600usize.div_ceil(n);
            assert_eq!(s.columns, columns);
            let received = sim
                .packets()
                .take_while(|p| (p.m as usize * n + p.slot) < 9600)
                .filter(|p| p.received.is_some())
                .count();
            assert_eq!(
                s.csi_mask.iter().filter(|&&b| b == 1).count(),
                received,
                "N = {n}"
            );
            // A cell past n = 80 S is never filled: the last column holds
            // only the slots before it.
            for (li, lk) in sim.links().iter().enumerate() {
                let slot = sim.transmitters().iter().position(|&t| t == lk.tx).unwrap();
                if (columns - 1) * n + slot >= 9600 {
                    assert_eq!(s.csi_mask[li * columns + columns - 1], 0, "N = {n}");
                }
            }
        });
    }

    #[test]
    fn the_first_transmission_at_or_after_a_time_is_rooms_sims() {
        let slot = 0.0125;
        for t in [0.5, 4.5, 120.5, 135.5, 600.5] {
            let n = first_at_or_after(t, slot);
            assert!(n as f64 * slot >= t);
            assert!(((n - 1) as f64 * slot) < t);
        }
        assert_eq!(first_at_or_after(120.5, slot), 9640);
    }
}
