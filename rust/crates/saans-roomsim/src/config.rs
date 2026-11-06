//! Run configuration: every parameter of spec/roomsim.md §3.2 with its
//! default, and the §3.2 validation.
//!
//! Source of each default: **REAL** = the firmware or a published
//! standard; **SIM** = a modelling assumption, to be calibrated in
//! Phases 9-10 (the table in §3.2 says which is which).

use crate::error::Error;
use crate::materials::SurfaceMaterial;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A node's TDMA role (§3.2 `roles`, §10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Transmits in its slot and receives the others (the planned mesh).
    #[default]
    Txrx,
    Tx,
    Rx,
}

impl Role {
    pub fn transmits(self) -> bool {
        matches!(self, Self::Txrx | Self::Tx)
    }
    pub fn receives(self) -> bool {
        matches!(self, Self::Txrx | Self::Rx)
    }
}

/// Whether a person walks or stays put (§8.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Walk,
    Stationary,
}

/// A person's posture (§8.2, §8.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Posture {
    Stand,
    Sit,
    Lie,
}

impl Posture {
    pub fn name(self) -> &'static str {
        match self {
            Self::Stand => "stand",
            Self::Sit => "sit",
            Self::Lie => "lie",
        }
    }
}

/// Every §3.2 parameter. Read from JSON with [`Config::from_json_str`]:
/// absent keys take their default and unknown keys are an error.
///
/// Maps keyed by node id or person index are JSON objects with string keys
/// (`{"1": "tx"}`); `loss_prob_link` is keyed `"tx,rx"`. `(lo, hi)` pairs
/// are two-element arrays.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    // run
    pub seed: u64,
    /// Packets with `t < duration_s` are produced.
    pub duration_s: f64,
    // radio
    /// REAL. 1..13; `carrier_hz = (2412 + 5*(channel-1)) * 1e6`.
    pub channel: u32,
    /// REAL (802.11).
    pub spacing_hz: f64,
    /// SIM. Effective EIRP, isotropic antennas (ESP-IDF's default 20 dBm
    /// conducted, less PCB-antenna and enclosure losses at both ends).
    /// -100..60.
    pub tx_power_dbm: f64,
    // TDMA
    /// SIM: the planned broadcast TDMA (`edge/dsp/tdma.hpp`, itself a
    /// model: 4 nodes -> 20 Hz per link); today's firmware has no TDMA and
    /// one transmitter beaconing at 20 Hz, which `roles` with one `tx` and
    /// `slot_s = 0.05` models (spec/roomsim.md §10).
    pub slot_s: f64,
    /// node id -> role; absent ids are `txrx`.
    pub roles: BTreeMap<u32, Role>,
    /// node id -> transmit every `tx_period` superframes (default 1).
    pub tx_period: BTreeMap<u32, u64>,
    // surfaces
    /// 0, 1 or 2.
    pub reflection_order: u32,
    pub floor_ceiling: bool,
    /// SIM.
    pub floor_material: SurfaceMaterial,
    /// SIM.
    pub ceiling_material: SurfaceMaterial,
    /// SIM.
    pub slab_thickness_m: f64,
    pub transmission: bool,
    /// SIM. Material of the walls implied by a scene that lists none.
    pub implied_wall_material: SurfaceMaterial,
    /// SIM.
    pub implied_wall_thickness_m: f64,
    // objects
    pub object_specular: bool,
    pub object_scatter: bool,
    /// SIM.
    pub object_scatter_scale: f64,
    pub object_blockage: bool,
    // clutter
    /// SIM.
    pub clutter_count: u32,
    /// SIM, log-uniform.
    pub clutter_rcs_m2: (f64, f64),
    pub clutter_margin_m: f64,
    // tail
    pub tail: bool,
    /// SIM.
    pub tail_count: u32,
    /// SIM, relative to specular power. -100..30.
    pub tail_power_db: f64,
    /// SIM. The tail's power-delay profile decays as `exp(-tau / tail_decay_s)`.
    pub tail_decay_s: f64,
    // people
    /// The scene's count when unset; at most 100.
    pub n_people: Option<usize>,
    /// person index -> mode (default §8.2).
    pub people_mode: BTreeMap<usize, Mode>,
    /// person index -> posture of a stationary person (default §8.2).
    pub people_posture: BTreeMap<usize, Posture>,
    /// person index -> heading of a stationary person, radians (default 0.0).
    pub start_heading_rad: BTreeMap<usize, f64>,
    /// 6..16.
    pub body_points: usize,
    /// SIM.
    pub body_rcs_m2: f64,
    /// SIM.
    pub body_radius_m: f64,
    /// SIM.
    pub height_m: (f64, f64),
    pub breathing_hz: (f64, f64),
    pub breathing_p2p_m: (f64, f64),
    pub speed_mps: (f64, f64),
    pub pause_s: (f64, f64),
    pub walk_margin_m: f64,
    pub object_clearance_m: f64,
    pub shadowing: bool,
    /// SIM.
    pub shadow_cap_db: f64,
    // phone
    /// Index of the person carrying the phone.
    pub phone_carrier: usize,
    /// Body frame (+x forward, +y left, +z up), metres.
    pub phone_offset_m: (f64, f64, f64),
    // impairments
    /// Master switch (§12.5).
    pub impair: bool,
    /// SIM, no measured figure.
    pub loss_prob: f64,
    /// `"tx,rx"` -> probability.
    pub loss_prob_link: BTreeMap<String, f64>,
    /// SIM.
    pub agc_jump_prob: f64,
    /// SIM. 0..60.
    pub agc_step_db: f64,
    /// SIM.
    pub agc_up_prob: f64,
    /// -60..60.
    pub agc_initial_db: f64,
    /// SIM. 0..60.
    pub agc_max_db: f64,
    /// SIM. lo and hi in -60..60.
    pub device_gain_db: (f64, f64),
    /// Common phase error: with `false` the drawn `phi0` is not applied.
    pub cpe: bool,
    /// SIM, +/- per packet.
    pub timing_offset_s: f64,
    /// SIM, per node.
    pub timing_offset_dev_s: (f64, f64),
    pub ripple: bool,
    /// SIM.
    pub ripple_amp: f64,
    /// SIM.
    pub ripple_phase_rad: f64,
    pub ripple_harmonics: u32,
    /// `|ripple_period_bins| >= 1`.
    pub ripple_period_bins: f64,
    pub snr_noise: bool,
    /// SIM, per component.
    pub chip_noise_counts: f64,
    /// SIM.
    pub noise_floor_dbm: (f64, f64),
    /// SIM. 0..60.
    pub rssi_jitter_db: f64,
    pub rssi_range_dbm: (f64, f64),
    /// SIM. > 0.
    pub target_counts: f64,
    /// REAL (`ltf_merge_en = true`).
    pub ltf_merge: bool,
    /// SIM.
    pub ltf_merge_weight: f64,
    /// REAL: off (`channel_filter_en = false`, csi_capture.cpp).
    pub smoothing: bool,
    pub smoothing_kernel: (f64, f64, f64),
    pub smoothing_edge: (f64, f64),
    // output
    /// int8 buffer and a datagram; `false` keeps floats and emits no datagram.
    pub quantize: bool,
    /// REAL (Kconfig default); 8..64.
    pub fw_subcarriers: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 0,
            duration_s: 10.0,
            channel: 6,
            spacing_hz: 312500.0,
            tx_power_dbm: 10.0,
            slot_s: 0.0125,
            roles: BTreeMap::new(),
            tx_period: BTreeMap::new(),
            reflection_order: 2,
            floor_ceiling: true,
            floor_material: SurfaceMaterial::Concrete,
            ceiling_material: SurfaceMaterial::Concrete,
            slab_thickness_m: 0.2,
            transmission: true,
            implied_wall_material: SurfaceMaterial::Drywall,
            implied_wall_thickness_m: 0.1,
            object_specular: true,
            object_scatter: true,
            object_scatter_scale: 1.0,
            object_blockage: true,
            clutter_count: 24,
            clutter_rcs_m2: (0.005, 0.2),
            clutter_margin_m: 0.1,
            tail: true,
            tail_count: 40,
            tail_power_db: -15.0,
            tail_decay_s: 20e-9,
            n_people: None,
            people_mode: BTreeMap::new(),
            people_posture: BTreeMap::new(),
            start_heading_rad: BTreeMap::new(),
            body_points: 10,
            body_rcs_m2: 0.5,
            body_radius_m: 0.18,
            height_m: (1.55, 1.85),
            breathing_hz: (0.1, 0.3),
            breathing_p2p_m: (0.004, 0.012),
            speed_mps: (0.3, 1.2),
            pause_s: (0.0, 20.0),
            walk_margin_m: 0.4,
            object_clearance_m: 0.05,
            shadowing: true,
            shadow_cap_db: 20.0,
            phone_carrier: 0,
            phone_offset_m: (0.05, -0.25, 1.0),
            impair: true,
            loss_prob: 0.02,
            loss_prob_link: BTreeMap::new(),
            agc_jump_prob: 0.01,
            agc_step_db: 3.0,
            agc_up_prob: 0.5,
            agc_initial_db: 0.0,
            agc_max_db: 6.0,
            device_gain_db: (-2.0, 2.0),
            cpe: true,
            timing_offset_s: 50e-9,
            timing_offset_dev_s: (0.0, 0.0),
            ripple: true,
            ripple_amp: 0.05,
            ripple_phase_rad: 0.1,
            ripple_harmonics: 3,
            ripple_period_bins: 64.0,
            snr_noise: true,
            chip_noise_counts: 0.5,
            noise_floor_dbm: (-97.0, -92.0),
            rssi_jitter_db: 1.0,
            rssi_range_dbm: (-100.0, 0.0),
            target_counts: 24.0,
            ltf_merge: true,
            ltf_merge_weight: 0.5,
            smoothing: false,
            smoothing_kernel: (0.25, 0.5, 0.25),
            smoothing_edge: (2.0 / 3.0, 1.0 / 3.0),
            quantize: true,
            fw_subcarriers: 56,
        }
    }
}

/// The maps whose keys are integers (node ids or person indices), written
/// as JSON object keys (§3.2).
pub const INTEGER_KEYED_MAPS: [&str; 5] = [
    "roles",
    "tx_period",
    "people_mode",
    "people_posture",
    "start_heading_rad",
];

/// A canonical decimal integer key (§3.2): ASCII digits only, no sign,
/// spaces, underscores or leading zeros (`"0"` is the only key starting
/// with `0`).
pub fn is_canonical_decimal(k: &str) -> bool {
    !k.is_empty() && k.bytes().all(|b| b.is_ascii_digit()) && (k == "0" || !k.starts_with('0'))
}

fn bad(param: &str, reason: impl Into<String>) -> Error {
    Error::Config {
        param: param.to_string(),
        reason: reason.into(),
    }
}

impl Config {
    /// Parse a configuration from JSON (absent keys take their defaults) and
    /// validate it. An unknown key or a value of the wrong type is a §3.2
    /// rejection naming that parameter; text that is not a JSON object is
    /// [`Error::ConfigJson`].
    pub fn from_json_str(text: &str) -> Result<Self, Error> {
        let value: serde_json::Value =
            serde_json::from_str(text).map_err(|e| Error::ConfigJson(e.to_string()))?;
        let Some(map) = value.as_object() else {
            return Err(Error::ConfigJson("a configuration is a JSON object".into()));
        };
        // Integer map keys are canonical decimal (§3.2), whatever serde
        // would make of `"01"` or `"+1"`.
        for name in INTEGER_KEYED_MAPS {
            if let Some(m) = map.get(name).and_then(serde_json::Value::as_object) {
                if let Some(k) = m.keys().find(|k| !is_canonical_decimal(k)) {
                    return Err(bad(
                        name,
                        format!(
                            "key {k:?} is not a canonical decimal integer (ASCII digits only, no sign, spaces, underscores or leading zeros)"
                        ),
                    ));
                }
            }
        }
        let cfg: Config = match serde_json::from_value(value.clone()) {
            Ok(cfg) => cfg,
            Err(e) => {
                // Name the parameter: the first key that fails on its own.
                for (k, v) in map {
                    let one =
                        serde_json::Value::Object([(k.clone(), v.clone())].into_iter().collect());
                    if let Err(e) = serde_json::from_value::<Config>(one) {
                        return Err(bad(k, e.to_string()));
                    }
                }
                return Err(Error::ConfigJson(e.to_string()));
            }
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// `carrier_hz = (2412 + 5*(channel-1)) * 1e6`. Valid after [`Config::validate`].
    pub fn carrier_hz(&self) -> f64 {
        crate::numerics::carrier_hz(self.channel)
    }

    /// The §3.2 checks that need no scene, in the reference's order. The
    /// scene-dependent ones are [`Config::validate_for_scene`], which
    /// `Simulator::new` makes.
    // `!(x >= 0.0)` rather than `x < 0.0` and so on: a NaN must be rejected too.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn validate(&self) -> Result<(), Error> {
        let finite = |name: &str, xs: &[f64]| -> Result<(), Error> {
            match xs.iter().find(|x| !x.is_finite()) {
                Some(x) => Err(bad(name, format!("{x} is not a finite number"))),
                None => Ok(()),
            }
        };
        let need = |ok: bool, name: &str, reason: &str| -> Result<(), Error> {
            if ok {
                Ok(())
            } else {
                Err(bad(name, reason))
            }
        };
        let within = |name: &str, x: f64, lo: f64, hi: f64| -> Result<(), Error> {
            if lo <= x && x <= hi {
                Ok(())
            } else {
                Err(bad(name, format!("{x} is not in {lo}..{hi}")))
            }
        };
        finite("duration_s", &[self.duration_s])?;
        need(self.duration_s >= 0.0, "duration_s", "must be >= 0")?;
        if !(1..=13).contains(&self.channel) {
            return Err(bad("channel", format!("{} is not in 1..13", self.channel)));
        }
        finite("spacing_hz", &[self.spacing_hz])?;
        need(self.spacing_hz > 0.0, "spacing_hz", "must be > 0")?;
        finite("tx_power_dbm", &[self.tx_power_dbm])?;
        within("tx_power_dbm", self.tx_power_dbm, -100.0, 60.0)?;
        finite("slot_s", &[self.slot_s])?;
        need(self.slot_s > 0.0, "slot_s", "must be > 0")?;
        for (id, p) in &self.tx_period {
            if *p < 1 {
                return Err(bad(
                    "tx_period",
                    format!("node {id}: {p} is not an integer >= 1"),
                ));
            }
        }
        if self.reflection_order > 2 {
            return Err(bad(
                "reflection_order",
                format!("{} is not 0, 1 or 2", self.reflection_order),
            ));
        }
        // floor_material, ceiling_material, implied_wall_material: the type
        // admits only concrete, brick, drywall, glass and wood.
        finite("slab_thickness_m", &[self.slab_thickness_m])?;
        need(
            self.slab_thickness_m > 0.0,
            "slab_thickness_m",
            "must be > 0",
        )?;
        finite("implied_wall_thickness_m", &[self.implied_wall_thickness_m])?;
        need(
            self.implied_wall_thickness_m > 0.0,
            "implied_wall_thickness_m",
            "must be > 0",
        )?;
        finite("object_scatter_scale", &[self.object_scatter_scale])?;
        need(
            self.object_scatter_scale >= 0.0,
            "object_scatter_scale",
            "must be >= 0",
        )?;
        finite("clutter_margin_m", &[self.clutter_margin_m])?;
        need(
            self.clutter_margin_m >= 0.0,
            "clutter_margin_m",
            "must be >= 0",
        )?;
        finite("tail_power_db", &[self.tail_power_db])?;
        within("tail_power_db", self.tail_power_db, -100.0, 30.0)?;
        finite("tail_decay_s", &[self.tail_decay_s])?;
        need(self.tail_decay_s > 0.0, "tail_decay_s", "must be > 0")?;

        let pairs: [(&str, (f64, f64)); 10] = [
            ("clutter_rcs_m2", self.clutter_rcs_m2),
            ("height_m", self.height_m),
            ("breathing_hz", self.breathing_hz),
            ("breathing_p2p_m", self.breathing_p2p_m),
            ("speed_mps", self.speed_mps),
            ("pause_s", self.pause_s),
            ("device_gain_db", self.device_gain_db),
            ("timing_offset_dev_s", self.timing_offset_dev_s),
            ("noise_floor_dbm", self.noise_floor_dbm),
            ("rssi_range_dbm", self.rssi_range_dbm),
        ];
        for (name, (lo, hi)) in pairs {
            finite(name, &[lo, hi])?;
            if !(lo <= hi) {
                return Err(bad(name, format!("({lo}, {hi}): lo must be <= hi")));
            }
        }
        need(
            self.clutter_rcs_m2.0 > 0.0,
            "clutter_rcs_m2",
            "must be > 0 (log-uniform)",
        )?;
        need(self.speed_mps.0 > 0.0, "speed_mps", "must be > 0")?;
        need(self.pause_s.0 >= 0.0, "pause_s", "must be >= 0")?;
        within("device_gain_db", self.device_gain_db.0, -60.0, 60.0)?;
        within("device_gain_db", self.device_gain_db.1, -60.0, 60.0)?;
        let (nlo, nhi) = self.noise_floor_dbm;
        if !(-127.0 <= nlo && nhi <= -1.0) {
            return Err(bad(
                "noise_floor_dbm",
                format!("({nlo}, {nhi}) is not within -127..-1 dBm"),
            ));
        }
        let (rlo, rhi) = self.rssi_range_dbm;
        if !(rlo.fract() == 0.0 && rhi.fract() == 0.0 && -128.0 <= rlo && rhi <= 127.0) {
            return Err(bad(
                "rssi_range_dbm",
                format!("({rlo}, {rhi}): must be integers within -128..127 (an int8 on the wire)"),
            ));
        }
        if let Some(n) = self.n_people {
            if n > crate::room::MAX_PEOPLE {
                return Err(bad("n_people", format!("{n} > 100")));
            }
        }
        for (i, h) in &self.start_heading_rad {
            finite("start_heading_rad", &[*h]).map_err(|_| {
                bad(
                    "start_heading_rad",
                    format!("people[{i}]: {h} is not finite"),
                )
            })?;
        }
        if !(6..=16).contains(&self.body_points) {
            return Err(bad(
                "body_points",
                format!("{} is not in 6..16", self.body_points),
            ));
        }
        for (name, x) in [
            ("body_rcs_m2", self.body_rcs_m2),
            ("body_radius_m", self.body_radius_m),
            ("walk_margin_m", self.walk_margin_m),
        ] {
            finite(name, &[x])?;
            need(x >= 0.0, name, "must be >= 0")?;
        }
        finite("object_clearance_m", &[self.object_clearance_m])?;
        finite("shadow_cap_db", &[self.shadow_cap_db])?;
        need(self.shadow_cap_db >= 0.0, "shadow_cap_db", "must be >= 0")?;
        let (ox, oy, oz) = self.phone_offset_m;
        finite("phone_offset_m", &[ox, oy, oz])?;
        for (name, x) in [
            ("loss_prob", self.loss_prob),
            ("agc_jump_prob", self.agc_jump_prob),
            ("agc_up_prob", self.agc_up_prob),
            ("ltf_merge_weight", self.ltf_merge_weight),
        ] {
            finite(name, &[x])?;
            need(
                (0.0..=1.0).contains(&x),
                name,
                "is not a probability in 0..1",
            )?;
        }
        for (k, x) in &self.loss_prob_link {
            finite("loss_prob_link", &[*x])?;
            need(
                (0.0..=1.0).contains(x),
                "loss_prob_link",
                &format!("{k}: {x} is not a probability in 0..1"),
            )?;
        }
        for (name, x) in [
            ("agc_step_db", self.agc_step_db),
            ("agc_initial_db", self.agc_initial_db),
            ("agc_max_db", self.agc_max_db),
            ("timing_offset_s", self.timing_offset_s),
            ("ripple_amp", self.ripple_amp),
            ("ripple_phase_rad", self.ripple_phase_rad),
            ("chip_noise_counts", self.chip_noise_counts),
            ("rssi_jitter_db", self.rssi_jitter_db),
            ("target_counts", self.target_counts),
        ] {
            finite(name, &[x])?;
        }
        // Decibel and scale parameters stay where every formula is finite.
        within("agc_step_db", self.agc_step_db, 0.0, 60.0)?;
        within("agc_initial_db", self.agc_initial_db, -60.0, 60.0)?;
        within("agc_max_db", self.agc_max_db, 0.0, 60.0)?;
        within("rssi_jitter_db", self.rssi_jitter_db, 0.0, 60.0)?;
        within("ripple_amp", self.ripple_amp, 0.0, 1.0)?;
        need(
            self.ripple_phase_rad >= 0.0,
            "ripple_phase_rad",
            "must be >= 0",
        )?;
        within("timing_offset_s", self.timing_offset_s, 0.0, 1e-3)?;
        let (dlo, dhi) = self.timing_offset_dev_s;
        within("timing_offset_dev_s", dlo, -1e-3, 1e-3)?;
        within("timing_offset_dev_s", dhi, -1e-3, 1e-3)?;
        within("chip_noise_counts", self.chip_noise_counts, 0.0, 1000.0)?;
        within("target_counts", self.target_counts, 0.0, 1000.0)?;
        need(self.target_counts > 0.0, "target_counts", "must be > 0")?;
        for (name, (lo, hi), top) in [
            ("breathing_p2p_m", self.breathing_p2p_m, 1.0),
            ("height_m", self.height_m, 3.0),
        ] {
            within(name, lo, 0.0, top)?;
            within(name, hi, 0.0, top)?;
        }
        finite("ripple_period_bins", &[self.ripple_period_bins])?;
        need(
            self.ripple_period_bins.abs() >= 1.0,
            "ripple_period_bins",
            "|ripple_period_bins| must be >= 1 (a period in bins)",
        )?;
        let (k0, k1, k2) = self.smoothing_kernel;
        finite("smoothing_kernel", &[k0, k1, k2])?;
        let (e0, e1) = self.smoothing_edge;
        finite("smoothing_edge", &[e0, e1])?;
        if !(8..=64).contains(&self.fw_subcarriers) {
            return Err(bad(
                "fw_subcarriers",
                format!("{} is not in 8..64", self.fw_subcarriers),
            ));
        }
        Ok(())
    }

    /// The §3.2 checks that need the scene: `roles`, `tx_period` and
    /// `loss_prob_link` keys name nodes of this scene (`loss_prob_link`
    /// keys written `"tx,rx"` exactly as `format!("{tx},{rx}")`), at least
    /// one link (a transmitter and a different receiver), and `people_mode`,
    /// `people_posture` and `start_heading_rad` keys are indices of
    /// simulated people (`n_people` of them).
    pub fn validate_for_scene(&self, node_ids: &[u8], n_people: usize) -> Result<(), Error> {
        let known = |id: &u32| u8::try_from(*id).is_ok_and(|i| node_ids.contains(&i));
        for id in self.roles.keys() {
            if !known(id) {
                return Err(bad("roles", format!("node {id} is not in the scene")));
            }
        }
        for id in self.tx_period.keys() {
            if !known(id) {
                return Err(bad("tx_period", format!("node {id} is not in the scene")));
            }
        }
        let role = |id: u8| self.roles.get(&u32::from(id)).copied().unwrap_or_default();
        let has_link = node_ids.iter().any(|&t| {
            role(t).transmits() && node_ids.iter().any(|&r| r != t && role(r).receives())
        });
        if !has_link {
            return Err(bad(
                "roles",
                "no link: a run needs a transmitter and a different receiver (a one-node scene has none)",
            ));
        }
        let person_keys = [
            (
                "people_mode",
                self.people_mode.keys().copied().collect::<Vec<_>>(),
            ),
            (
                "people_posture",
                self.people_posture.keys().copied().collect(),
            ),
            (
                "start_heading_rad",
                self.start_heading_rad.keys().copied().collect(),
            ),
        ];
        for (name, keys) in person_keys {
            if let Some(i) = keys.iter().find(|&&i| i >= n_people) {
                return Err(bad(
                    name,
                    format!("index {i} is not a simulated person (there are {n_people})"),
                ));
            }
        }
        for k in self.loss_prob_link.keys() {
            let ok =
                k.split_once(',')
                    .is_some_and(|(a, b)| match (a.parse::<u8>(), b.parse::<u8>()) {
                        (Ok(a), Ok(b)) => {
                            format!("{a},{b}") == *k
                                && node_ids.contains(&a)
                                && node_ids.contains(&b)
                        }
                        _ => false,
                    });
            if !ok {
                return Err(bad(
                    "loss_prob_link",
                    format!("key {k:?} is not \"tx,rx\" of two nodes in this scene"),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn param_of(e: Error) -> String {
        match e {
            Error::Config { param, .. } => param,
            other => panic!("not a config error: {other}"),
        }
    }

    #[test]
    fn defaults_validate_and_round_trip_through_json() {
        let c = Config::default();
        c.validate().unwrap();
        let json = serde_json::to_string(&c).unwrap();
        let back = Config::from_json_str(&json).unwrap();
        assert_eq!(back, c);
        assert_eq!(Config::from_json_str("{}").unwrap(), c);
        assert_eq!(c.carrier_hz(), 2437e6);
        assert_eq!(c.smoothing_edge.0, 2.0 / 3.0);
    }

    #[test]
    fn json_maps_and_unknown_keys() {
        let c = Config::from_json_str(
            r#"{"roles": {"0": "tx", "1": "rx"}, "tx_period": {"1": 2}, "people_mode": {"0": "stationary"},
               "loss_prob_link": {"0,1": 0.5}, "clutter_rcs_m2": [0.01, 0.1], "floor_material": "wood"}"#,
        )
        .unwrap();
        assert_eq!(c.roles[&0], Role::Tx);
        assert_eq!(c.tx_period[&1], 2);
        assert_eq!(c.people_mode[&0], Mode::Stationary);
        assert_eq!(c.loss_prob_link["0,1"], 0.5);
        assert_eq!(c.floor_material, SurfaceMaterial::Wood);
        // Unknown keys and wrong types name the parameter (§3.2).
        assert_eq!(
            param_of(Config::from_json_str(r#"{"seed": 1, "chanel": 6}"#).unwrap_err()),
            "chanel"
        );
        for (json, param) in [
            (r#"{"floor_material": "metal"}"#, "floor_material"),
            (r#"{"seed": 3, "channel": 6.5}"#, "channel"),
            (r#"{"seed": -1}"#, "seed"),
            (r#"{"impair": 1}"#, "impair"),
            (r#"{"roles": {"0": "listener"}}"#, "roles"),
            (r#"{"tx_period": {"1": 1.5}}"#, "tx_period"),
            (r#"{"height_m": [1.5]}"#, "height_m"),
            (r#"{"people_mode": {"x": "walk"}}"#, "people_mode"),
        ] {
            assert_eq!(
                param_of(Config::from_json_str(json).unwrap_err()),
                param,
                "{json}"
            );
        }
        assert!(matches!(
            Config::from_json_str("[1, 2]"),
            Err(Error::ConfigJson(_))
        ));
        assert!(matches!(
            Config::from_json_str("{"),
            Err(Error::ConfigJson(_))
        ));
    }

    #[test]
    fn validation_names_the_parameter() {
        type Case = (&'static str, Box<dyn Fn(&mut Config)>);
        let cases: Vec<Case> = vec![
            ("channel", Box::new(|c| c.channel = 0)),
            ("channel", Box::new(|c| c.channel = 14)),
            ("fw_subcarriers", Box::new(|c| c.fw_subcarriers = 7)),
            ("fw_subcarriers", Box::new(|c| c.fw_subcarriers = 65)),
            (
                "noise_floor_dbm",
                Box::new(|c| c.noise_floor_dbm = (-128.0, -92.0)),
            ),
            (
                "noise_floor_dbm",
                Box::new(|c| c.noise_floor_dbm = (-97.0, 0.0)),
            ),
            ("tx_period", Box::new(|c| _ = c.tx_period.insert(1, 0))),
            ("body_points", Box::new(|c| c.body_points = 5)),
            ("body_points", Box::new(|c| c.body_points = 17)),
            ("n_people", Box::new(|c| c.n_people = Some(101))),
            ("height_m", Box::new(|c| c.height_m = (1.9, 1.5))),
            ("pause_s", Box::new(|c| c.pause_s = (1.0, 0.0))),
            (
                "clutter_rcs_m2",
                Box::new(|c| c.clutter_rcs_m2 = (0.3, 0.2)),
            ),
            (
                "rssi_range_dbm",
                Box::new(|c| c.rssi_range_dbm = (0.0, -100.0)),
            ),
            ("reflection_order", Box::new(|c| c.reflection_order = 3)),
            ("slot_s", Box::new(|c| c.slot_s = 0.0)),
            // The rest of the list, as the Python reference checks it.
            ("duration_s", Box::new(|c| c.duration_s = -1.0)),
            ("duration_s", Box::new(|c| c.duration_s = f64::INFINITY)),
            ("spacing_hz", Box::new(|c| c.spacing_hz = 0.0)),
            ("tx_power_dbm", Box::new(|c| c.tx_power_dbm = f64::NAN)),
            ("slab_thickness_m", Box::new(|c| c.slab_thickness_m = 0.0)),
            (
                "implied_wall_thickness_m",
                Box::new(|c| c.implied_wall_thickness_m = -0.1),
            ),
            (
                "object_scatter_scale",
                Box::new(|c| c.object_scatter_scale = -1.0),
            ),
            ("clutter_margin_m", Box::new(|c| c.clutter_margin_m = -0.1)),
            ("tail_decay_s", Box::new(|c| c.tail_decay_s = 0.0)),
            ("tail_power_db", Box::new(|c| c.tail_power_db = f64::NAN)),
            (
                "clutter_rcs_m2",
                Box::new(|c| c.clutter_rcs_m2 = (0.0, 0.2)),
            ),
            ("speed_mps", Box::new(|c| c.speed_mps = (0.0, 1.0))),
            ("pause_s", Box::new(|c| c.pause_s = (-1.0, 1.0))),
            ("height_m", Box::new(|c| c.height_m = (1.5, f64::INFINITY))),
            (
                "rssi_range_dbm",
                Box::new(|c| c.rssi_range_dbm = (-100.5, 0.0)),
            ),
            (
                "rssi_range_dbm",
                Box::new(|c| c.rssi_range_dbm = (-129.0, 0.0)),
            ),
            (
                "start_heading_rad",
                Box::new(|c| _ = c.start_heading_rad.insert(0, f64::NAN)),
            ),
            ("body_rcs_m2", Box::new(|c| c.body_rcs_m2 = -0.5)),
            ("body_radius_m", Box::new(|c| c.body_radius_m = -0.1)),
            ("walk_margin_m", Box::new(|c| c.walk_margin_m = -0.1)),
            (
                "object_clearance_m",
                Box::new(|c| c.object_clearance_m = f64::NAN),
            ),
            ("shadow_cap_db", Box::new(|c| c.shadow_cap_db = -1.0)),
            (
                "phone_offset_m",
                Box::new(|c| c.phone_offset_m = (0.0, f64::NAN, 1.0)),
            ),
            ("loss_prob", Box::new(|c| c.loss_prob = 1.5)),
            ("agc_jump_prob", Box::new(|c| c.agc_jump_prob = -0.1)),
            ("agc_up_prob", Box::new(|c| c.agc_up_prob = 2.0)),
            ("ltf_merge_weight", Box::new(|c| c.ltf_merge_weight = 1.1)),
            (
                "loss_prob_link",
                Box::new(|c| _ = c.loss_prob_link.insert("0,1".into(), 1.5)),
            ),
            ("agc_step_db", Box::new(|c| c.agc_step_db = f64::NAN)),
            ("agc_max_db", Box::new(|c| c.agc_max_db = -1.0)),
            (
                "ripple_period_bins",
                Box::new(|c| c.ripple_period_bins = 0.0),
            ),
            (
                "smoothing_kernel",
                Box::new(|c| c.smoothing_kernel = (0.25, f64::NAN, 0.25)),
            ),
            (
                "smoothing_edge",
                Box::new(|c| c.smoothing_edge = (f64::INFINITY, 0.0)),
            ),
        ];
        for (want, f) in cases {
            let mut c = Config::default();
            f(&mut c);
            let got = param_of(c.validate().unwrap_err());
            assert_eq!(got, want);
        }
        let c = Config {
            n_people: Some(100),
            body_points: 16,
            fw_subcarriers: 64,
            ..Config::default()
        };
        c.validate().unwrap();
    }

    /// §3.2: decibel and scale parameters stay where every formula is
    /// finite. Each bound is inclusive; just outside it is rejected,
    /// naming the parameter.
    #[test]
    fn decibel_and_scale_bounds() {
        type Set = fn(&mut Config, f64);
        // (parameter, setter, lowest accepted, highest accepted); a pair's
        // lo and hi are checked one at a time, the other at its own limit.
        let cases: [(&str, Set, Option<f64>, Option<f64>); 21] = [
            (
                "tx_power_dbm",
                |c, x| c.tx_power_dbm = x,
                Some(-100.0),
                Some(60.0),
            ),
            (
                "tail_power_db",
                |c, x| c.tail_power_db = x,
                Some(-100.0),
                Some(30.0),
            ),
            (
                "agc_step_db",
                |c, x| c.agc_step_db = x,
                Some(0.0),
                Some(60.0),
            ),
            ("agc_max_db", |c, x| c.agc_max_db = x, Some(0.0), Some(60.0)),
            (
                "agc_initial_db",
                |c, x| c.agc_initial_db = x,
                Some(-60.0),
                Some(60.0),
            ),
            (
                "device_gain_db",
                |c, x| c.device_gain_db = (x, 60.0),
                Some(-60.0),
                Some(60.0),
            ),
            (
                "device_gain_db",
                |c, x| c.device_gain_db = (-60.0, x),
                Some(-60.0),
                Some(60.0),
            ),
            (
                "rssi_jitter_db",
                |c, x| c.rssi_jitter_db = x,
                Some(0.0),
                Some(60.0),
            ),
            ("ripple_amp", |c, x| c.ripple_amp = x, Some(0.0), Some(1.0)),
            (
                "ripple_phase_rad",
                |c, x| c.ripple_phase_rad = x,
                Some(0.0),
                None,
            ),
            (
                "timing_offset_s",
                |c, x| c.timing_offset_s = x,
                Some(0.0),
                Some(1e-3),
            ),
            (
                "timing_offset_dev_s",
                |c, x| c.timing_offset_dev_s = (x, 1e-3),
                Some(-1e-3),
                Some(1e-3),
            ),
            (
                "timing_offset_dev_s",
                |c, x| c.timing_offset_dev_s = (-1e-3, x),
                Some(-1e-3),
                Some(1e-3),
            ),
            (
                "chip_noise_counts",
                |c, x| c.chip_noise_counts = x,
                Some(0.0),
                Some(1000.0),
            ),
            // target_counts > 0 as well: 0 itself is below.
            (
                "target_counts",
                |c, x| c.target_counts = x,
                None,
                Some(1000.0),
            ),
            (
                "breathing_p2p_m",
                |c, x| c.breathing_p2p_m = (x, 1.0),
                Some(0.0),
                Some(1.0),
            ),
            (
                "breathing_p2p_m",
                |c, x| c.breathing_p2p_m = (0.0, x),
                Some(0.0),
                Some(1.0),
            ),
            (
                "height_m",
                |c, x| c.height_m = (x, 3.0),
                Some(0.0),
                Some(3.0),
            ),
            (
                "height_m",
                |c, x| c.height_m = (0.0, x),
                Some(0.0),
                Some(3.0),
            ),
            (
                "breathing_p2p_m",
                |c, x| c.breathing_p2p_m = (x, x),
                Some(0.0),
                Some(1.0),
            ),
            ("height_m", |c, x| c.height_m = (x, x), Some(0.0), Some(3.0)),
        ];
        // Just outside a bound: a relative step far below any bound's size.
        let step = |b: f64| b.abs().max(1.0) * 1e-12;
        for (name, set, lo, hi) in cases {
            let with = |x: f64| {
                let mut c = Config::default();
                set(&mut c, x);
                c.validate()
            };
            for ok in [lo, hi].into_iter().flatten() {
                with(ok).unwrap_or_else(|e| panic!("{name} = {ok} is in range: {e}"));
            }
            let outside = [lo.map(|l| l - step(l)), hi.map(|h| h + step(h))];
            for x in outside.into_iter().flatten() {
                assert_eq!(param_of(with(x).unwrap_err()), name, "{name} = {x}");
            }
        }
        let target = |x: f64| {
            Config {
                target_counts: x,
                ..Config::default()
            }
            .validate()
        };
        assert_eq!(param_of(target(0.0).unwrap_err()), "target_counts");
        assert_eq!(param_of(target(-1.0).unwrap_err()), "target_counts");
        target(1e-3).unwrap();
        target(f64::MIN_POSITIVE).unwrap();
        // |ripple_period_bins| >= 1: a period in bins, so x_m stays finite.
        let period = |x: f64| {
            Config {
                ripple_period_bins: x,
                ..Config::default()
            }
            .validate()
        };
        for bad in [0.0, 0.5, -0.5, 1e-310, -1e-310] {
            assert_eq!(
                param_of(period(bad).unwrap_err()),
                "ripple_period_bins",
                "{bad}"
            );
        }
        for ok in [1.0, -1.0, 64.0, -64.0] {
            period(ok).unwrap();
        }
    }

    /// §3.2: every integer map key is canonical decimal; anything else is
    /// an error naming the map.
    #[test]
    fn integer_map_keys_are_canonical_decimal() {
        let maps = [
            ("roles", r#""rx""#),
            ("tx_period", "2"),
            ("people_mode", r#""walk""#),
            ("people_posture", r#""stand""#),
            ("start_heading_rad", "0.5"),
        ];
        let bad_keys = [
            "01", "+1", " 1", "1 ", "1_0", "-1", "", "00", "1.0", "1e1", "\u{0661}", "0x1",
        ];
        for (name, value) in maps {
            for key in bad_keys {
                let json = format!(r#"{{"{name}": {{"{key}": {value}}}}}"#);
                assert_eq!(
                    param_of(Config::from_json_str(&json).unwrap_err()),
                    name,
                    "{json}"
                );
            }
            for key in ["0", "1", "10", "254"] {
                let json = format!(r#"{{"{name}": {{"{key}": {value}}}}}"#);
                let c = Config::from_json_str(&json).unwrap_or_else(|e| panic!("{json}: {e}"));
                let n: usize = key.parse().unwrap();
                let present = match name {
                    "roles" => c.roles.contains_key(&(n as u32)),
                    "tx_period" => c.tx_period.contains_key(&(n as u32)),
                    "people_mode" => c.people_mode.contains_key(&n),
                    "people_posture" => c.people_posture.contains_key(&n),
                    _ => c.start_heading_rad.contains_key(&n),
                };
                assert!(present, "{json}");
            }
        }
        assert!(is_canonical_decimal("0") && is_canonical_decimal("120"));
        assert!(!is_canonical_decimal("012") && !is_canonical_decimal("+0"));
    }
}
