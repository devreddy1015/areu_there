//! The gateway's deterministic core: datagrams in, per-space state and
//! health events out. No I/O and no clocks: the caller supplies `t_s`, so
//! live mode, replay and tests all run exactly the same code.
//!
//! ```text
//! datagram ─┬─ FeatureFrame v1, or v2 checked (frameauth) ┐
//!           └─ raw CSI ─► FeatureExtractor (per node)┴─► route by link ─► SpaceFusion
//! tick(t) every 0.5 s: SpaceFusion::sample ─► SpaceMonitor::update + SystemMonitor::observe
//! ```

use crate::config::{GatewayConfig, RawConfig};
use crate::frameauth::{self, AuthStats, Refused, ReplayGuard, SiteKey};
use crate::privacy;
use crate::raw;
use saans_dsp::features::{
    DecodeError, FeatureFrame, FEATURE_VERSION_SIGNED, FLAG_RAW_MODE, SIGNED_FRAME_SIZE,
};
use saans_dsp::{CsiFrame, FeatureExtractor, FeatureExtractorConfig};
use saans_health::{
    HealthEvent, LinkId, SpaceConfig, SpaceFusion, SpaceMonitor, SpaceSample, SpaceState,
    SystemMonitor,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::net::SocketAddr;

/// Fusion and detectors run at the FeatureFrame rate.
pub const TICK_S: f64 = 0.5;
/// A link counts as "raw mode active" if a raw-flagged frame arrived this recently.
pub const RAW_ACTIVE_S: f64 = 10.0;
/// Gateway-computed features defer to the node's own FeatureFrames for
/// the same link while those keep arriving.
pub const NATIVE_PREFERRED_S: f64 = 5.0;
/// Raw tx id: raw-CSI does not say who transmitted the frame.
pub const RAW_TX: u8 = 0xFF;

#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct DecodeErrors {
    /// FeatureFrame magic but not 68 bytes.
    pub size: u64,
    pub crc: u64,
    pub version: u64,
    /// Neither FeatureFrame nor raw-CSI magic.
    pub unknown_type: u64,
    /// raw-CSI magic but short header or truncated payload.
    pub raw_malformed: u64,
    /// raw-CSI frame whose subcarrier count differs from the node's first.
    pub raw_subcarrier_mismatch: u64,
    /// Valid CRC but a NaN or infinite field: dropped before fusion, whose
    /// comparisons assume finite values.
    pub non_finite: u64,
}

impl DecodeErrors {
    pub fn total(&self) -> u64 {
        self.size
            + self.crc
            + self.version
            + self.unknown_type
            + self.raw_malformed
            + self.raw_subcarrier_mismatch
            + self.non_finite
    }
}

#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct IngestStats {
    pub datagrams: u64,
    /// Valid FeatureFrame datagrams.
    pub feature_frames: u64,
    /// Valid raw-CSI datagrams.
    pub raw_datagrams: u64,
    /// FeatureFrames the gateway computed from raw CSI.
    pub raw_feature_frames: u64,
    /// Computed frames not used because the node sends its own for the link.
    pub raw_shadowed: u64,
    /// Frames (or raw datagrams) from links no space lists.
    pub unassigned: u64,
    /// Frames that carried vitals for a space with vitals off (zeroed here).
    pub vitals_gated: u64,
    pub errors: DecodeErrors,
    /// Signed frames (ADR-012): accepted, and refused by reason.
    pub frame_auth: AuthStats,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct UnknownLink {
    pub rx: u8,
    pub tx: u8,
    /// "feature" or "raw".
    pub kind: &'static str,
    pub frames: u64,
    pub first_seen_s: f64,
    pub last_seen_s: f64,
    /// Where the last datagram came from: finds the misconfigured node.
    pub last_src: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ingested {
    Frame {
        space: usize,
        link: LinkId,
    },
    /// Raw CSI accepted; `frames` FeatureFrames were due.
    Raw {
        space: usize,
        link: LinkId,
        frames: usize,
    },
    Shadowed {
        link: LinkId,
    },
    Unknown(LinkId),
    Rejected(&'static str),
}

pub struct SpaceRt {
    pub config: SpaceConfig,
    pub fusion: SpaceFusion,
    pub monitor: SpaceMonitor,
    pub system: SystemMonitor,
    pub state: SpaceState,
    pub sample: Option<SpaceSample>,
    /// Last time a node-computed FeatureFrame arrived, per link.
    last_native: BTreeMap<LinkId, f64>,
    /// Links that sent vitals although this space has not opted in.
    pub vitals_gated_links: BTreeSet<LinkId>,
}

impl SpaceRt {
    /// Links whose latest frame says the node is also streaming raw CSI.
    pub fn raw_mode_links(&self, t_s: f64) -> Vec<LinkId> {
        self.fusion
            .links
            .values()
            .filter(|l| {
                l.last.is_some_and(|f| f.flag(FLAG_RAW_MODE)) && t_s - l.last_seen_s <= RAW_ACTIVE_S
            })
            .map(|l| l.link)
            .collect()
    }
}

/// Every float field finite (NaN would panic fusion's confidence sort).
fn is_finite(f: &FeatureFrame) -> bool {
    [
        f.motion_energy,
        f.presence_threshold,
        f.gross_motion,
        f.breath_rms,
        f.breathing_bpm,
        f.breathing_conf,
        f.heart_bpm,
        f.heart_conf,
        f.best_bnr,
    ]
    .iter()
    .all(|x| x.is_finite())
}

struct RawLink {
    extractor: FeatureExtractor,
}

pub struct Engine {
    pub spaces: Vec<SpaceRt>,
    route: HashMap<LinkId, usize>,
    raw_cfg: RawConfig,
    raw: BTreeMap<u8, RawLink>,
    pub stats: IngestStats,
    pub unknown: BTreeMap<LinkId, UnknownLink>,
    /// Site keys by `key_id`; empty = authentication off.
    frame_keys: BTreeMap<u8, SiteKey>,
    /// The highest (boot, seq) accepted per link; persisted by the hub.
    pub replay: ReplayGuard,
}

impl Engine {
    pub fn new(cfg: &GatewayConfig) -> Self {
        let mut route = HashMap::new();
        let spaces = cfg
            .spaces
            .iter()
            .enumerate()
            .map(|(i, sc)| {
                for l in &sc.links {
                    route.insert(*l, i);
                }
                SpaceRt {
                    config: sc.clone(),
                    fusion: SpaceFusion::new(sc.clone()),
                    monitor: SpaceMonitor::new(sc.clone()),
                    system: SystemMonitor::new(),
                    state: SpaceState {
                        space_id: sc.id.clone(),
                        vitals: sc.vitals,
                        ..Default::default()
                    },
                    sample: None,
                    last_native: BTreeMap::new(),
                    vitals_gated_links: BTreeSet::new(),
                }
            })
            .collect();
        Self {
            spaces,
            route,
            raw_cfg: cfg.raw.clone(),
            raw: BTreeMap::new(),
            stats: IngestStats::default(),
            unknown: BTreeMap::new(),
            frame_keys: BTreeMap::new(),
            replay: ReplayGuard::default(),
        }
    }

    /// Turn frame authentication on (ADR-012): from now on only signed
    /// frames checked against these keys are accepted. The keys come from
    /// `frameauth::load_site_keys`; the engine does no I/O itself.
    pub fn set_frame_keys(&mut self, keys: BTreeMap<u8, SiteKey>) {
        self.frame_keys = keys;
    }

    pub fn frame_auth_required(&self) -> bool {
        !self.frame_keys.is_empty()
    }

    pub fn frame_key_ids(&self) -> Vec<u8> {
        self.frame_keys.keys().copied().collect()
    }

    pub fn space_index(&self, id: &str) -> Option<usize> {
        self.spaces.iter().position(|s| s.config.id == id)
    }

    /// One UDP datagram received at gateway time `t_s`.
    pub fn ingest(&mut self, data: &[u8], t_s: f64, src: Option<SocketAddr>) -> Ingested {
        self.stats.datagrams += 1;
        let signed =
            data.len() == SIGNED_FRAME_SIZE && data.get(4) == Some(&FEATURE_VERSION_SIGNED);
        if FeatureFrame::sniff(data) && signed {
            self.ingest_signed(data, t_s, src)
        } else if FeatureFrame::sniff(data) && self.frame_auth_required() {
            // A v1 frame, or anything else wearing the magic: accepting
            // it would leave the unsigned way in open beside the signed one.
            self.stats.frame_auth.unsigned_refused += 1;
            Ingested::Rejected("unsigned FeatureFrame")
        } else if FeatureFrame::sniff(data) {
            match FeatureFrame::decode(data) {
                Ok(f) if !is_finite(&f) => {
                    self.stats.errors.non_finite += 1;
                    Ingested::Rejected("non-finite field")
                }
                Ok(f) => {
                    self.stats.feature_frames += 1;
                    self.route_frame(f, t_s, src, true)
                }
                Err(e) => {
                    let errs = &mut self.stats.errors;
                    match e {
                        DecodeError::Size(_) => errs.size += 1,
                        DecodeError::Crc => errs.crc += 1,
                        DecodeError::Version(_) => errs.version += 1,
                        DecodeError::Magic(_) => errs.unknown_type += 1,
                    }
                    Ingested::Rejected("feature frame")
                }
            }
        } else if raw::sniff(data) && self.frame_auth_required() {
            self.stats.frame_auth.raw_refused += 1;
            Ingested::Rejected("raw CSI is unauthenticated")
        } else if raw::sniff(data) {
            self.ingest_raw(data, t_s, src)
        } else {
            self.stats.errors.unknown_type += 1;
            Ingested::Rejected("unknown datagram type")
        }
    }

    /// A v2 frame: check the tag, then the replay floor, then route it as
    /// any frame (ADR-012, `spec/wire.md` §2b).
    fn ingest_signed(&mut self, data: &[u8], t_s: f64, src: Option<SocketAddr>) -> Ingested {
        let auth = &mut self.stats.frame_auth;
        if self.frame_keys.is_empty() {
            auth.no_key += 1;
            return Ingested::Rejected("signed FeatureFrame, but no frame_key is configured");
        }
        let v = match frameauth::verify(data, &self.frame_keys) {
            Ok(v) => v,
            Err(Refused::UnknownKey(_)) => {
                auth.unknown_key += 1;
                return Ingested::Rejected("unknown key_id");
            }
            Err(Refused::Tag) => {
                auth.bad_tag += 1;
                return Ingested::Rejected("bad tag");
            }
            // Ruled out by the dispatch in `ingest`.
            Err(Refused::Size(_) | Refused::Magic | Refused::Version(_)) => {
                self.stats.errors.version += 1;
                return Ingested::Rejected("feature frame");
            }
        };
        let f = v.frame;
        if !self.replay.accept(f.node_id, f.tx_id, v.boot, f.seq) {
            auth.replayed += 1;
            return Ingested::Rejected("replayed or stale counter");
        }
        auth.accepted += 1;
        if !is_finite(&f) {
            self.stats.errors.non_finite += 1;
            return Ingested::Rejected("non-finite field");
        }
        self.stats.feature_frames += 1;
        self.route_frame(f, t_s, src, true)
    }

    fn ingest_raw(&mut self, data: &[u8], t_s: f64, src: Option<SocketAddr>) -> Ingested {
        let fr = match raw::parse(data) {
            // No subcarriers or no antennas carries no CSI (and an extractor
            // sized from it would assert); a short I/Q list cannot be fed.
            Ok(fr)
                if fr.n_subcarriers > 0
                    && fr.n_antennas > 0
                    && fr.iq.len() == fr.n_subcarriers as usize =>
            {
                fr
            }
            _ => {
                self.stats.errors.raw_malformed += 1;
                return Ingested::Rejected("malformed raw-CSI frame");
            }
        };
        self.stats.raw_datagrams += 1;
        let link = LinkId {
            rx: fr.node_id,
            tx: RAW_TX,
        };
        let Some(&space) = self.route.get(&link) else {
            self.note_unknown(link, "raw", t_s, src);
            return Ingested::Unknown(link);
        };
        let rc = &self.raw_cfg;
        let vitals = self.spaces[space].config.vitals;
        let rl = self.raw.entry(fr.node_id).or_insert_with(|| RawLink {
            extractor: FeatureExtractor::new(FeatureExtractorConfig {
                n_subcarriers: if rc.subcarriers > 0 {
                    rc.subcarriers
                } else {
                    fr.n_subcarriers as usize
                },
                sample_rate_hz: rc.sample_rate_hz,
                node_id: fr.node_id,
                tx_id: RAW_TX,
                vitals_enabled: vitals,
                heart_enabled: false,
                calibration_s: rc.calibration_s,
                raw_mode: true,
                ..Default::default()
            }),
        });
        if fr.n_subcarriers as usize != rl.extractor.config.n_subcarriers {
            self.stats.errors.raw_subcarrier_mismatch += 1;
            return Ingested::Rejected("subcarrier count changed");
        }
        let csi = CsiFrame::from_iq(
            &fr.iq,
            fr.sequence as u64,
            rc.sample_rate_hz,
            fr.rssi_dbm as i32,
            fr.node_id,
        );
        let noise = (fr.noise_floor_dbm != i8::MIN).then_some(fr.noise_floor_dbm as i32);
        let frames = rl
            .extractor
            .push(csi, Some(fr.sequence as u64), Some(t_s), noise);
        let n = frames.len();
        for f in frames {
            if !is_finite(&f) {
                self.stats.errors.non_finite += 1;
                continue;
            }
            self.stats.raw_feature_frames += 1;
            self.route_frame(f, t_s, src, false);
        }
        Ingested::Raw {
            space,
            link,
            frames: n,
        }
    }

    fn note_unknown(
        &mut self,
        link: LinkId,
        kind: &'static str,
        t_s: f64,
        src: Option<SocketAddr>,
    ) {
        self.stats.unassigned += 1;
        let u = self.unknown.entry(link).or_insert(UnknownLink {
            rx: link.rx,
            tx: link.tx,
            kind,
            frames: 0,
            first_seen_s: t_s,
            last_seen_s: t_s,
            last_src: None,
        });
        u.frames += 1;
        u.last_seen_s = t_s;
        u.kind = kind;
        if let Some(a) = src {
            u.last_src = Some(a.to_string());
        }
    }

    fn route_frame(
        &mut self,
        f: FeatureFrame,
        t_s: f64,
        src: Option<SocketAddr>,
        native: bool,
    ) -> Ingested {
        let link = LinkId {
            rx: f.node_id,
            tx: f.tx_id,
        };
        let Some(&space) = self.route.get(&link) else {
            self.note_unknown(link, "feature", t_s, src);
            return Ingested::Unknown(link);
        };
        let sp = &mut self.spaces[space];
        if native {
            sp.last_native.insert(link, t_s);
        } else if sp
            .last_native
            .get(&link)
            .is_some_and(|&tn| t_s - tn < NATIVE_PREFERRED_S)
        {
            self.stats.raw_shadowed += 1;
            return Ingested::Shadowed { link };
        }
        let f = if !sp.config.vitals && privacy::frame_has_vitals(&f) {
            self.stats.vitals_gated += 1;
            sp.vitals_gated_links.insert(link);
            privacy::gate_frame(f)
        } else {
            f
        };
        sp.fusion.ingest(link, f, t_s);
        Ingested::Frame { space, link }
    }

    /// Advance every space to `t_s`; returns (space index, event) pairs.
    pub fn tick(&mut self, t_s: f64) -> Vec<(usize, HealthEvent)> {
        let mut out = Vec::new();
        for (i, sp) in self.spaces.iter_mut().enumerate() {
            let sample = sp.fusion.sample(t_s);
            let mut events = sp.monitor.update(&sample);
            events.extend(sp.system.observe(&sp.config.id, &sp.fusion, t_s));
            sp.state = sp.monitor.state();
            sp.sample = Some(sample);
            out.extend(events.into_iter().map(|e| (i, e)));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use saans_dsp::features::{FLAG_CALIBRATED, FLAG_OCCUPIED};

    fn cfg() -> GatewayConfig {
        GatewayConfig::from_toml(
            "[raw]\ncalibration_s = 2.0\n\
             [[space]]\nid = \"a\"\nvitals = true\nlinks = [{ rx = 1, tx = 2 }, { rx = 5, tx = 255 }]\n\
             [[space]]\nid = \"b\"\nlinks = [{ rx = 3, tx = 2 }]\n",
        )
        .unwrap()
    }

    fn frame(rx: u8, tx: u8) -> FeatureFrame {
        FeatureFrame {
            node_id: rx,
            tx_id: tx,
            flags: FLAG_CALIBRATED | FLAG_OCCUPIED,
            csi_rate_chz: 2000,
            presence_threshold: 0.05,
            motion_energy: 0.2,
            ..Default::default()
        }
    }

    #[test]
    fn routes_by_link_and_counts_errors() {
        let mut e = Engine::new(&cfg());
        assert_eq!(
            e.ingest(&frame(1, 2).encode(), 0.0, None),
            Ingested::Frame {
                space: 0,
                link: LinkId { rx: 1, tx: 2 }
            }
        );
        assert_eq!(
            e.ingest(&frame(3, 2).encode(), 0.0, None),
            Ingested::Frame {
                space: 1,
                link: LinkId { rx: 3, tx: 2 }
            }
        );
        assert_eq!(
            e.ingest(&frame(9, 9).encode(), 0.0, None),
            Ingested::Unknown(LinkId { rx: 9, tx: 9 })
        );
        let mut bad = frame(1, 2).encode();
        bad[30] ^= 1;
        assert!(matches!(e.ingest(&bad, 0.0, None), Ingested::Rejected(_)));
        assert!(matches!(
            e.ingest(&bad[..60], 0.0, None),
            Ingested::Rejected(_)
        ));
        assert!(matches!(
            e.ingest(b"hello world", 0.0, None),
            Ingested::Rejected(_)
        ));
        assert_eq!(
            e.stats.errors,
            DecodeErrors {
                crc: 1,
                size: 1,
                unknown_type: 1,
                ..Default::default()
            }
        );
        assert_eq!(e.unknown[&LinkId { rx: 9, tx: 9 }].frames, 1);
        assert_eq!(e.stats.datagrams, 6);
    }

    #[test]
    fn vitals_are_gated_for_spaces_without_consent() {
        let mut e = Engine::new(&cfg());
        let mut f = frame(3, 2);
        f.breathing_bpm = 15.0;
        f.breathing_conf = 0.9;
        e.ingest(&f.encode(), 0.0, None);
        let last = e.spaces[1].fusion.links[&LinkId { rx: 3, tx: 2 }]
            .last
            .unwrap();
        assert_eq!((last.breathing_bpm, last.breathing_conf), (0.0, 0.0));
        assert_eq!(e.stats.vitals_gated, 1);
        e.ingest(
            &{
                let mut g = frame(1, 2);
                g.breathing_bpm = 15.0;
                g
            }
            .encode(),
            0.0,
            None,
        );
        assert_eq!(
            e.spaces[0].fusion.links[&LinkId { rx: 1, tx: 2 }]
                .last
                .unwrap()
                .breathing_bpm,
            15.0
        );
    }

    #[test]
    fn raw_csi_becomes_raw_mode_feature_frames() {
        let mut e = Engine::new(&cfg());
        let mut produced = 0;
        // Presence warms its filter up for 50 s before calibrating (2 s here).
        for i in 0..1200u32 {
            let t = i as f64 / 20.0;
            let iq: Vec<(i8, i8)> = (0..56)
                .map(|k| {
                    (
                        ((t * 0.3 + k as f64).sin() * 20.0) as i8,
                        40 + ((t * 1.3).sin() * 5.0) as i8,
                    )
                })
                .collect();
            match e.ingest(&raw::build(&iq, 5, i, -50, -95), t, None) {
                Ingested::Raw { frames, .. } => produced += frames,
                other => panic!("{other:?}"),
            }
        }
        assert_eq!(
            produced, 120,
            "1200 CSI frames at 20 Hz -> 120 FeatureFrames at 2 Hz"
        );
        let track = &e.spaces[0].fusion.links[&LinkId { rx: 5, tx: 255 }];
        let last = track.last.unwrap();
        assert!(last.flag(FLAG_RAW_MODE));
        assert!(
            last.flag(FLAG_CALIBRATED),
            "50 s warm-up + 2 s calibration < 60 s"
        );
        assert_eq!(last.n_subcarriers, 56);
        assert_eq!(
            e.spaces[0].raw_mode_links(60.0),
            vec![LinkId { rx: 5, tx: 255 }]
        );
        // A different subcarrier count is rejected, not mixed in.
        assert!(matches!(
            e.ingest(&raw::build(&[(1, 1); 52], 5, 1200, -50, -95), 60.0, None),
            Ingested::Rejected(_)
        ));
        // A raw node no space lists is unknown.
        assert!(matches!(
            e.ingest(&raw::build(&[(1, 1); 52], 6, 0, -50, -95), 60.0, None),
            Ingested::Unknown(_)
        ));
        assert_eq!(e.unknown[&LinkId { rx: 6, tx: 255 }].kind, "raw");
    }

    #[test]
    fn native_frames_take_precedence_over_computed_ones() {
        let mut e = Engine::new(&cfg());
        e.ingest(&frame(5, 255).encode(), 0.0, None);
        let mut shadowed = 0;
        for i in 0..20u32 {
            if let Ingested::Raw { frames, .. } = e.ingest(
                &raw::build(&[(3, 4); 56], 5, i, -50, -95),
                0.05 * i as f64,
                None,
            ) {
                shadowed += frames;
            }
        }
        assert_eq!(e.stats.raw_shadowed as usize, shadowed);
        assert!(shadowed > 0);
    }
}
