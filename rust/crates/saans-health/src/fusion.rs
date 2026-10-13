//! Link-diversity fusion (ADR-006): many links' FeatureFrames in, one
//! 2 Hz `SpaceSample` per space out.
//!
//! Why fuse at all: P0a measured that a single link at 20 MHz sits in a
//! whole-band Fresnel blind spot for a few percent of positions, and
//! nothing in the subcarrier domain rescues it. A second link at another
//! angle almost never shares the blind spot. So presence is a vote across
//! links and breathing comes from whichever link currently sees it best,
//! with hysteresis so the source does not flap between near-equals.

use crate::types::{Breathing, LinkId, Rate, SpaceConfig, SpaceSample};
use saans_dsp::features::{
    FeatureFrame, FLAG_BREATHING_VALID, FLAG_CALIBRATED, FLAG_HEART_VALID, FLAG_OCCUPIED, FLAG_VITALS_ENABLED,
};
use std::collections::{BTreeMap, VecDeque};

/// A frame older than this means the link is not currently usable.
pub const FRESH_S: f64 = 2.5;
/// Below this achieved CSI rate presence still works, vitals do not.
pub const MIN_RATE_PRESENCE_HZ: f64 = 5.0;
/// Campus plan constraint 4: breathing needs >= 10 Hz sustained.
pub const MIN_RATE_VITALS_HZ: f64 = 10.0;
pub const MAX_LOSS_VITALS: f64 = 0.20;
/// Switch the breathing source only for a clearly better link.
pub const SWITCH_MARGIN: f64 = 0.10;

#[derive(Debug, Clone)]
pub struct LinkTrack {
    pub link: LinkId,
    pub last: Option<FeatureFrame>,
    pub last_seen_s: f64,
    pub frames: u64,
    /// The link's empty-room breath_rms (median of recent empty frames),
    /// so a breathing-pause detector can tell "stopped" from "never above
    /// noise". Only learnable with vitals on (breath_rms is gated).
    noise: VecDeque<f64>,
}

impl LinkTrack {
    fn new(link: LinkId) -> Self {
        Self { link, last: None, last_seen_s: f64::NEG_INFINITY, frames: 0, noise: VecDeque::new() }
    }

    pub fn noise_rms(&self) -> Option<f64> {
        if self.noise.len() < 20 {
            return None;
        }
        let mut v: Vec<f64> = self.noise.iter().copied().collect();
        v.sort_by(|a, b| a.total_cmp(b));
        Some(v[v.len() / 2])
    }

    pub fn rate_hz(&self) -> f64 {
        self.last.map_or(0.0, |f| f.csi_rate_chz as f64 / 100.0)
    }

    pub fn loss(&self) -> f64 {
        self.last.map_or(1.0, |f| f.loss_permille as f64 / 1000.0)
    }

    /// Fresh, calibrated and fast enough for presence.
    pub fn usable(&self, t_s: f64) -> bool {
        match self.last {
            Some(f) => {
                t_s - self.last_seen_s <= FRESH_S
                    && f.flag(FLAG_CALIBRATED)
                    && f.presence_threshold > 0.0
                    && self.rate_hz() >= MIN_RATE_PRESENCE_HZ
            }
            None => false,
        }
    }

    /// Usable, and good enough to trust for vital signs.
    pub fn vitals_usable(&self, t_s: f64) -> bool {
        self.usable(t_s) && self.rate_hz() >= MIN_RATE_VITALS_HZ && self.loss() <= MAX_LOSS_VITALS
    }

    fn index(&self, value: f32) -> f64 {
        match self.last {
            Some(f) if f.presence_threshold > 0.0 => value as f64 / f.presence_threshold as f64,
            _ => 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpaceFusion {
    pub config: SpaceConfig,
    pub links: BTreeMap<LinkId, LinkTrack>,
    breathing_link: Option<LinkId>,
}

impl SpaceFusion {
    pub fn new(config: SpaceConfig) -> Self {
        let links = config.links.iter().map(|&l| (l, LinkTrack::new(l))).collect();
        Self { config, links, breathing_link: None }
    }

    /// Accept a frame. Frames from links not listed in the config are
    /// adopted only when the config lists none (handy for a single test
    /// link); otherwise they belong to some other space and are ignored.
    pub fn ingest(&mut self, link: LinkId, frame: FeatureFrame, t_s: f64) -> bool {
        if !self.links.contains_key(&link) {
            if !self.config.links.is_empty() {
                return false;
            }
            self.links.insert(link, LinkTrack::new(link));
        }
        let track = self.links.get_mut(&link).expect("inserted above");
        if frame.flag(FLAG_CALIBRATED) && !frame.flag(FLAG_OCCUPIED) && frame.flag(FLAG_VITALS_ENABLED) && frame.breath_rms.is_finite() {
            if track.noise.len() == 600 {
                track.noise.pop_front();
            }
            track.noise.push_back(frame.breath_rms as f64);
        }
        track.last = Some(frame);
        track.last_seen_s = t_s;
        track.frames += 1;
        true
    }

    fn quorum(&self, usable: usize) -> usize {
        match self.config.presence_quorum {
            0 if usable < 3 => 1,
            0 => usable / 2 + 1,
            k => k.min(usable.max(1)),
        }
    }

    /// The fused state of the space at `t_s`. Call at the frame rate (2 Hz).
    pub fn sample(&mut self, t_s: f64) -> SpaceSample {
        let usable: Vec<&LinkTrack> = self.links.values().filter(|l| l.usable(t_s)).collect();
        let n_usable = usable.len();
        let votes = usable.iter().filter(|l| l.last.unwrap().flag(FLAG_OCCUPIED)).count();
        let occupied = if n_usable == 0 { None } else { Some(votes >= self.quorum(n_usable)) };
        let presence_index = usable.iter().map(|l| l.index(l.last.unwrap().motion_energy)).fold(0.0, f64::max);
        let motion_index = usable.iter().map(|l| l.index(l.last.unwrap().gross_motion)).fold(0.0, f64::max);

        let mut breathing = None;
        let mut heart = None;
        if self.config.vitals && occupied == Some(true) {
            let candidates: Vec<&LinkTrack> = self
                .links
                .values()
                .filter(|l| l.vitals_usable(t_s) && l.last.unwrap().flag(FLAG_BREATHING_VALID))
                .collect();
            let best = candidates
                .iter()
                .filter(|l| l.last.unwrap().breathing_conf.is_finite())
                .max_by(|a, b| a.last.unwrap().breathing_conf.total_cmp(&b.last.unwrap().breathing_conf))
                .map(|l| (l.link, l.last.unwrap().breathing_conf as f64));
            let current = self
                .breathing_link
                .and_then(|id| candidates.iter().find(|l| l.link == id))
                .map(|l| (l.link, l.last.unwrap().breathing_conf as f64));
            let chosen = match (current, best) {
                (Some(cur), Some(b)) if b.1 <= cur.1 + SWITCH_MARGIN => Some(cur.0),
                (_, Some(b)) => Some(b.0),
                (_, None) => None,
            };
            self.breathing_link = chosen;
            if let Some(id) = chosen {
                let track = &self.links[&id];
                let f = track.last.unwrap();
                breathing = Some(Breathing {
                    link: id,
                    bpm: f.breathing_bpm as f64,
                    confidence: f.breathing_conf as f64,
                    band_rms: f.breath_rms as f64,
                    noise_rms: track.noise_rms(),
                });
                if f.flag(FLAG_HEART_VALID) {
                    heart = Some(Rate { bpm: f.heart_bpm as f64, confidence: f.heart_conf as f64 });
                }
            }
        } else {
            self.breathing_link = None;
        }

        SpaceSample {
            t_s,
            occupied,
            presence_index,
            motion_index,
            links_usable: n_usable as u8,
            links_total: self.links.len() as u8,
            breathing,
            heart,
        }
    }

    pub fn breathing_link(&self) -> Option<LinkId> {
        self.breathing_link
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(occupied: bool, conf: f32, rate_hz: f32) -> FeatureFrame {
        let mut flags = FLAG_CALIBRATED | FLAG_VITALS_ENABLED;
        if occupied {
            flags |= FLAG_OCCUPIED | FLAG_BREATHING_VALID;
        }
        FeatureFrame {
            flags,
            csi_rate_chz: (rate_hz * 100.0) as u16,
            presence_threshold: 0.05,
            motion_energy: if occupied { 0.15 } else { 0.03 },
            gross_motion: 0.02,
            breathing_bpm: 14.0,
            breathing_conf: conf,
            breath_rms: if occupied { 0.2 } else { 0.01 },
            ..Default::default()
        }
    }

    fn space(n: u8) -> SpaceFusion {
        SpaceFusion::new(SpaceConfig {
            links: (0..n).map(|i| LinkId { rx: i, tx: 9 }).collect(),
            vitals: true,
            ..Default::default()
        })
    }

    #[test]
    fn no_usable_link_means_unknown_not_empty() {
        let mut s = space(2);
        assert_eq!(s.sample(10.0).occupied, None);
        s.ingest(LinkId { rx: 0, tx: 9 }, frame(false, 0.0, 20.0), 10.0);
        assert_eq!(s.sample(10.5).occupied, Some(false));
        assert_eq!(s.sample(20.0).occupied, None, "stale link must not vote");
    }

    #[test]
    fn one_link_in_a_blind_spot_does_not_hide_the_person() {
        let mut s = space(2);
        s.ingest(LinkId { rx: 0, tx: 9 }, frame(true, 0.9, 20.0), 1.0);
        s.ingest(LinkId { rx: 1, tx: 9 }, frame(false, 0.0, 20.0), 1.0);
        let out = s.sample(1.2);
        assert_eq!(out.occupied, Some(true));
        assert_eq!(out.breathing.unwrap().link, LinkId { rx: 0, tx: 9 });
    }

    #[test]
    fn majority_vote_with_three_links() {
        let mut s = space(3);
        s.ingest(LinkId { rx: 0, tx: 9 }, frame(true, 0.9, 20.0), 1.0);
        s.ingest(LinkId { rx: 1, tx: 9 }, frame(false, 0.0, 20.0), 1.0);
        s.ingest(LinkId { rx: 2, tx: 9 }, frame(false, 0.0, 20.0), 1.0);
        assert_eq!(s.sample(1.2).occupied, Some(false), "one wall-leaking link must not outvote two");
    }

    #[test]
    fn breathing_source_has_hysteresis() {
        let mut s = space(2);
        let (a, b) = (LinkId { rx: 0, tx: 9 }, LinkId { rx: 1, tx: 9 });
        s.ingest(a, frame(true, 0.80, 20.0), 1.0);
        s.ingest(b, frame(true, 0.70, 20.0), 1.0);
        assert_eq!(s.sample(1.1).breathing.unwrap().link, a);
        s.ingest(b, frame(true, 0.85, 20.0), 1.5);
        assert_eq!(s.sample(1.6).breathing.unwrap().link, a, "0.05 better is not enough to switch");
        s.ingest(b, frame(true, 0.95, 20.0), 2.0);
        assert_eq!(s.sample(2.1).breathing.unwrap().link, b);
    }

    #[test]
    fn slow_links_vote_on_presence_but_not_vitals() {
        let mut s = space(1);
        s.ingest(LinkId { rx: 0, tx: 9 }, frame(true, 0.9, 7.0), 1.0);
        let out = s.sample(1.1);
        assert_eq!(out.occupied, Some(true));
        assert!(out.breathing.is_none(), "7 Hz is below the 10 Hz vitals floor");
    }

    #[test]
    fn nan_fields_do_not_panic() {
        // CRC-valid frames can still carry NaN; one bad node must not take
        // the gateway down (found by the gateway's end-to-end tests).
        let mut s = space(2);
        let mut bad = frame(true, f32::NAN, 20.0);
        bad.breath_rms = f32::NAN;
        s.ingest(LinkId { rx: 0, tx: 9 }, bad, 1.0);
        s.ingest(LinkId { rx: 1, tx: 9 }, frame(true, 0.8, 20.0), 1.0);
        let out = s.sample(1.1);
        assert_eq!(out.breathing.unwrap().link, LinkId { rx: 1, tx: 9 });
    }

    #[test]
    fn vitals_off_means_no_breathing_even_if_frames_carry_it() {
        let mut s = SpaceFusion::new(SpaceConfig { vitals: false, ..Default::default() });
        s.ingest(LinkId { rx: 0, tx: 9 }, frame(true, 0.9, 20.0), 1.0);
        assert!(s.sample(1.1).breathing.is_none());
    }
}
