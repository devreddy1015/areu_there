//! A bounded, in-memory time series, so the dashboard can draw a trend
//! instead of a number that jumps.
//!
//! Nothing here is persisted. Events go to disk because someone may have
//! to answer for them later; a motion trace is a minute-by-minute record
//! of what a room was doing, which is exactly the thing DPDP retention
//! limits exist for. Keeping it in RAM, bounded, means the window is the
//! retention policy and a restart is a delete.
//!
//! Two rings per deployment:
//!
//! - one per **space**: occupancy, motion, usable links, and (only where
//!   the space consented) the breathing estimate;
//! - one per **link**: CSI rate, loss, BNR and RSSI — the numbers that
//!   answer "is this sensor still any good?".
//!
//! Both are sampled at `cadence_s`, not at the 0.5 s tick: an hour of
//! 2 Hz samples is 7200 points that no chart can show and no browser
//! wants to parse. The ring is sized from `window_s / cadence_s`, so the
//! memory is fixed at startup and cannot grow with uptime.

use crate::roles::Principal;
use saans_health::types::{LinkId, SpaceSample};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, VecDeque};

/// Longest window the config may ask for. 6 h of 2 s samples is 10,800
/// points a series; beyond that this is a time-series database's job.
pub const MAX_WINDOW_S: f64 = 6.0 * 3600.0;
pub const MIN_CADENCE_S: f64 = 0.5;

#[derive(Debug, Clone, Copy)]
pub struct SpacePoint {
    pub t_s: f64,
    pub unix_ms: i64,
    /// `None` = no usable link: not the same as "empty".
    pub occupied: Option<bool>,
    pub motion_index: f32,
    pub presence_index: f32,
    pub links_usable: u8,
    pub links_total: u8,
    /// `None` when vitals are off, nobody is there, or no link is usable.
    pub breathing: Option<(f32, f32, f32)>,
}

#[derive(Debug, Clone, Copy)]
pub struct LinkPoint {
    pub t_s: f64,
    pub unix_ms: i64,
    pub up: bool,
    pub rate_hz: f32,
    pub loss: f32,
    pub best_bnr: f32,
    pub rssi_dbm: i8,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HistoryConfig {
    pub window_s: f64,
    pub cadence_s: f64,
}

impl Default for HistoryConfig {
    fn default() -> Self {
        Self {
            window_s: 3600.0,
            cadence_s: 2.0,
        }
    }
}

impl HistoryConfig {
    /// Points per series. At least one, so a degenerate config cannot
    /// produce a ring that drops everything.
    pub fn capacity(&self) -> usize {
        ((self.window_s / self.cadence_s).ceil() as usize)
            .clamp(1, (MAX_WINDOW_S / MIN_CADENCE_S) as usize + 1)
    }
}

#[derive(Debug)]
struct Ring<T> {
    points: VecDeque<T>,
    last_t_s: Option<f64>,
}

// Hand-written: the derive would demand `T: Default`, which a sample
// point has no sensible value for.
impl<T> Default for Ring<T> {
    fn default() -> Self {
        Self {
            points: VecDeque::new(),
            last_t_s: None,
        }
    }
}

impl<T> Ring<T> {
    /// True if `t_s` is due given the cadence. Time can also go backwards
    /// — a replay restarts at zero — in which case the ring is cleared
    /// rather than left with a series that jumps back in time.
    fn due(&mut self, t_s: f64, cadence_s: f64) -> bool {
        match self.last_t_s {
            Some(prev) if t_s < prev => {
                self.points.clear();
                true
            }
            // Half a cadence of slack: the tick is 0.5 s and the cadence
            // is a multiple of it, so exact equality cannot be relied on.
            Some(prev) => t_s - prev >= cadence_s - 1e-9,
            None => true,
        }
    }

    fn push(&mut self, t_s: f64, cadence_s: f64, cap: usize, value: T) {
        if !self.due(t_s, cadence_s) {
            return;
        }
        self.last_t_s = Some(t_s);
        if self.points.len() >= cap {
            self.points.pop_front();
        }
        self.points.push_back(value);
    }
}

#[derive(Debug)]
pub struct History {
    cfg: HistoryConfig,
    spaces: BTreeMap<String, Ring<SpacePoint>>,
    links: BTreeMap<(String, LinkId), Ring<LinkPoint>>,
}

impl History {
    pub fn new(cfg: HistoryConfig) -> Self {
        Self {
            cfg,
            spaces: BTreeMap::new(),
            links: BTreeMap::new(),
        }
    }

    pub fn config(&self) -> HistoryConfig {
        self.cfg
    }

    /// Points currently held, across every series. For `/api/health`, so
    /// an operator can see what the bound is actually costing.
    pub fn points(&self) -> usize {
        self.spaces.values().map(|r| r.points.len()).sum::<usize>()
            + self.links.values().map(|r| r.points.len()).sum::<usize>()
    }

    pub fn record_space(&mut self, space_id: &str, unix_ms: i64, s: &SpaceSample, vitals: bool) {
        let cap = self.cfg.capacity();
        let point = SpacePoint {
            t_s: s.t_s,
            unix_ms,
            occupied: s.occupied,
            motion_index: s.motion_index as f32,
            presence_index: s.presence_index as f32,
            links_usable: s.links_usable,
            links_total: s.links_total,
            // Belt and braces: fusion already withholds breathing for a
            // space without consent, and this will not record it either.
            breathing: s
                .breathing
                .filter(|_| vitals)
                .map(|b| (b.bpm as f32, b.confidence as f32, b.band_rms as f32)),
        };
        self.spaces.entry(space_id.to_string()).or_default().push(
            s.t_s,
            self.cfg.cadence_s,
            cap,
            point,
        );
    }

    pub fn record_link(&mut self, space_id: &str, link: LinkId, p: LinkPoint) {
        let cap = self.cfg.capacity();
        self.links
            .entry((space_id.to_string(), link))
            .or_default()
            .push(p.t_s, self.cfg.cadence_s, cap, p);
    }

    /// Spaces and links no longer configured (a config is loaded once per
    /// run, so this only matters to long-lived tests and replays).
    pub fn retain(&mut self, keep: &dyn Fn(&str) -> bool) {
        self.spaces.retain(|k, _| keep(k));
        self.links.retain(|(k, _), _| keep(k));
    }

    /// Columnar JSON for `/api/history`: parallel arrays, not an array of
    /// objects. An hour of six series is ~10x smaller this way, which is
    /// the difference between a chart that updates and one that stutters
    /// on a Raspberry Pi.
    ///
    /// `space` limits to one space; `since_t_s` returns only later
    /// points; `max_points` strides the result down (never interpolates,
    /// so every value shown was actually measured).
    pub fn to_json(
        &self,
        principal: &Principal,
        consented: &dyn Fn(&str) -> bool,
        named: &dyn Fn(&str) -> String,
        space: Option<&str>,
        since_t_s: Option<f64>,
        max_points: usize,
    ) -> Value {
        let want = |id: &str| space.is_none_or(|s| s == id) && principal.can_see_space(id);
        let spaces: Vec<Value> = self
            .spaces
            .iter()
            .filter(|(id, _)| want(id))
            .map(|(id, ring)| {
                let pts = select(&ring.points, since_t_s, max_points, |p| p.t_s);
                let show_vitals = principal.can_see_vitals(id, consented(id));
                let mut m = Map::new();
                m.insert("space_id".into(), id.as_str().into());
                m.insert("name".into(), named(id).into());
                m.insert("points".into(), pts.len().into());
                m.insert(
                    "t_s".into(),
                    pts.iter().map(|p| p.t_s).collect::<Vec<_>>().into(),
                );
                m.insert(
                    "unix_ms".into(),
                    pts.iter().map(|p| p.unix_ms).collect::<Vec<_>>().into(),
                );
                // null, not false: "we cannot see the room" is a third state.
                m.insert(
                    "occupied".into(),
                    pts.iter()
                        .map(|p| match p.occupied {
                            Some(b) => Value::from(b),
                            None => Value::Null,
                        })
                        .collect::<Vec<_>>()
                        .into(),
                );
                m.insert(
                    "motion_index".into(),
                    pts.iter()
                        .map(|p| p.motion_index)
                        .collect::<Vec<_>>()
                        .into(),
                );
                m.insert(
                    "presence_index".into(),
                    pts.iter()
                        .map(|p| p.presence_index)
                        .collect::<Vec<_>>()
                        .into(),
                );
                m.insert(
                    "links_usable".into(),
                    pts.iter()
                        .map(|p| p.links_usable)
                        .collect::<Vec<_>>()
                        .into(),
                );
                m.insert(
                    "links_total".into(),
                    pts.iter().map(|p| p.links_total).collect::<Vec<_>>().into(),
                );
                if show_vitals {
                    let f = |g: fn(&(f32, f32, f32)) -> f32| {
                        pts.iter()
                            .map(|p| p.breathing.as_ref().map_or(Value::Null, |b| g(b).into()))
                            .collect::<Vec<_>>()
                    };
                    m.insert("breathing_bpm".into(), f(|b| b.0).into());
                    m.insert("breathing_conf".into(), f(|b| b.1).into());
                    m.insert("breath_rms".into(), f(|b| b.2).into());
                }
                Value::Object(m)
            })
            .collect();

        let links: Vec<Value> = if principal.can_see_system() {
            self.links
                .iter()
                .filter(|((id, _), _)| want(id))
                .map(|((id, link), ring)| {
                    let pts = select(&ring.points, since_t_s, max_points, |p| p.t_s);
                    json!({
                        "space_id": id,
                        "rx": link.rx,
                        "tx": link.tx,
                        "points": pts.len(),
                        "t_s": pts.iter().map(|p| p.t_s).collect::<Vec<_>>(),
                        "unix_ms": pts.iter().map(|p| p.unix_ms).collect::<Vec<_>>(),
                        "up": pts.iter().map(|p| p.up).collect::<Vec<_>>(),
                        "rate_hz": pts.iter().map(|p| p.rate_hz).collect::<Vec<_>>(),
                        "loss": pts.iter().map(|p| p.loss).collect::<Vec<_>>(),
                        "best_bnr": pts.iter().map(|p| p.best_bnr).collect::<Vec<_>>(),
                        "rssi_dbm": pts.iter().map(|p| p.rssi_dbm).collect::<Vec<_>>(),
                    })
                })
                .collect()
        } else {
            Vec::new()
        };

        json!({
            "cadence_s": self.cfg.cadence_s,
            "window_s": self.cfg.window_s,
            "capacity": self.cfg.capacity(),
            "spaces": spaces,
            "links": links,
        })
    }
}

/// `since` filter then stride-decimation, keeping the newest point so the
/// right-hand end of a chart is the current value.
fn select<T>(
    points: &VecDeque<T>,
    since_t_s: Option<f64>,
    max_points: usize,
    t: fn(&T) -> f64,
) -> Vec<&T> {
    let from: Vec<&T> = points
        .iter()
        .filter(|p| since_t_s.is_none_or(|s| t(p) > s))
        .collect();
    if max_points == 0 || from.len() <= max_points {
        return from;
    }
    let stride = from.len().div_ceil(max_points);
    let mut out: Vec<&T> = from.iter().copied().rev().step_by(stride).collect();
    out.reverse();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roles::Role;
    use saans_health::types::Breathing;

    fn principal(role: Role) -> Principal {
        Principal {
            name: "x".into(),
            role,
            spaces: None,
        }
    }

    fn sample(t_s: f64, motion: f64, bpm: Option<f64>) -> SpaceSample {
        SpaceSample {
            t_s,
            occupied: Some(true),
            presence_index: motion,
            motion_index: motion,
            links_usable: 2,
            links_total: 2,
            breathing: bpm.map(|bpm| Breathing {
                link: LinkId { rx: 1, tx: 2 },
                bpm,
                confidence: 0.8,
                band_rms: 0.01,
                noise_rms: None,
            }),
            heart: None,
        }
    }

    fn filled(cfg: HistoryConfig, n: usize) -> History {
        let mut h = History::new(cfg);
        for i in 0..n {
            let t = i as f64 * cfg.cadence_s;
            h.record_space(
                "lab",
                1_700_000_000_000 + (t * 1000.0) as i64,
                &sample(t, 0.1 * i as f64, Some(14.0)),
                true,
            );
        }
        h
    }

    fn all(h: &History, role: Role) -> Value {
        h.to_json(
            &principal(role),
            &|_| true,
            &|id| id.to_uppercase(),
            None,
            None,
            0,
        )
    }

    #[test]
    fn the_ring_is_bounded_by_the_window() {
        let cfg = HistoryConfig {
            window_s: 20.0,
            cadence_s: 2.0,
        };
        assert_eq!(cfg.capacity(), 10);
        let h = filled(cfg, 100);
        assert_eq!(h.points(), 10, "100 samples, 10 kept");
        let v = all(&h, Role::Admin);
        let t: Vec<f64> = serde_json::from_value(v["spaces"][0]["t_s"].clone()).unwrap();
        assert_eq!(
            t,
            vec![180.0, 182.0, 184.0, 186.0, 188.0, 190.0, 192.0, 194.0, 196.0, 198.0],
            "the newest window"
        );
    }

    #[test]
    fn samples_faster_than_the_cadence_are_dropped() {
        let cfg = HistoryConfig {
            window_s: 3600.0,
            cadence_s: 2.0,
        };
        let mut h = History::new(cfg);
        // A 0.5 s tick: only every fourth sample is recorded.
        for i in 0..40 {
            let t = i as f64 * 0.5;
            h.record_space("lab", 0, &sample(t, 1.0, None), false);
        }
        assert_eq!(h.points(), 10);
        let v = all(&h, Role::Admin);
        let t: Vec<f64> = serde_json::from_value(v["spaces"][0]["t_s"].clone()).unwrap();
        assert_eq!(t.first(), Some(&0.0));
        assert_eq!(t.last(), Some(&18.0));
    }

    #[test]
    fn time_going_backwards_restarts_the_series() {
        let cfg = HistoryConfig {
            window_s: 3600.0,
            cadence_s: 2.0,
        };
        let mut h = filled(cfg, 5);
        assert_eq!(h.points(), 5);
        // A replay starting over must not leave a series that goes back.
        h.record_space("lab", 0, &sample(0.0, 0.0, None), false);
        let v = all(&h, Role::Admin);
        let t: Vec<f64> = serde_json::from_value(v["spaces"][0]["t_s"].clone()).unwrap();
        assert_eq!(t, vec![0.0]);
    }

    #[test]
    fn decimation_keeps_the_newest_point_and_never_invents_one() {
        let h = filled(
            HistoryConfig {
                window_s: 3600.0,
                cadence_s: 2.0,
            },
            100,
        );
        let v = h.to_json(
            &principal(Role::Admin),
            &|_| true,
            &|id| id.into(),
            None,
            None,
            10,
        );
        let t: Vec<f64> = serde_json::from_value(v["spaces"][0]["t_s"].clone()).unwrap();
        assert!(t.len() <= 10, "asked for 10, got {}", t.len());
        assert_eq!(
            t.last(),
            Some(&198.0),
            "the newest sample is always the last"
        );
        let motion: Vec<f64> =
            serde_json::from_value(v["spaces"][0]["motion_index"].clone()).unwrap();
        // Every value returned is one that was recorded: t/2 * 0.1.
        for (ts, m) in t.iter().zip(&motion) {
            assert!(
                (m - ts / 2.0 * 0.1).abs() < 1e-5,
                "t={ts} motion={m} is not a measured value"
            );
        }
    }

    #[test]
    fn since_returns_only_later_points() {
        let h = filled(
            HistoryConfig {
                window_s: 3600.0,
                cadence_s: 2.0,
            },
            10,
        );
        let v = h.to_json(
            &principal(Role::Admin),
            &|_| true,
            &|id| id.into(),
            None,
            Some(10.0),
            0,
        );
        let t: Vec<f64> = serde_json::from_value(v["spaces"][0]["t_s"].clone()).unwrap();
        assert_eq!(t, vec![12.0, 14.0, 16.0, 18.0], "strictly after 10 s");
    }

    #[test]
    fn breathing_needs_consent_and_a_role() {
        let h = filled(
            HistoryConfig {
                window_s: 3600.0,
                cadence_s: 2.0,
            },
            5,
        );
        let admin = all(&h, Role::Admin);
        assert_eq!(admin["spaces"][0]["breathing_bpm"][0], json!(14.0));

        // Ops has the room but never the person.
        let ops = all(&h, Role::Ops);
        assert!(ops["spaces"][0].get("breathing_bpm").is_none());
        assert_eq!(ops["spaces"][0]["motion_index"][0], json!(0.0));

        // Without consent the series is not even recorded, so no role
        // can produce it.
        let mut no = History::new(HistoryConfig {
            window_s: 3600.0,
            cadence_s: 2.0,
        });
        no.record_space("lab", 0, &sample(0.0, 1.0, Some(14.0)), false);
        let v = no.to_json(
            &principal(Role::Admin),
            &|_| false,
            &|id| id.into(),
            None,
            None,
            0,
        );
        assert!(v["spaces"][0].get("breathing_bpm").is_none());
    }

    #[test]
    fn link_series_are_for_system_roles_only() {
        let mut h = History::new(HistoryConfig {
            window_s: 3600.0,
            cadence_s: 2.0,
        });
        let link = LinkId { rx: 1, tx: 2 };
        for i in 0..4 {
            h.record_link(
                "lab",
                link,
                LinkPoint {
                    t_s: i as f64 * 2.0,
                    unix_ms: 0,
                    up: true,
                    rate_hz: 20.0,
                    loss: 0.01,
                    best_bnr: 3.0,
                    rssi_dbm: -50,
                },
            );
        }
        assert_eq!(all(&h, Role::Admin)["links"][0]["rate_hz"][0], json!(20.0));
        assert_eq!(
            all(&h, Role::Ops)["links"].as_array().map(Vec::len),
            Some(1)
        );
        assert_eq!(
            all(&h, Role::Safety)["links"].as_array().map(Vec::len),
            Some(0),
            "safety is not ops"
        );
        assert_eq!(
            all(&h, Role::Occupancy)["links"].as_array().map(Vec::len),
            Some(0)
        );
    }

    #[test]
    fn a_space_outside_scope_has_no_series() {
        let h = filled(
            HistoryConfig {
                window_s: 3600.0,
                cadence_s: 2.0,
            },
            5,
        );
        let p = Principal {
            name: "x".into(),
            role: Role::Admin,
            spaces: Some(["other".to_string()].into()),
        };
        let v = h.to_json(&p, &|_| true, &|id| id.into(), None, None, 0);
        assert_eq!(v["spaces"].as_array().map(Vec::len), Some(0));
    }

    #[test]
    fn capacity_is_clamped_however_absurd_the_config() {
        assert_eq!(
            HistoryConfig {
                window_s: 0.0,
                cadence_s: 2.0
            }
            .capacity(),
            1
        );
        assert_eq!(
            HistoryConfig {
                window_s: 1e12,
                cadence_s: 0.5
            }
            .capacity(),
            (MAX_WINDOW_S / MIN_CADENCE_S) as usize + 1
        );
    }
}
