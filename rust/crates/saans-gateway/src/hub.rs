//! The hub wraps the deterministic [`Engine`] with everything that has a
//! side effect: event logs, history persistence, retention, the replay
//! `--events-out` file, and the snapshots the HTTP side serves. Live mode
//! and replay drive the same hub; only the source of `t_s` differs.
//!
//! ## Time
//! `t_s` is seconds on a monotonic clock, anchored so that it equals local
//! wall-clock seconds since 1970 (in `utc_offset_minutes`) when the clock
//! starts: at gateway start in live mode, at the first packet in replay.
//! The anchor matters because `saans_health::DailySummary::day` is
//! `floor(t_s / 86400)`: with a clock that restarted at 0 every run, the
//! persisted history would collide with "day 0" after each restart. The
//! clock itself never jumps (NTP steps do not move it); the wall-clock
//! `time` recorded with each event comes from the system clock.

use crate::config::GatewayConfig;
use crate::engine::{Engine, Ingested, TICK_S};
use crate::privacy;
use crate::shared::Shared;
use crate::store::{self, EventLog};
use crate::timefmt::{iso8601, now_unix_ms};
use saans_health::{LinkId, Severity};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

pub const HISTORY_SAVE_S: f64 = 600.0;
pub const RETENTION_CHECK_S: f64 = 3600.0;
pub const ADVISORY: &str = "advisory — not a medical device";

/// How a `t_s` maps to Unix wall-clock milliseconds.
#[derive(Debug, Clone, Copy)]
pub enum Wall {
    /// Live: the system clock now (events are stamped as they happen).
    System,
    /// Replay and manual clocks: `unix_ms_at_t0 + (t - t0)`.
    Anchored { t0: f64, unix_ms_at_t0: i64 },
}

impl Wall {
    pub fn unix_ms(&self, t_s: f64, now_t_s: f64) -> i64 {
        match *self {
            Wall::System => now_unix_ms() - ((now_t_s - t_s) * 1000.0).round() as i64,
            Wall::Anchored { t0, unix_ms_at_t0 } => unix_ms_at_t0 + ((t_s - t0) * 1000.0).round() as i64,
        }
    }
}

/// Local-epoch seconds for a Unix time: the anchor of `t_s`.
pub fn local_epoch_s(unix_ms: i64, offset_min: i32) -> f64 {
    (unix_ms + offset_min as i64 * 60_000) as f64 / 1000.0
}

fn severity_str(s: Severity) -> &'static str {
    match s {
        Severity::Info => "info",
        Severity::Advisory => "advisory",
        Severity::Check => "check",
    }
}

/// The event type an active-alert name in `SpaceState` comes from.
fn alert_event_type(alert: &str) -> &str {
    match alert {
        "blind" => "space_blind",
        "inactivity_no_breathing" => "inactivity",
        other => other,
    }
}

fn alert_default_severity(alert: &str) -> &'static str {
    match alert {
        "possible_collapse" | "inactivity_no_breathing" => "check",
        _ => "advisory",
    }
}

pub struct Hub {
    pub engine: Engine,
    pub cfg: GatewayConfig,
    shared: Option<Arc<Shared>>,
    log: Option<EventLog>,
    out: Option<BufWriter<File>>,
    wall: Wall,
    /// Per space: (event type, severity) -> count, after redaction.
    pub counts: Vec<BTreeMap<(String, &'static str), u64>>,
    last_by_type: Vec<HashMap<String, Arc<Value>>>,
    pub events_emitted: u64,
    /// Events withheld by the consent gate (should stay 0).
    pub events_withheld: u64,
    persist: bool,
    t_start: f64,
    now_t: f64,
    history_saved_t: Option<f64>,
    history_errors: u64,
    history_last_error: Option<String>,
    retention_t: Option<f64>,
    rate_window: VecDeque<(f64, u64)>,
    /// Startup warnings, shown in /api/health.
    pub notes: Vec<String>,
}

impl Hub {
    /// `persist`: write event logs and history under `data_dir` and load
    /// what is there (live mode). Replay never touches `data_dir`.
    pub fn new(cfg: GatewayConfig, shared: Option<Arc<Shared>>, persist: bool, wall: Wall, t_start: f64) -> Self {
        let engine = Engine::new(&cfg);
        let n = engine.spaces.len();
        let mut hub = Self {
            engine,
            shared,
            log: None,
            out: None,
            wall,
            counts: vec![BTreeMap::new(); n],
            last_by_type: vec![HashMap::new(); n],
            events_emitted: 0,
            events_withheld: 0,
            persist,
            t_start,
            now_t: t_start,
            history_saved_t: None,
            history_errors: 0,
            history_last_error: None,
            retention_t: None,
            rate_window: VecDeque::new(),
            notes: Vec::new(),
            cfg,
        };
        if persist {
            hub.load_persisted();
        }
        for s in &hub.cfg.spaces {
            if s.links.is_empty() {
                hub.notes.push(format!("space {:?} lists no links: it will always be unknown", s.id));
            }
        }
        hub
    }

    fn load_persisted(&mut self) {
        let dir = self.cfg.data_dir.clone();
        match EventLog::new(&dir, self.cfg.utc_offset_minutes) {
            Ok(l) => self.log = Some(l),
            Err(e) => self.notes.push(format!("cannot create data_dir {}: {e}; events are not being logged", dir.display())),
        }
        let (history, err) = store::load_history(&dir);
        if let Some(e) = err {
            self.notes.push(e);
        }
        for (id, h) in history {
            match self.engine.space_index(&id) {
                Some(i) => self.engine.spaces[i].monitor.import_history(h),
                None => self.notes.push(format!("history.json has space {id:?}, which is not configured; kept on disk")),
            }
        }
        if let Some(shared) = &self.shared {
            for mut v in store::load_recent_events(&dir, crate::shared::EVENT_RING) {
                let space = v.get("space_id").and_then(Value::as_str).unwrap_or("").to_string();
                let unix_ms = v.get("unix_ms").and_then(Value::as_i64).unwrap_or(0);
                // Re-apply today's consent setting: a space that has since
                // opted out must not show yesterday's breathing events.
                let vitals = self.cfg.spaces.iter().any(|s| s.id == space && s.vitals);
                if privacy::event_allowed(&v, vitals) {
                    privacy::redact_value(&mut v, vitals);
                    shared.push_event(&space, unix_ms, v);
                }
            }
        }
    }

    /// Write every event (as JSON lines) to `path` as well (replay).
    pub fn set_events_out(&mut self, path: &Path) -> std::io::Result<()> {
        self.out = Some(BufWriter::new(File::create(path)?));
        Ok(())
    }

    pub fn ingest(&mut self, data: &[u8], t_s: f64, src: Option<SocketAddr>) -> Ingested {
        self.now_t = self.now_t.max(t_s);
        self.engine.ingest(data, t_s, src)
    }

    /// One 0.5 s step: sample, detect, record, publish.
    pub fn tick(&mut self, t_s: f64) {
        self.now_t = self.now_t.max(t_s);
        let events = self.engine.tick(t_s);
        let offset = self.cfg.utc_offset_minutes;
        for (i, ev) in events {
            let vitals = self.engine.spaces[i].config.vitals;
            let Ok(mut v) = serde_json::to_value(&ev) else { continue };
            if !privacy::event_allowed(&v, vitals) {
                self.events_withheld += 1;
                continue;
            }
            privacy::redact_value(&mut v, vitals);
            let Value::Object(map) = v else { continue };
            let unix_ms = self.wall.unix_ms(ev.t_s, t_s);
            let time = iso8601(unix_ms, offset);
            let line = store::event_line(&time, unix_ms, &map);
            if let Some(log) = &mut self.log {
                log.append(unix_ms, &line);
            }
            if let Some(out) = &mut self.out {
                let _ = writeln!(out, "{line}");
            }
            let ty = map.get("type").and_then(Value::as_str).unwrap_or("unknown").to_string();
            *self.counts[i].entry((ty.clone(), severity_str(ev.severity))).or_default() += 1;
            self.events_emitted += 1;
            let mut record = map;
            record.insert("time".into(), Value::String(time));
            record.insert("unix_ms".into(), unix_ms.into());
            let record = Value::Object(record);
            match &self.shared {
                Some(shared) => {
                    let stored = shared.push_event(&ev.space_id, unix_ms, record);
                    shared.broadcast(json!({"type": "event", "event": &*stored}).to_string().into());
                    self.last_by_type[i].insert(ty, stored);
                }
                None => {
                    self.last_by_type[i].insert(ty, Arc::new(record));
                }
            }
        }
        if self.persist {
            self.housekeeping(t_s);
        }
        if self.shared.is_some() {
            self.publish(t_s);
        }
    }

    fn housekeeping(&mut self, t_s: f64) {
        if self.retention_t.is_none_or(|r| t_s - r >= RETENTION_CHECK_S) {
            self.retention_t = Some(t_s);
            let removed = store::prune_logs(
                &self.cfg.data_dir,
                self.cfg.event_retention_days,
                self.cfg.utc_offset_minutes,
                self.wall.unix_ms(t_s, t_s),
            );
            for p in removed {
                crate::log(&format!("retention: deleted {}", p.display()));
            }
        }
        let since = self.history_saved_t.unwrap_or(self.t_start);
        if t_s - since >= HISTORY_SAVE_S {
            self.save_history(t_s);
        }
    }

    pub fn save_history(&mut self, t_s: f64) {
        self.history_saved_t = Some(t_s);
        let history: BTreeMap<String, _> =
            self.engine.spaces.iter().map(|s| (s.config.id.clone(), s.monitor.export_history())).collect();
        let time = iso8601(self.wall.unix_ms(t_s, t_s), self.cfg.utc_offset_minutes);
        if let Err(e) = store::save_history(&self.cfg.data_dir, &history, &time) {
            self.history_errors += 1;
            self.history_last_error = Some(e.to_string());
            crate::log(&format!("cannot save history: {e}"));
        }
    }

    /// Persist and flush; call once when stopping.
    pub fn shutdown(&mut self, t_s: f64) {
        if self.persist {
            self.save_history(t_s);
        }
        if let Some(out) = &mut self.out {
            let _ = out.flush();
        }
    }

    // ------------------------------------------------------------ publish

    pub fn publish(&mut self, t_s: f64) {
        let Some(shared) = self.shared.clone() else { return };
        let datagrams = self.engine.stats.datagrams;
        self.rate_window.push_back((t_s, datagrams));
        while self.rate_window.len() > 2 && t_s - self.rate_window[1].0 >= 10.0 {
            self.rate_window.pop_front();
        }
        let spaces: Vec<(String, Value)> =
            (0..self.engine.spaces.len()).map(|i| (self.engine.spaces[i].config.id.clone(), self.space_json(i, t_s))).collect();
        let state_msg: Arc<str> =
            json!({"type": "state", "t_s": t_s, "spaces": spaces.iter().map(|(_, v)| v).collect::<Vec<_>>()})
                .to_string()
                .into();
        let links = self.links_json(t_s);
        let health = self.health_json(t_s, &shared);
        let metrics = self.metrics_text(t_s, &shared);
        let unix_ms = self.wall.unix_ms(t_s, t_s);
        if shared.outputs.receiver_count() > 0 {
            shared.emit(crate::outputs::OutputMsg::States { unix_ms, spaces: Arc::new(spaces.clone()) });
            shared.emit(crate::outputs::OutputMsg::System { unix_ms, links: Arc::new(links.clone()), health: Arc::new(health.clone()) });
        }
        {
            let mut p = shared.write();
            p.t_s = t_s;
            p.spaces = spaces;
            p.links = links;
            p.health = health;
            p.metrics = metrics;
            p.state_msg = Some(state_msg.clone());
        }
        shared.broadcast(state_msg);
    }

    /// A space's state for the API: `SpaceState` plus its name, privacy
    /// badges and active alerts with severity and evidence, redacted.
    pub fn space_json(&self, i: usize, t_s: f64) -> Value {
        let sp = &self.engine.spaces[i];
        let vitals = sp.config.vitals;
        let mut v = serde_json::to_value(&sp.state).unwrap_or_else(|_| json!({}));
        let raw: Vec<String> = sp.raw_mode_links(t_s).iter().map(LinkId::to_string).collect();
        let alerts: Vec<Value> = sp
            .state
            .active_alerts
            .iter()
            .filter(|a| privacy::alert_allowed(a, vitals))
            .map(|a| {
                let last = self.last_by_type[i].get(alert_event_type(a));
                json!({
                    "name": a,
                    "severity": last.and_then(|e| e.get("severity").cloned()).unwrap_or(alert_default_severity(a).into()),
                    "evidence": last.and_then(|e| e.get("evidence").cloned()),
                    "since_s": last.and_then(|e| e.get("t_start_s").cloned().filter(|x| !x.is_null()).or_else(|| e.get("t_s").cloned())),
                    "notice": ADVISORY,
                })
            })
            .collect();
        if let Value::Object(m) = &mut v {
            m.insert("name".into(), GatewayConfig::space_name(&sp.config).into());
            m.insert("collapse_detection".into(), sp.config.collapse_detection.into());
            m.insert("raw_mode".into(), (!raw.is_empty()).into());
            m.insert("raw_mode_links".into(), raw.into());
            m.insert("vitals_gated_links".into(), sp.vitals_gated_links.iter().map(LinkId::to_string).collect::<Vec<_>>().into());
            m.insert("active_alerts".into(), alerts.iter().map(|a| a["name"].clone()).collect::<Vec<_>>().into());
            m.insert("alerts".into(), alerts.into());
        }
        privacy::redact_value(&mut v, vitals);
        v
    }

    pub fn links_json(&self, t_s: f64) -> Value {
        Value::Array(
            self.engine
                .spaces
                .iter()
                .map(|sp| {
                    let mut reports = serde_json::to_value(sp.system.link_report(&sp.fusion, t_s)).unwrap_or(json!([]));
                    privacy::redact_value(&mut reports, sp.config.vitals);
                    json!({
                        "space_id": sp.config.id,
                        "name": GatewayConfig::space_name(&sp.config),
                        "links_usable": sp.fusion.links.values().filter(|l| l.usable(t_s)).count(),
                        "links": reports,
                    })
                })
                .collect(),
        )
    }

    fn datagram_rate(&self) -> f64 {
        match (self.rate_window.front(), self.rate_window.back()) {
            (Some(a), Some(b)) if b.0 > a.0 => (b.1 - a.1) as f64 / (b.0 - a.0),
            _ => 0.0,
        }
    }

    pub fn health_json(&self, t_s: f64, shared: &Shared) -> Value {
        let e = &self.engine;
        let raw_links: Vec<Value> = e
            .spaces
            .iter()
            .flat_map(|sp| sp.raw_mode_links(t_s).into_iter().map(move |l| json!({"space_id": sp.config.id, "rx": l.rx, "tx": l.tx})))
            .collect();
        let gated: Vec<Value> = e
            .spaces
            .iter()
            .flat_map(|sp| sp.vitals_gated_links.iter().map(move |l| json!({"space_id": sp.config.id, "rx": l.rx, "tx": l.tx})))
            .collect();
        let configured: usize = e.spaces.iter().map(|s| s.fusion.links.len()).sum();
        let usable: usize = e.spaces.iter().map(|s| s.fusion.links.values().filter(|l| l.usable(t_s)).count()).sum();
        let blind: Vec<&str> =
            e.spaces.iter().filter(|s| s.sample.is_some_and(|x| x.occupied.is_none())).map(|s| s.config.id.as_str()).collect();
        let mut h = json!({
            "version": env!("CARGO_PKG_VERSION"),
            "mode": shared.mode,
            "uptime_s": shared.started.elapsed().as_secs_f64(),
            "t_s": t_s,
            "time": iso8601(self.wall.unix_ms(t_s, t_s), self.cfg.utc_offset_minutes),
            "datagrams_total": e.stats.datagrams,
            "datagrams_per_s": self.datagram_rate(),
            "decode_errors_total": e.stats.errors.total(),
            "ingest": &e.stats,
            "unknown_links": e.unknown.values().collect::<Vec<_>>(),
            "raw_mode_links_active": raw_links.len(),
            "raw_mode_links": raw_links,
            "vitals_gated_links": gated,
            "spaces": e.spaces.len(),
            "spaces_blind": blind,
            "links_configured": configured,
            "links_usable": usable,
            "events_emitted": self.events_emitted,
            "events_withheld": self.events_withheld,
            "ws_clients": shared.ws_clients(),
            "notes": &self.notes,
        });
        if let Some(log) = &self.log {
            h["event_log"] = json!({
                "dir": self.cfg.data_dir.display().to_string(),
                "written": log.written,
                "write_errors": log.write_errors,
                "last_error": log.last_error,
                "retention_days": self.cfg.event_retention_days,
            });
            h["history"] = json!({
                "last_saved_t_s": self.history_saved_t,
                "errors": self.history_errors,
                "last_error": self.history_last_error,
            });
        }
        h
    }

    /// Prometheus text exposition (format 0.0.4). Deliberately no vital
    /// signs: a TSDB is not a place for health data (ADR-001).
    pub fn metrics_text(&self, t_s: f64, shared: &Shared) -> String {
        let e = &self.engine;
        let mut p = Prom::default();
        p.family("saans_gateway_uptime_seconds", "gauge", "Seconds since the gateway process started.");
        p.sample("saans_gateway_uptime_seconds", &[], shared.started.elapsed().as_secs_f64());
        p.family("saans_datagrams_total", "counter", "UDP datagrams received, by type.");
        p.sample("saans_datagrams_total", &[("type", "feature")], e.stats.feature_frames as f64);
        p.sample("saans_datagrams_total", &[("type", "raw")], e.stats.raw_datagrams as f64);
        p.sample("saans_datagrams_total", &[("type", "invalid")], e.stats.errors.total() as f64);
        p.family("saans_datagrams_per_second", "gauge", "Datagram rate over the last 10 s.");
        p.sample("saans_datagrams_per_second", &[], self.datagram_rate());
        p.family("saans_decode_errors_total", "counter", "Datagrams rejected, by reason.");
        let er = &e.stats.errors;
        for (reason, n) in [
            ("size", er.size),
            ("crc", er.crc),
            ("version", er.version),
            ("unknown_type", er.unknown_type),
            ("raw_malformed", er.raw_malformed),
            ("raw_subcarrier_mismatch", er.raw_subcarrier_mismatch),
            ("non_finite", er.non_finite),
        ] {
            p.sample("saans_decode_errors_total", &[("reason", reason)], n as f64);
        }
        p.family("saans_raw_feature_frames_total", "counter", "FeatureFrames the gateway computed from raw CSI.");
        p.sample("saans_raw_feature_frames_total", &[], e.stats.raw_feature_frames as f64);
        p.family("saans_vitals_gated_frames_total", "counter", "Frames that carried vitals for a space without consent (zeroed).");
        p.sample("saans_vitals_gated_frames_total", &[], e.stats.vitals_gated as f64);
        p.family("saans_unknown_link_frames_total", "counter", "Frames from links that no space lists.");
        for u in e.unknown.values() {
            let (rx, tx) = (u.rx.to_string(), u.tx.to_string());
            p.sample("saans_unknown_link_frames_total", &[("rx", &rx), ("tx", &tx), ("kind", u.kind)], u.frames as f64);
        }

        let mut link_rows = Vec::new();
        for sp in &e.spaces {
            for r in sp.system.link_report(&sp.fusion, t_s) {
                let usable = sp.fusion.links.get(&r.link).is_some_and(|l| l.usable(t_s));
                link_rows.push((sp.config.id.as_str(), r, usable));
            }
        }
        type LinkGauge = (&'static str, &'static str, &'static str, fn(&saans_health::system::LinkReport, bool) -> f64);
        let gauges: [LinkGauge; 8] = [
            ("saans_link_frames_total", "counter", "FeatureFrames received per link.", |r, _| r.frames as f64),
            ("saans_link_up", "gauge", "1 if the link sent a frame within 5 s.", |r, _| r.up as u8 as f64),
            ("saans_link_usable", "gauge", "1 if fresh, calibrated and fast enough to vote on presence.", |_, u| u as u8 as f64),
            ("saans_link_csi_rate_hz", "gauge", "Achieved CSI rate reported by the node.", |r, _| r.rate_hz),
            ("saans_link_loss_ratio", "gauge", "CSI sequence loss reported by the node (0-1).", |r, _| r.loss),
            ("saans_link_silent_seconds", "gauge", "Seconds since the link's last frame.", |r, _| r.silent_s),
            ("saans_link_rssi_dbm", "gauge", "Mean RSSI of the link's last frame interval.", |r, _| r.rssi_dbm as f64),
            ("saans_link_raw_mode", "gauge", "1 if the node is also streaming raw CSI (privacy notice).", |r, _| r.raw_mode as u8 as f64),
        ];
        for (name, kind, help, f) in gauges {
            p.family(name, kind, help);
            for (space, r, usable) in &link_rows {
                let (rx, tx) = (r.link.rx.to_string(), r.link.tx.to_string());
                p.sample(name, &[("space", space), ("rx", &rx), ("tx", &tx)], f(r, *usable));
            }
        }
        p.family("saans_link_calibration_age_hours", "gauge", "Age of the link's empty-room calibration.");
        for (space, r, _) in &link_rows {
            if let Some(age) = r.calib_age_h {
                let (rx, tx) = (r.link.rx.to_string(), r.link.tx.to_string());
                p.sample("saans_link_calibration_age_hours", &[("space", space), ("rx", &rx), ("tx", &tx)], age);
            }
        }

        // One family at a time: the text format requires each family's
        // lines to be one contiguous group.
        type SpaceGauge = (&'static str, &'static str, fn(&crate::engine::SpaceRt) -> f64);
        let space_gauges: [SpaceGauge; 4] = [
            ("saans_space_occupied", "1 occupied, 0 empty, -1 unknown (no usable link).", |sp| match sp.state.occupied {
                Some(true) => 1.0,
                Some(false) => 0.0,
                None => -1.0,
            }),
            ("saans_space_links_usable", "Links currently usable in the space.", |sp| sp.state.links_usable as f64),
            ("saans_space_links_total", "Links configured for the space.", |sp| sp.fusion.links.len() as f64),
            ("saans_space_active_alerts", "Alerts currently active in the space.", |sp| {
                sp.state.active_alerts.iter().filter(|a| privacy::alert_allowed(a, sp.config.vitals)).count() as f64
            }),
        ];
        for (name, help, f) in space_gauges {
            p.family(name, "gauge", help);
            for sp in &e.spaces {
                p.sample(name, &[("space", sp.config.id.as_str())], f(sp));
            }
        }
        p.family("saans_events_total", "counter", "Health and system events emitted, by type and severity.");
        for (i, counts) in self.counts.iter().enumerate() {
            let id = e.spaces[i].config.id.as_str();
            for ((ty, sev), n) in counts {
                p.sample("saans_events_total", &[("space", id), ("type", ty), ("severity", sev)], *n as f64);
            }
        }
        p.family("saans_ws_clients", "gauge", "Connected WebSocket clients.");
        p.sample("saans_ws_clients", &[], shared.ws_clients() as f64);
        p.out
    }

    /// Per-space event counts by type, for the replay summary.
    pub fn counts_by_type(&self) -> Vec<(String, BTreeMap<String, u64>)> {
        self.engine
            .spaces
            .iter()
            .zip(&self.counts)
            .map(|(sp, c)| {
                let mut by: BTreeMap<String, u64> = BTreeMap::new();
                for ((ty, _), n) in c {
                    *by.entry(ty.clone()).or_default() += n;
                }
                (sp.config.id.clone(), by)
            })
            .collect()
    }

    pub fn now_t(&self) -> f64 {
        self.now_t
    }
}

/// Run ticks on the 0.5 s grid from `*next` up to `t` (inclusive).
pub fn tick_until(hub: &mut Hub, next: &mut f64, t: f64) {
    while *next <= t + 1e-9 {
        hub.tick(*next);
        *next += TICK_S;
    }
}

#[derive(Default)]
struct Prom {
    out: String,
}

impl Prom {
    fn family(&mut self, name: &str, kind: &str, help: &str) {
        let _ = writeln!(self.out, "# HELP {name} {}", help.replace('\\', "\\\\").replace('\n', "\\n"));
        let _ = writeln!(self.out, "# TYPE {name} {kind}");
    }

    fn sample(&mut self, name: &str, labels: &[(&str, &str)], value: f64) {
        self.out.push_str(name);
        if !labels.is_empty() {
            self.out.push('{');
            for (i, (k, v)) in labels.iter().enumerate() {
                if i > 0 {
                    self.out.push(',');
                }
                let v = v.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
                let _ = write!(self.out, "{k}=\"{v}\"");
            }
            self.out.push('}');
        }
        let v = if value.is_nan() {
            "NaN".to_string()
        } else if value.is_infinite() {
            if value > 0.0 { "+Inf" } else { "-Inf" }.to_string()
        } else {
            format!("{value}")
        };
        let _ = writeln!(self.out, " {v}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prom_escapes_labels() {
        let mut p = Prom::default();
        p.family("x_total", "counter", "Help.");
        p.sample("x_total", &[("a", "q\"b\\n")], 1.5);
        p.sample("x_total", &[], f64::INFINITY);
        assert_eq!(p.out, "# HELP x_total Help.\n# TYPE x_total counter\nx_total{a=\"q\\\"b\\\\n\"} 1.5\nx_total +Inf\n");
    }

    #[test]
    fn anchored_wall_clock() {
        let w = Wall::Anchored { t0: 100.0, unix_ms_at_t0: 5_000 };
        assert_eq!(w.unix_ms(101.25, 0.0), 6_250);
    }
}
