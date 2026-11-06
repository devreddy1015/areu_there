//! The room stream's messages (spec/livedemo.md §3.4, §4): `state`,
//! `frame` and `error`, as JSON. Pure: built from records, never from a
//! socket or a clock.
//!
//! Records are encoded by the functions the records file uses
//! (`saans_locate::records`), and truth is scored by the functions the
//! statistics use (`saans_locate::score`), so a frame, a file and a
//! summary cannot disagree.
//!
//! - Every message carries `mode` and `label`.
//! - A position is an `Outcome` estimate, which always has its radius, or
//!   it is truth.
//! - Truth exists only in mode `sim`. Elsewhere its key is never built,
//!   so redaction deletes rather than nulls (ADR-008 decisions 10, 15).
//! - People are `i`, their index; the phone has no name (decision 3).

use saans_locate::phone::EpochRecord;
use saans_locate::pipeline::TickRecord;
use saans_locate::records::{
    epoch_truth_json, link_json, outcome_json, provenance_name, report_json, tick_truth_json,
};
use saans_locate::score::{person_score, phone_score};
use saans_locate::stream::LinkValue;
use saans_locate::variant::Variant;
use saans_scene::Scene;
use serde_json::{json, Value};

use super::Embedded;
use crate::auth::sha256_hex;
use crate::shared::Mode;

/// The room stream's only label: every number on it is SIM.
pub(crate) const LABEL: &str = saans_locate::LABEL;

/// Whether frames of `mode` carry truth: `sim` only (ADR-008 decision 15).
/// Exhaustive, with no default branch.
pub(crate) fn shows_truth(mode: Mode) -> bool {
    match mode {
        Mode::Sim => true,
        Mode::Replay | Mode::Live => false,
    }
}

/// `scene` (ADR-008 decision 4): the embedded id, the scene's name, the
/// sha256 of the file as embedded and its provenance.
pub(crate) fn scene_block(e: Embedded, scene: &Scene) -> Value {
    json!({
        "id": e.id,
        "name": scene.name,
        "sha256": sha256_hex(e.text),
        "provenance": provenance_name(scene.provenance),
    })
}

/// `run` (§4): what a session fixes, and the run's index in it.
pub(crate) fn run_block(index: u64, variant: Variant, seed: u64, duration_s: f64) -> Value {
    json!({ "index": index, "variant": variant.as_str(), "seed": seed, "duration_s": duration_s })
}

/// `{type:"state", state, mode, label, scene, run}` (§3.4): the first
/// message of every session.
pub(crate) fn state_message(state: &str, mode: Mode, scene: &Value, run: &Value) -> Value {
    json!({ "type": "state", "state": state, "mode": mode.as_str(), "label": LABEL, "scene": scene, "run": run })
}

/// `{type:"error", mode, label, param, message}` (§3.4).
pub(crate) fn error_message(mode: Mode, param: &str, message: &str) -> Value {
    json!({ "type": "error", "mode": mode.as_str(), "label": LABEL, "param": param, "message": message })
}

/// One frame's inputs (§4). The record slices are the run's records with
/// `t <= T`, in time order; the ones from `new_*_from` on have
/// `T_prev < t <= T`.
pub(crate) struct Frame<'a> {
    pub(crate) seq: u64,
    pub(crate) run: Value,
    pub(crate) t: f64,
    pub(crate) mode: Mode,
    pub(crate) scene: &'a Value,
    pub(crate) speed: f64,
    pub(crate) effective_speed: Option<f64>,
    pub(crate) paused: bool,
    pub(crate) ended: bool,
    /// The canonical link order `(tx, rx)`, the order of every tick's links.
    pub(crate) links: &'a [(i64, i64)],
    pub(crate) has_phone: bool,
    pub(crate) ticks: &'a [TickRecord],
    /// One per entry of `ticks`.
    pub(crate) tick_links: &'a [Vec<LinkValue>],
    pub(crate) epochs: &'a [EpochRecord],
    pub(crate) new_ticks_from: usize,
    pub(crate) new_epochs_from: usize,
    /// §5's block, `empty_run` included.
    pub(crate) stats: Value,
}

impl Frame<'_> {
    pub(crate) fn to_json(&self) -> Value {
        let tick = self.ticks.last();
        let epoch = self.epochs.last();
        let persons = tick.map_or(
            Value::Null,
            |k| json!({ "t": k.t, "wlm": outcome_json(&k.wlm), "rti": outcome_json(&k.rti) }),
        );
        let links = match (tick, self.tick_links.last()) {
            (Some(k), Some(values)) => {
                let items: Vec<Value> = self
                    .links
                    .iter()
                    .zip(values)
                    .map(|(&(tx, rx), v)| link_json(tx, rx, v))
                    .collect();
                json!({ "t": k.t, "items": items })
            }
            _ => Value::Null,
        };
        let new_persons: Vec<Value> = self.ticks[self.new_ticks_from..]
            .iter()
            .map(|k| json!({ "t": k.t, "wlm": outcome_json(&k.wlm), "rti": outcome_json(&k.rti) }))
            .collect();
        let mut v = json!({
            "type": "frame",
            "seq": self.seq,
            "run": self.run,
            "t": self.t,
            "mode": self.mode.as_str(),
            "label": LABEL,
            "scene": self.scene,
            "speed": self.speed,
            "effective_speed": self.effective_speed,
            "paused": self.paused,
            "ended": self.ended,
            "state": if tick.is_some() { "running" } else { "warming_up" },
            "persons": persons,
            "links": links,
            "new": { "persons": new_persons },
            "stats": self.stats,
        });
        if self.has_phone {
            v["phone"] = epoch.map_or(Value::Null, |e| {
                let reports: Vec<Value> = e.reports.iter().map(report_json).collect();
                json!({ "t": e.t, "rtt": outcome_json(&e.rtt), "rssi": outcome_json(&e.rssi), "reports": reports })
            });
            v["new"]["phone"] = self.epochs[self.new_epochs_from..]
                .iter()
                .map(|e| json!({ "t": e.t, "rtt": outcome_json(&e.rtt), "rssi": outcome_json(&e.rssi) }))
                .collect();
        }
        if shows_truth(self.mode) {
            v["truth"] = self.truth(tick, epoch);
        }
        v
    }

    /// §4 `truth`: each part its record's own truth, and each estimate's
    /// score against it, so a drawn gap equals the printed error.
    /// `blocked` is computed from true positions, so it lives only here.
    fn truth(&self, tick: Option<&TickRecord>, epoch: Option<&EpochRecord>) -> Value {
        let persons = tick.map_or(Value::Null, |k| {
            json!({
                "t": k.t,
                "people": tick_truth_json(&k.truth),
                "wlm": person_score(&k.wlm, &k.truth).to_json(),
                "rti": person_score(&k.rti, &k.truth).to_json(),
            })
        });
        let mut truth = json!({ "persons": persons });
        if self.has_phone {
            truth["phone"] = epoch.map_or(Value::Null, |e| {
                let mut p = epoch_truth_json(e.truth, &e.reports);
                p["t"] = e.t.into();
                p["rtt"] = phone_score(&e.rtt, e.truth).to_json();
                p["rssi"] = phone_score(&e.rssi, e.truth).to_json();
                p
            });
        }
        truth
    }
}
