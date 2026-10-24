//! What the gateway keeps on disk, all under `data_dir`:
//!
//! - `events-YYYY-MM-DD.jsonl`: one JSON object per health event, with the
//!   wall-clock `time` (ISO 8601, configured UTC offset) and `unix_ms`
//!   first. One file per local day; a file is deleted once every event in
//!   it is older than `event_retention_days` (DPDP retention limit).
//! - `history.json`: each space's `DailySummary` list (trend baselines),
//!   written atomically every 10 minutes and on shutdown.
//!
//! Events are written already redacted (`privacy`), so a vitals-off space
//! never has breathing data on disk either.

use crate::timefmt::{date_string, local_day};
use saans_health::detectors::DailySummary;
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// `{"time":..., "unix_ms":..., <event fields>}` as one line (no newline).
pub fn event_line(time: &str, unix_ms: i64, event: &Map<String, Value>) -> String {
    let mut s = String::with_capacity(256);
    s.push_str("{\"time\":");
    s.push_str(&Value::String(time.to_string()).to_string());
    s.push_str(",\"unix_ms\":");
    s.push_str(&unix_ms.to_string());
    for (k, v) in event {
        if k == "time" || k == "unix_ms" {
            continue;
        }
        s.push(',');
        s.push_str(&Value::String(k.clone()).to_string());
        s.push(':');
        s.push_str(&v.to_string());
    }
    s.push('}');
    s
}

pub struct EventLog {
    dir: PathBuf,
    offset_min: i32,
    day: i64,
    file: Option<File>,
    pub written: u64,
    pub write_errors: u64,
    pub last_error: Option<String>,
}

impl EventLog {
    pub fn new(dir: &Path, offset_min: i32) -> io::Result<Self> {
        fs::create_dir_all(dir)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            offset_min,
            day: i64::MIN,
            file: None,
            written: 0,
            write_errors: 0,
            last_error: None,
        })
    }

    pub fn path_for_day(&self, day: i64) -> PathBuf {
        self.dir.join(format!("events-{}.jsonl", date_string(day)))
    }

    /// Append one line; failures are counted, never fatal.
    pub fn append(&mut self, unix_ms: i64, line: &str) {
        let day = local_day(unix_ms, self.offset_min);
        if day != self.day || self.file.is_none() {
            self.day = day;
            self.file = match OpenOptions::new()
                .create(true)
                .append(true)
                .open(self.path_for_day(day))
            {
                Ok(f) => Some(f),
                Err(e) => {
                    self.fail(e);
                    return;
                }
            };
        }
        let mut buf = Vec::with_capacity(line.len() + 1);
        buf.extend_from_slice(line.as_bytes());
        buf.push(b'\n');
        // One write per line (O_APPEND), so a crash cannot interleave lines.
        match self.file.as_mut().map(|f| f.write_all(&buf)) {
            Some(Ok(())) => self.written += 1,
            Some(Err(e)) => {
                self.file = None;
                self.fail(e);
            }
            None => {}
        }
    }

    fn fail(&mut self, e: io::Error) {
        self.write_errors += 1;
        self.last_error = Some(e.to_string());
    }
}

/// What the building did: swept on `event_retention_days`.
pub const EVENT_LOG_PREFIX: &str = "events-";
/// What people did with the system (`audit.rs`): swept on
/// `audit_retention_days`, its own clock (ADR-010). Both families live in
/// `data_dir`, so one dated-file sweep serves both, keyed by prefix.
pub const AUDIT_LOG_PREFIX: &str = "audit-";

fn log_file_day(name: &str, prefix: &str) -> Option<i64> {
    let date = name.strip_prefix(prefix)?.strip_suffix(".jsonl")?;
    let ms = crate::timefmt::parse_iso8601(date)?;
    (date.len() == 10).then(|| ms.div_euclid(86_400_000))
}

/// Dated logs of one family in `dir`, oldest first, as (local day, path).
pub fn list_logs_prefixed(dir: &Path, prefix: &str) -> Vec<(i64, PathBuf)> {
    let mut out: Vec<(i64, PathBuf)> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            Some((log_file_day(&name, prefix)?, e.path()))
        })
        .collect();
    out.sort();
    out
}

/// Event logs in `dir`, oldest first, as (local day, path).
pub fn list_logs(dir: &Path) -> Vec<(i64, PathBuf)> {
    list_logs_prefixed(dir, EVENT_LOG_PREFIX)
}

/// Delete event logs whose every event is older than `retention_days`.
pub fn prune_logs(dir: &Path, retention_days: u32, offset_min: i32, now_ms: i64) -> Vec<PathBuf> {
    prune_prefixed(dir, EVENT_LOG_PREFIX, retention_days, offset_min, now_ms)
}

/// Delete audit logs whose every line is older than `retention_days`.
/// Same day-boundary rule as the event logs, separate clock (ADR-010).
pub fn prune_audit_logs(
    dir: &Path,
    retention_days: u32,
    offset_min: i32,
    now_ms: i64,
) -> Vec<PathBuf> {
    prune_prefixed(dir, AUDIT_LOG_PREFIX, retention_days, offset_min, now_ms)
}

fn prune_prefixed(
    dir: &Path,
    prefix: &str,
    retention_days: u32,
    offset_min: i32,
    now_ms: i64,
) -> Vec<PathBuf> {
    let today = local_day(now_ms, offset_min);
    let mut removed = Vec::new();
    for (day, path) in list_logs_prefixed(dir, prefix) {
        // The file's newest possible event is at the end of `day`.
        if today - (day + 1) >= retention_days as i64 && fs::remove_file(&path).is_ok() {
            removed.push(path);
        }
    }
    removed
}

/// Up to `max` most recent events from the logs, oldest first.
pub fn load_recent_events(dir: &Path, max: usize) -> Vec<Value> {
    let mut chunks: Vec<Vec<Value>> = Vec::new();
    let mut n = 0;
    for (_, path) in list_logs(dir).into_iter().rev() {
        let Ok(f) = File::open(&path) else { continue };
        let mut day: Vec<Value> = BufReader::new(f)
            .lines()
            .map_while(Result::ok)
            .filter_map(|l| serde_json::from_str(&l).ok())
            .collect();
        if day.len() > max - n {
            day.drain(..day.len() - (max - n));
        }
        n += day.len();
        chunks.push(day);
        if n >= max {
            break;
        }
    }
    chunks.into_iter().rev().flatten().collect()
}

pub fn history_path(dir: &Path) -> PathBuf {
    dir.join("history.json")
}

/// Read `history.json`. A file that does not parse is moved aside (never
/// overwritten) and the gateway starts with empty baselines.
pub fn load_history(dir: &Path) -> (BTreeMap<String, Vec<DailySummary>>, Option<String>) {
    let path = history_path(dir);
    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return (BTreeMap::new(), None),
        Err(e) => return (BTreeMap::new(), Some(format!("{}: {e}", path.display()))),
    };
    let parsed = serde_json::from_str::<Value>(&text).ok().and_then(|v| {
        let spaces = v.get("spaces")?.as_object()?;
        spaces
            .iter()
            .map(|(k, v)| {
                serde_json::from_value::<Vec<DailySummary>>(v.clone())
                    .ok()
                    .map(|h| (k.clone(), h))
            })
            .collect::<Option<BTreeMap<_, _>>>()
    });
    match parsed {
        Some(h) => (h, None),
        None => {
            let aside = dir.join(format!(
                "history.json.unreadable-{}",
                crate::timefmt::now_unix_ms()
            ));
            let moved = fs::rename(&path, &aside).is_ok();
            let what = if moved {
                format!("moved to {}", aside.display())
            } else {
                "left in place".into()
            };
            (
                BTreeMap::new(),
                Some(format!(
                    "{} does not parse; {what}; starting with empty baselines",
                    path.display()
                )),
            )
        }
    }
}

/// Drop day summaries older than `retention_days`, and spaces left with
/// nothing. Returns the number of summaries removed.
///
/// `today` and `DailySummary::day` must be the same index. They are: a
/// summary's `day` is `floor(t_s / 86400)` and in live mode `t_s` is
/// `local_epoch_s`, local-adjusted unix seconds (`hub.rs:57`), so the
/// floor of it over a day is exactly `timefmt::local_day`. Only live mode
/// persists (`Hub::new`'s `persist`, false for replay and simulate), which
/// is what makes the comparison meaningful -- under a manual clock `day`
/// counts from `t0` and, as `hub.rs` puts it, "persisted history would
/// collide with day 0 after each restart".
///
/// The boundary matches `prune_logs`: a summary for `day` is removed once
/// `today - day >= retention_days`, so a summary survives its whole
/// retention period and is dropped on the first sweep after it ends.
pub fn prune_history(
    history: &mut BTreeMap<String, Vec<DailySummary>>,
    retention_days: u32,
    today: i64,
) -> usize {
    let mut removed = 0;
    for days in history.values_mut() {
        let before = days.len();
        days.retain(|s| today - s.day < retention_days as i64);
        removed += before - days.len();
    }
    history.retain(|_, days| !days.is_empty());
    removed
}

/// Write `history.json` atomically (temp file + rename).
pub fn save_history(
    dir: &Path,
    history: &BTreeMap<String, Vec<DailySummary>>,
    saved_time: &str,
) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    let doc = serde_json::json!({ "version": 1, "saved": saved_time, "spaces": history });
    let tmp = dir.join("history.json.tmp");
    {
        let mut f = File::create(&tmp)?;
        f.write_all(
            serde_json::to_string_pretty(&doc)
                .map_err(io::Error::other)?
                .as_bytes(),
        )?;
        f.write_all(b"\n")?;
        f.sync_all()?;
    }
    fs::rename(&tmp, history_path(dir))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timefmt::days_from_civil as day_of;

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("saans-store-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn line_puts_time_first() {
        let mut m = Map::new();
        m.insert("type".into(), "link_down".into());
        m.insert("evidence".into(), "said \"hi\"".into());
        let l = event_line("2026-10-13T00:00:00.000Z", 5, &m);
        assert!(l.starts_with("{\"time\":\"2026-10-13T00:00:00.000Z\",\"unix_ms\":5,"));
        let v: Value = serde_json::from_str(&l).unwrap();
        assert_eq!(v["evidence"], "said \"hi\"");
    }

    #[test]
    fn logs_rotate_by_local_day_and_are_pruned() {
        let dir = tmpdir("logs");
        let mut log = EventLog::new(&dir, 330).unwrap();
        let day_ms = 86_400_000;
        let oct13 = day_of(2026, 10, 13) * day_ms;
        // 20:00 UTC on Oct 12 is 01:30 IST on Oct 13.
        log.append(oct13 - 4 * 3_600_000, "{\"n\":1}");
        log.append(oct13 - 40 * day_ms, "{\"n\":0}");
        log.append(oct13 + 3_600_000, "{\"n\":2}");
        let names: Vec<String> = list_logs(&dir)
            .iter()
            .map(|(_, p)| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            vec!["events-2026-09-03.jsonl", "events-2026-10-13.jsonl"]
        );
        assert_eq!(load_recent_events(&dir, 10).len(), 3);
        assert_eq!(
            load_recent_events(&dir, 2),
            vec![serde_json::json!({"n":1}), serde_json::json!({"n":2})]
        );
        let removed = prune_logs(&dir, 30, 330, oct13 + 3_600_000);
        assert_eq!(removed.len(), 1);
        assert_eq!(list_logs(&dir).len(), 1);
        // A file is kept while any of its events may still be in retention.
        assert!(prune_logs(&dir, 1, 330, oct13 + day_ms + 3_600_000).is_empty());
        assert_eq!(
            prune_logs(&dir, 1, 330, oct13 + 2 * day_ms + 3_600_000).len(),
            1
        );
        let _ = fs::remove_dir_all(&dir);
    }

    fn summary(day: i64) -> DailySummary {
        DailySummary {
            day,
            resting_bpm_median: Some(14.0),
            resting_samples: 10,
            occupied_s: 3600.0,
            motion_bouts: 2,
            pauses: 0,
        }
    }

    #[test]
    fn history_is_pruned_by_age_not_by_count() {
        // The thing this exists to catch: `HISTORY_MAX_DAYS` bounds the
        // list at 90 *entries*. A gateway that runs one day a month keeps
        // ninety summaries spanning seven and a half years, and every one
        // of them is inside the count cap.
        let today = 10_000;
        let sparse: Vec<DailySummary> = (0..90).map(|i| summary(today - i * 30)).collect();
        assert!(
            sparse.len() <= saans_health::detectors::HISTORY_MAX_DAYS,
            "the count cap is not what is being tested here"
        );
        let mut h = BTreeMap::from([("lab".to_string(), sparse)]);
        let removed = prune_history(&mut h, 90, today);
        assert_eq!(removed, 87, "only the three within 90 days should survive");
        assert_eq!(h["lab"].len(), 3);
        assert!(h["lab"].iter().all(|s| today - s.day < 90));
    }

    #[test]
    fn history_retention_boundary_matches_the_event_log() {
        // `prune_logs` keeps a file until `today - (day + 1) >= days`; a
        // summary is one day, so the equivalent is `today - day >= days`.
        // A summary exactly `days` old goes; one day younger stays.
        let today = 500;
        let mut h = BTreeMap::from([(
            "lab".to_string(),
            vec![
                summary(today - 31),
                summary(today - 30),
                summary(today - 29),
            ],
        )]);
        assert_eq!(prune_history(&mut h, 30, today), 2);
        assert_eq!(h["lab"].len(), 1);
        assert_eq!(h["lab"][0].day, today - 29);
    }

    #[test]
    fn a_space_with_nothing_left_is_dropped_entirely() {
        // Otherwise `history.json` keeps the id of every space that ever
        // existed, which is a list of the rooms in the building.
        let today = 500;
        let mut h = BTreeMap::from([
            ("old".to_string(), vec![summary(today - 400)]),
            ("live".to_string(), vec![summary(today - 1)]),
        ]);
        assert_eq!(prune_history(&mut h, 90, today), 1);
        assert!(!h.contains_key("old"), "an emptied space must not linger");
        assert_eq!(h.len(), 1);
    }

    #[test]
    fn pruning_an_empty_or_current_history_removes_nothing() {
        let mut empty: BTreeMap<String, Vec<DailySummary>> = BTreeMap::new();
        assert_eq!(prune_history(&mut empty, 90, 500), 0);
        let mut fresh = BTreeMap::from([("lab".to_string(), vec![summary(500), summary(499)])]);
        assert_eq!(prune_history(&mut fresh, 90, 500), 0);
        assert_eq!(fresh["lab"].len(), 2);
    }

    #[test]
    fn audit_logs_are_swept_on_their_own_clock_and_never_by_the_event_sweep() {
        let dir = tmpdir("audit-ret");
        let day_ms = 86_400_000;
        let today = day_of(2026, 10, 24);
        let touch = |prefix: &str, days_ago: i64| {
            let name = format!("{prefix}{}.jsonl", date_string(today - days_ago));
            fs::write(dir.join(name), "{}\n").unwrap();
        };
        for d in [400, 366, 364, 40, 1] {
            touch(AUDIT_LOG_PREFIX, d);
            touch("events-", d);
        }
        touch("history.json.", 0); // not a dated log of either kind
        let now = today * day_ms + 3_600_000;

        // The event sweep does not touch audit files...
        let gone = prune_logs(&dir, 30, 0, now);
        assert_eq!(gone.len(), 4, "events 400, 366, 364 and 40 days old");
        assert_eq!(list_logs_prefixed(&dir, AUDIT_LOG_PREFIX).len(), 5);
        // ...and the audit sweep does not touch event files.
        let gone = prune_audit_logs(&dir, 365, 0, now);
        let names: Vec<String> = gone
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names.len(), 2, "audit 400 and 366 days old: {names:?}");
        assert!(names.iter().all(|n| n.starts_with(AUDIT_LOG_PREFIX)));
        assert_eq!(
            list_logs(&dir).len(),
            1,
            "the one-day-old event log is untouched"
        );
        assert_eq!(list_logs_prefixed(&dir, AUDIT_LOG_PREFIX).len(), 3);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn unreadable_history_is_moved_aside() {
        let dir = tmpdir("hist");
        assert!(load_history(&dir).0.is_empty());
        save_history(&dir, &BTreeMap::new(), "now").unwrap();
        let (h, err) = load_history(&dir);
        assert!(h.is_empty() && err.is_none());
        fs::write(history_path(&dir), "{not json").unwrap();
        let (_, err) = load_history(&dir);
        assert!(err.unwrap().contains("does not parse"));
        assert!(!history_path(&dir).exists());
        let _ = fs::remove_dir_all(&dir);
    }
}
