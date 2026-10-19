//! Audit log: who read health data or acted on an alert, when
//! (SAANS_PLAN.md §10.1).
//!
//! Separate from the event log on purpose. `events-*.jsonl` records what
//! the *building* did and is deleted on the DPDP retention clock;
//! `audit-*.jsonl` records what *people* did with the system, which is
//! the record you need when someone asks who looked at a space's
//! breathing data. Different question, different file, different
//! lifetime.
//!
//! Failures are counted and surfaced in `/api/health`, never fatal: a
//! full disk must not stop a safety system from sensing. That is a
//! deliberate trade and it is visible rather than silent.

use crate::roles::Principal;
use crate::timefmt::{date_string, iso8601, local_day};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// One `vitals_read` per principal per space per this long. Without a
/// throttle a dashboard polling at 2 Hz would write 7,200 identical
/// lines an hour and bury the entries that matter.
const VITALS_READ_EVERY_MS: i64 = 60 * 60 * 1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Login,
    LoginFailed,
    Logout,
    /// A request with no valid credential.
    Denied,
    /// A principal read a space's consented vital signs.
    VitalsRead,
}

impl Action {
    fn as_str(self) -> &'static str {
        match self {
            Action::Login => "login",
            Action::LoginFailed => "login_failed",
            Action::Logout => "logout",
            Action::Denied => "denied",
            Action::VitalsRead => "vitals_read",
        }
    }
}

#[derive(Debug, Default)]
struct State {
    day: i64,
    file: Option<File>,
    last_vitals_read: HashMap<(String, String), i64>,
}

pub struct AuditLog {
    dir: Option<PathBuf>,
    offset_min: i32,
    state: Mutex<State>,
    written: Mutex<(u64, u64, Option<String>)>,
}

impl AuditLog {
    /// `dir` of `None` disables the log (replay, tests): the gateway
    /// still runs, it just has nothing to write to.
    pub fn new(dir: Option<&Path>, offset_min: i32) -> Self {
        let dir = dir.and_then(|d| fs::create_dir_all(d).ok().map(|_| d.to_path_buf()));
        Self {
            dir,
            offset_min,
            state: Mutex::new(State {
                day: i64::MIN,
                ..Default::default()
            }),
            written: Mutex::new((0, 0, None)),
        }
    }

    pub fn enabled(&self) -> bool {
        self.dir.is_some()
    }

    /// `{written, write_errors, last_error}` for `/api/health`.
    pub fn status(&self) -> Value {
        let (written, errors, last) = {
            let w = self.written.lock().unwrap_or_else(|p| p.into_inner());
            (w.0, w.1, w.2.clone())
        };
        json!({ "enabled": self.enabled(), "written": written, "write_errors": errors, "last_error": last })
    }

    pub fn record(
        &self,
        unix_ms: i64,
        action: Action,
        who: &str,
        role: Option<&str>,
        detail: Value,
    ) {
        let Some(dir) = &self.dir else { return };
        let line = json!({
            "time": iso8601(unix_ms, self.offset_min),
            "unix_ms": unix_ms,
            "action": action.as_str(),
            "who": who,
            "role": role,
            "detail": detail,
        });
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let day = local_day(unix_ms, self.offset_min);
        if day != st.day || st.file.is_none() {
            st.day = day;
            st.file = match OpenOptions::new()
                .create(true)
                .append(true)
                .open(dir.join(format!("audit-{}.jsonl", date_string(day))))
            {
                Ok(f) => Some(f),
                Err(e) => return self.fail(e.to_string()),
            };
        }
        let mut buf = line.to_string().into_bytes();
        buf.push(b'\n');
        // One write per line under O_APPEND, so a crash cannot interleave.
        match st.file.as_mut().map(|f| f.write_all(&buf)) {
            Some(Ok(())) => {
                let mut w = self.written.lock().unwrap_or_else(|p| p.into_inner());
                w.0 += 1;
            }
            Some(Err(e)) => {
                st.file = None;
                self.fail(e.to_string());
            }
            None => {}
        }
    }

    /// Record a read of a space's vital signs, at most once an hour per
    /// (principal, space). Returns whether a line was written.
    pub fn vitals_read(&self, unix_ms: i64, p: &Principal, space_id: &str) -> bool {
        if !self.enabled() {
            return false;
        }
        {
            let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
            let key = (p.name.clone(), space_id.to_string());
            match st.last_vitals_read.get(&key) {
                // Time can also move backwards (a replay); treat that as due.
                Some(&prev) if unix_ms >= prev && unix_ms - prev < VITALS_READ_EVERY_MS => {
                    return false
                }
                _ => st.last_vitals_read.insert(key, unix_ms),
            };
        }
        self.record(
            unix_ms,
            Action::VitalsRead,
            &p.name,
            Some(p.role.as_str()),
            json!({ "space_id": space_id }),
        );
        true
    }

    fn fail(&self, e: String) {
        let mut w = self.written.lock().unwrap_or_else(|p| p.into_inner());
        w.1 += 1;
        w.2 = Some(e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roles::Role;

    fn principal(name: &str) -> Principal {
        Principal {
            name: name.into(),
            role: Role::Safety,
            spaces: None,
        }
    }

    fn lines(dir: &Path) -> Vec<Value> {
        let mut out = Vec::new();
        for e in fs::read_dir(dir).expect("read dir") {
            let p = e.expect("entry").path();
            if p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("audit-"))
            {
                for l in fs::read_to_string(&p).expect("read").lines() {
                    out.push(serde_json::from_str(l).expect("json"));
                }
            }
        }
        out
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("saans-audit-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn actions_are_appended_as_json_lines() {
        let dir = tmp("actions");
        let log = AuditLog::new(Some(&dir), 330);
        log.record(
            1_700_000_000_000,
            Action::Login,
            "ops",
            Some("ops"),
            json!({"from": "127.0.0.1"}),
        );
        log.record(
            1_700_000_001_000,
            Action::Denied,
            "-",
            None,
            json!({"path": "/api/spaces"}),
        );
        let l = lines(&dir);
        assert_eq!(l.len(), 2);
        assert_eq!(l[0]["action"], "login");
        assert_eq!(l[0]["who"], "ops");
        assert_eq!(l[0]["role"], "ops");
        assert_eq!(l[0]["detail"]["from"], "127.0.0.1");
        assert_eq!(l[1]["action"], "denied");
        assert_eq!(l[1]["role"], Value::Null);
        assert!(
            l[0]["time"].as_str().is_some_and(|t| t.ends_with("+05:30")),
            "local offset is recorded"
        );
        assert_eq!(log.status()["written"], 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn vitals_reads_are_throttled_per_principal_and_space() {
        let dir = tmp("vitals");
        let log = AuditLog::new(Some(&dir), 0);
        let t = 1_700_000_000_000;
        assert!(log.vitals_read(t, &principal("ana"), "lab"));
        assert!(
            !log.vitals_read(t + 1000, &principal("ana"), "lab"),
            "same hour, same space"
        );
        assert!(
            log.vitals_read(t + 1000, &principal("ana"), "rest"),
            "a different space is a new read"
        );
        assert!(
            log.vitals_read(t + 1000, &principal("bo"), "lab"),
            "a different person is a new read"
        );
        assert!(
            log.vitals_read(t + VITALS_READ_EVERY_MS, &principal("ana"), "lab"),
            "an hour later"
        );
        assert_eq!(lines(&dir).len(), 4);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_disabled_log_is_harmless() {
        let log = AuditLog::new(None, 0);
        assert!(!log.enabled());
        log.record(0, Action::Login, "x", None, json!({}));
        assert!(!log.vitals_read(0, &principal("ana"), "lab"));
        assert_eq!(log.status()["written"], 0);
        assert_eq!(log.status()["enabled"], false);
    }
}
