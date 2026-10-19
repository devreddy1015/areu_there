//! Gateway configuration (`saans.toml`). See `saans.example.toml`.
//!
//! Gateway settings are top-level keys, raw-mode settings live in `[raw]`,
//! and each monitored space is a `[[space]]` table that deserializes into
//! `saans_health::SpaceConfig`. Unknown keys are errors rather than being
//! silently ignored: a misspelt `vitals` must not quietly change what a
//! space collects.

use crate::roles::Role;
use saans_health::{LinkId, Severity, SpaceConfig};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GatewayConfig {
    /// UDP port the sensors send FeatureFrames (and debug raw CSI) to.
    pub udp_port: u16,
    /// Address the UDP socket binds to. Sensors are remote, so the default
    /// is every interface; set the sensor-VLAN address to narrow it.
    pub udp_bind: String,
    /// HTTP API and dashboard. Localhost by default: exposing the API is
    /// the operator's explicit choice.
    pub http_addr: String,
    /// Event logs (`events-YYYY-MM-DD.jsonl`) and `history.json`.
    pub data_dir: PathBuf,
    /// Delete an event log once every event in it is older than this.
    pub event_retention_days: u32,
    /// Drop a day's summary from `history.json` once it is older than
    /// this. Aggregates are normally kept longer than the raw events they
    /// came from -- `SAANS_PLAN.md` §8 says "events 30 days raw, then
    /// 15-minute aggregates" -- but that is a preference and not enforced
    /// here: refusing to start over it would take a monitoring system
    /// down for a setting whose error points at keeping *less* data.
    ///
    /// `HISTORY_MAX_DAYS` in `saans-health` already caps the list at 90
    /// entries, but that is a memory bound and counts entries, not days:
    /// a gateway that runs one day a month keeps ninety summaries
    /// spanning seven years. Retention is about age, so it lives here.
    pub history_retention_days: u32,
    /// Offset of local time from UTC, in minutes (IST = 330). Sets the day
    /// boundary of the event logs and the offset printed in timestamps.
    pub utc_offset_minutes: i32,
    pub raw: RawConfig,
    /// Bounded in-memory time series behind `/api/history`. Never
    /// persisted: the window is the retention policy.
    pub history: crate::history::HistoryConfig,
    #[serde(rename = "space")]
    pub spaces: Vec<SpaceConfig>,
    // ---- output interfaces (SAANS_PLAN.md §10) ----
    pub auth: AuthConfig,
    #[serde(rename = "user")]
    pub users: Vec<UserConfig>,
    #[serde(rename = "token")]
    pub tokens: Vec<TokenConfig>,
    pub mqtt: Option<MqttConfig>,
    pub telegram: BTreeMap<String, TelegramChannel>,
    pub webhook: BTreeMap<String, WebhookChannel>,
    #[serde(rename = "alert_route")]
    pub alert_routes: Vec<AlertRoute>,
}

/// Authentication. Off is allowed only while the HTTP API listens on a
/// loopback address: the moment it is reachable from a network, every
/// request needs a session or a token.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AuthConfig {
    /// `None` = automatic: required unless `http_addr` is loopback.
    pub required: Option<bool>,
    /// Browser session lifetime.
    pub session_hours: f64,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            required: None,
            session_hours: 12.0,
        }
    }
}

/// A person who logs in to the dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UserConfig {
    pub name: String,
    pub role: Role,
    /// Argon2id PHC string (`saans-gateway --hash-password` prints one).
    /// Never a plain password.
    pub password_hash: String,
    /// Space ids the user is limited to; empty = all spaces.
    #[serde(default)]
    pub spaces: Vec<String>,
}

/// A machine client (building management system, Grafana, scripts):
/// `Authorization: Bearer <token>`. Only the token's SHA-256 is stored.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenConfig {
    pub name: String,
    pub role: Role,
    /// Lowercase hex SHA-256 of the token.
    pub sha256: String,
    #[serde(default)]
    pub spaces: Vec<String>,
}

/// Publish to an MQTT broker (facilities systems, Home Assistant).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MqttConfig {
    pub host: String,
    pub port: u16,
    pub client_id: String,
    pub username: Option<String>,
    /// Name of the environment variable holding the password (secrets
    /// never live in the config file).
    pub password_env: Option<String>,
    /// Topic prefix: `{base}/{space}/occupancy`, `{base}/{space}/event`, ...
    pub base_topic: String,
    /// Publish Home Assistant MQTT discovery for one occupancy sensor per space.
    pub homeassistant_discovery: bool,
    /// Also publish `{base}/{space}/vitals` for spaces with `vitals = true`.
    /// A separate topic so broker ACLs can keep it from occupancy clients.
    pub vitals_topics: bool,
}

impl Default for MqttConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 1883,
            client_id: "saans-gateway".into(),
            username: None,
            password_env: None,
            base_topic: "saans/v1".into(),
            homeassistant_discovery: false,
            vitals_topics: false,
        }
    }
}

/// A Telegram chat that alerts go to. Create a bot with @BotFather; the
/// token is read from an environment variable, never from this file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TelegramChannel {
    pub bot_token_env: String,
    pub chat_id: i64,
    /// Overridable for tests; default https://api.telegram.org.
    #[serde(default = "default_telegram_api")]
    pub api_base: String,
    /// Telegram user ids allowed to press Acknowledge / Resolve; empty =
    /// any member of the chat.
    #[serde(default)]
    pub allowed_user_ids: Vec<i64>,
}

fn default_telegram_api() -> String {
    "https://api.telegram.org".into()
}

/// An HTTPS endpoint that receives alert JSON (paging systems, scripts).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebhookChannel {
    pub url: String,
    /// Environment variable with an HMAC-SHA256 key; the body's signature
    /// goes in `X-Saans-Signature`.
    #[serde(default)]
    pub secret_env: Option<String>,
}

/// Which events become alerts, where they go, and who gets them next if
/// nobody acknowledges.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlertRoute {
    pub name: String,
    /// Space ids; empty = all spaces.
    #[serde(default)]
    pub spaces: Vec<String>,
    /// Event types (e.g. "inactivity", "breathing_pause", "space_blind");
    /// empty = any type at or above `min_severity`.
    #[serde(default)]
    pub kinds: Vec<String>,
    #[serde(default = "default_min_severity")]
    pub min_severity: Severity,
    /// "telegram:<name>" or "webhook:<name>".
    pub channels: Vec<String>,
    /// Escalate if not acknowledged within this; 0 = never.
    #[serde(default)]
    pub ack_timeout_min: f64,
    #[serde(default)]
    pub escalate_to: Vec<String>,
    /// Messages leave the campus through third parties (Telegram, a
    /// webhook host). By default they carry no vital-sign numbers or
    /// evidence text, only "an alert in <space>, see the dashboard".
    #[serde(default)]
    pub include_health_details: bool,
}

fn default_min_severity() -> Severity {
    Severity::Check
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            udp_port: 5566,
            udp_bind: "0.0.0.0".into(),
            http_addr: "127.0.0.1:8080".into(),
            data_dir: PathBuf::from("saans-data"),
            event_retention_days: 30,
            history_retention_days: 90,
            utc_offset_minutes: 0,
            raw: RawConfig::default(),
            spaces: Vec::new(),
            auth: AuthConfig::default(),
            history: crate::history::HistoryConfig::default(),
            users: Vec::new(),
            tokens: Vec::new(),
            mqtt: None,
            telegram: BTreeMap::new(),
            webhook: BTreeMap::new(),
            alert_routes: Vec::new(),
        }
    }
}

/// Settings for raw CSI links, whose features the gateway computes
/// itself with `saans_dsp::FeatureExtractor` (debug / data collection).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RawConfig {
    /// The CSI rate the nodes are driven at. Not carried on the wire.
    pub sample_rate_hz: f64,
    /// Subcarriers per frame; 0 = take the count from a node's first datagram.
    pub subcarriers: usize,
    /// Empty-room calibration period of the presence detector.
    pub calibration_s: f64,
}

impl Default for RawConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 20.0,
            subcarriers: 0,
            calibration_s: 30.0,
        }
    }
}

#[derive(Debug)]
pub struct ConfigError(pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

fn err<T>(msg: impl Into<String>) -> Result<T, ConfigError> {
    Err(ConfigError(msg.into()))
}

impl GatewayConfig {
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| ConfigError(format!("{}: {e}", path.display())))?;
        Self::from_toml(&text).map_err(|e| ConfigError(format!("{}: {e}", path.display())))
    }

    pub fn from_toml(text: &str) -> Result<Self, ConfigError> {
        let table: toml::Table = toml::from_str(text).map_err(|e| ConfigError(e.to_string()))?;
        check_space_keys(&table)?;
        let cfg: GatewayConfig = toml::from_str(text).map_err(|e| ConfigError(e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.event_retention_days == 0 {
            return err("event_retention_days must be at least 1");
        }
        if self.history_retention_days == 0 {
            return err("history_retention_days must be at least 1");
        }
        if !(-1440..=1440).contains(&self.utc_offset_minutes) {
            return err("utc_offset_minutes must be within +-1440");
        }
        if !(self.raw.sample_rate_hz >= 10.0 && self.raw.sample_rate_hz <= 1000.0) {
            return err(format!(
                "raw.sample_rate_hz must be in [10, 1000] (got {}): the gross-motion band reaches 4 Hz",
                self.raw.sample_rate_hz
            ));
        }
        if !(self.raw.calibration_s > 0.0 && self.raw.calibration_s.is_finite()) {
            return err("raw.calibration_s must be positive");
        }
        if self.raw.subcarriers > 255 {
            return err("raw.subcarriers must be at most 255 (0 = from the first datagram)");
        }
        self.http_socket_addr()?;
        self.udp_socket_addr()?;
        let mut ids = BTreeSet::new();
        let mut owner: BTreeMap<LinkId, &str> = BTreeMap::new();
        for s in &self.spaces {
            if s.id.is_empty()
                || !s
                    .id
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
            {
                return err(format!(
                    "space id {:?} must be non-empty and use only A-Z a-z 0-9 - _ .",
                    s.id
                ));
            }
            if !ids.insert(s.id.as_str()) {
                return err(format!("space id {:?} appears twice", s.id));
            }
            for l in &s.links {
                if let Some(other) = owner.insert(*l, &s.id) {
                    return err(format!(
                        "link {l} is listed by both space {other:?} and space {:?}",
                        s.id
                    ));
                }
            }
            if s.inactivity_alert_min.is_nan() || s.inactivity_alert_min <= 0.0 {
                return err(format!(
                    "space {:?}: inactivity_alert_min must be positive",
                    s.id
                ));
            }
        }
        self.validate_outputs(&ids)
    }

    /// Auth is required when the API is reachable beyond this machine.
    pub fn auth_required(&self) -> bool {
        let loopback = self
            .http_socket_addr()
            .map(|a| a.ip().is_loopback())
            .unwrap_or(false);
        self.auth.required.unwrap_or(!loopback)
    }

    fn validate_outputs(&self, space_ids: &BTreeSet<&str>) -> Result<(), ConfigError> {
        let loopback = self.http_socket_addr()?.ip().is_loopback();
        if self.auth.required == Some(false) && !loopback {
            return err(format!(
                "auth.required = false is only allowed on a loopback http_addr (got {})",
                self.http_addr
            ));
        }
        if !(self.auth.session_hours > 0.0 && self.auth.session_hours <= 24.0 * 30.0) {
            return err("auth.session_hours must be in (0, 720]");
        }
        if self.auth_required() && self.users.is_empty() && self.tokens.is_empty() {
            return err("authentication is required (http_addr is not loopback) but no [[user]] or [[token]] is configured");
        }
        let mut names = BTreeSet::new();
        let check_spaces = |who: &str, spaces: &[String]| -> Result<(), ConfigError> {
            for sp in spaces {
                if !space_ids.contains(sp.as_str()) {
                    return err(format!("{who}: unknown space {sp:?}"));
                }
            }
            Ok(())
        };
        for u in &self.users {
            if u.name.is_empty() || !names.insert(u.name.as_str()) {
                return err(format!("user name {:?} is empty or used twice", u.name));
            }
            if !u.password_hash.starts_with("$argon2") {
                return err(format!("user {:?}: password_hash must be an Argon2 PHC string (saans-gateway --hash-password)", u.name));
            }
            check_spaces(&format!("user {:?}", u.name), &u.spaces)?;
        }
        for t in &self.tokens {
            if t.name.is_empty() || !names.insert(t.name.as_str()) {
                return err(format!("token name {:?} is empty or used twice", t.name));
            }
            if t.sha256.len() != 64
                || !t
                    .sha256
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
            {
                return err(format!(
                    "token {:?}: sha256 must be 64 lowercase hex digits",
                    t.name
                ));
            }
            check_spaces(&format!("token {:?}", t.name), &t.spaces)?;
        }
        if self.mqtt.is_some() {
            // `mqtt::spawn` is still a stub. Accepting the section would
            // mean --check passes and the process then panics on startup,
            // which is worse than refusing it here.
            return err("[mqtt] is configured but the MQTT publisher is not implemented yet; remove the section");
        }
        let h = &self.history;
        if !(h.window_s > 0.0 && h.window_s <= crate::history::MAX_WINDOW_S) {
            return err(format!(
                "history.window_s must be in (0, {}]",
                crate::history::MAX_WINDOW_S
            ));
        }
        if !(h.cadence_s >= crate::history::MIN_CADENCE_S && h.cadence_s <= h.window_s) {
            return err(format!(
                "history.cadence_s must be in [{}, history.window_s]",
                crate::history::MIN_CADENCE_S
            ));
        }
        for (name, t) in &self.telegram {
            if t.bot_token_env.is_empty() {
                return err(format!("telegram.{name}: bot_token_env is required (the token itself never goes in the config)"));
            }
        }
        for (name, w) in &self.webhook {
            if !(w.url.starts_with("https://")
                || w.url.starts_with("http://127.")
                || w.url.starts_with("http://localhost"))
            {
                return err(format!(
                    "webhook.{name}: url must be https:// (plain http only to localhost)"
                ));
            }
        }
        let channel_exists = |c: &str| match c.split_once(':') {
            Some(("telegram", n)) => self.telegram.contains_key(n),
            Some(("webhook", n)) => self.webhook.contains_key(n),
            _ => false,
        };
        let mut route_names = BTreeSet::new();
        for r in &self.alert_routes {
            if r.name.is_empty() || !route_names.insert(r.name.as_str()) {
                return err(format!(
                    "alert_route name {:?} is empty or used twice",
                    r.name
                ));
            }
            check_spaces(&format!("alert_route {:?}", r.name), &r.spaces)?;
            if r.channels.is_empty() {
                return err(format!(
                    "alert_route {:?}: needs at least one channel",
                    r.name
                ));
            }
            for c in r.channels.iter().chain(&r.escalate_to) {
                if !channel_exists(c) {
                    return err(format!("alert_route {:?}: unknown channel {c:?} (use telegram:<name> or webhook:<name>)", r.name));
                }
            }
            if !(r.ack_timeout_min >= 0.0 && r.ack_timeout_min.is_finite()) {
                return err(format!(
                    "alert_route {:?}: ack_timeout_min must be >= 0",
                    r.name
                ));
            }
            if r.ack_timeout_min > 0.0 && r.escalate_to.is_empty() {
                return err(format!(
                    "alert_route {:?}: ack_timeout_min set but escalate_to is empty",
                    r.name
                ));
            }
        }
        if !self.alert_routes.is_empty() {
            // Checked last, so a route that is also *wrong* still gets
            // its specific error first. `AlertService::new` is a stub:
            // without this, --check passes and `Gateway::start` panics.
            return err("[[alert_route]] is configured but the alert router is not implemented yet; remove the routes");
        }
        Ok(())
    }

    pub fn http_socket_addr(&self) -> Result<SocketAddr, ConfigError> {
        self.http_addr.parse().map_err(|_| {
            ConfigError(format!(
                "http_addr {:?} is not an IP:port address",
                self.http_addr
            ))
        })
    }

    pub fn udp_socket_addr(&self) -> Result<SocketAddr, ConfigError> {
        format!("{}:{}", self.udp_bind, self.udp_port)
            .parse()
            .or_else(|_| format!("[{}]:{}", self.udp_bind, self.udp_port).parse())
            .map_err(|_| ConfigError(format!("udp_bind {:?} is not an IP address", self.udp_bind)))
    }

    /// A space's display name, falling back to its id.
    pub fn space_name(s: &SpaceConfig) -> &str {
        if s.name.is_empty() {
            &s.id
        } else {
            &s.name
        }
    }
}

/// `SpaceConfig` ignores unknown keys; check them here so a typo is an
/// error, not a silently different privacy or alerting setting.
fn check_space_keys(table: &toml::Table) -> Result<(), ConfigError> {
    let Some(spaces) = table.get("space") else {
        return Ok(());
    };
    let Some(spaces) = spaces.as_array() else {
        return err("`space` must be an array of tables: write [[space]]");
    };
    let known: BTreeSet<String> = match serde_json::to_value(SpaceConfig::default()) {
        Ok(serde_json::Value::Object(m)) => m.keys().cloned().collect(),
        _ => return Ok(()),
    };
    for (i, s) in spaces.iter().enumerate() {
        let Some(s) = s.as_table() else {
            return err("each [[space]] must be a table");
        };
        let id = s
            .get("id")
            .and_then(|v| v.as_str())
            .map_or_else(|| format!("#{}", i + 1), |s| format!("{s:?}"));
        if !s.contains_key("id") {
            return err(format!("[[space]] {id} has no id"));
        }
        for k in s.keys() {
            if !known.contains(k) {
                let list: Vec<&str> = known.iter().map(String::as_str).collect();
                return err(format!(
                    "[[space]] {id}: unknown key `{k}` (known keys: {})",
                    list.join(", ")
                ));
            }
        }
        if let Some(links) = s.get("links").and_then(|v| v.as_array()) {
            for l in links {
                if let Some(t) = l.as_table() {
                    if let Some(k) = t.keys().find(|k| *k != "rx" && *k != "tx") {
                        return err(format!("[[space]] {id}: unknown key `{k}` in a link (use {{ rx = N, tx = N }})"));
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_config_parses() {
        let cfg = GatewayConfig::from_toml(include_str!("../saans.example.toml")).unwrap();
        assert_eq!(cfg.udp_port, 5566);
        assert_eq!(cfg.http_addr, "127.0.0.1:8080");
        assert!(!cfg.spaces.is_empty());
    }

    #[test]
    fn defaults_are_conservative() {
        let cfg = GatewayConfig::from_toml("[[space]]\nid = \"a\"\nlinks = [{ rx = 1, tx = 2 }]\n")
            .unwrap();
        assert_eq!(cfg.http_addr, "127.0.0.1:8080");
        assert_eq!(cfg.event_retention_days, 30);
        assert!(!cfg.spaces[0].vitals);
        assert!(!cfg.spaces[0].collapse_detection);
        assert_eq!(cfg.spaces[0].links, vec![LinkId { rx: 1, tx: 2 }]);
    }

    #[test]
    fn typos_are_errors() {
        let e = GatewayConfig::from_toml("[[space]]\nid = \"a\"\nvital = true\n").unwrap_err();
        assert!(e.0.contains("unknown key `vital`"), "{e}");
        assert!(GatewayConfig::from_toml("udp_prot = 1\n").is_err());
        assert!(GatewayConfig::from_toml("[raw]\nrate = 1\n").is_err());
        assert!(
            GatewayConfig::from_toml("[[space]]\nid = \"a\"\nlinks = [{ rx = 1, txx = 2 }]\n")
                .is_err()
        );
    }

    #[test]
    fn a_link_belongs_to_one_space() {
        let t = "[[space]]\nid = \"a\"\nlinks = [{ rx = 1, tx = 2 }]\n[[space]]\nid = \"b\"\nlinks = [{ rx = 1, tx = 2 }]\n";
        assert!(GatewayConfig::from_toml(t)
            .unwrap_err()
            .0
            .contains("listed by both"));
        assert!(GatewayConfig::from_toml("[[space]]\nid = \"a b\"\n").is_err());
        assert!(
            GatewayConfig::from_toml("[[space]]\nid = \"a\"\n[[space]]\nid = \"a\"\n").is_err()
        );
    }
}
