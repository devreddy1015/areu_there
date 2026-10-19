//! Test harness: an in-process gateway on ephemeral ports driven by its
//! manual clock, plus just enough HTTP/1.1, WebSocket (RFC 6455) and
//! Prometheus text parsing to talk to it over real sockets without a
//! client library.

#![allow(dead_code)]

use saans_dsp::features::{
    FeatureFrame, FLAG_BREATHING_VALID, FLAG_CALIBRATED, FLAG_OCCUPIED, FLAG_VITALS_ENABLED,
};
use saans_gateway::{ClockMode, Gateway, GatewayConfig};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpStream, UdpSocket};
use tokio::time::timeout;

/// Gateway time at start: an arbitrary instant (local-epoch seconds).
pub const T0: f64 = 1_790_000_000.0;
const IO_TIMEOUT: Duration = Duration::from_secs(10);

/// A fresh, empty directory under Cargo's per-target temp dir.
pub fn temp_dir(name: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "saans-gw-{name}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// `spaces_toml` (spaces and `[raw]`) plus loopback, ephemeral ports and
/// `data_dir`, as TOML text.
pub fn config_toml(data_dir: &std::path::Path, spaces_toml: &str) -> String {
    format!(
        "udp_port = 0\nudp_bind = \"127.0.0.1\"\nhttp_addr = \"127.0.0.1:0\"\ndata_dir = '{}'\n{spaces_toml}",
        data_dir.display()
    )
}

/// [`config_toml`] with a fresh temp data dir, parsed.
pub fn config(name: &str, spaces_toml: &str) -> GatewayConfig {
    let text = config_toml(&temp_dir(name), spaces_toml);
    GatewayConfig::from_toml(&text).unwrap_or_else(|e| panic!("test config: {e}\n{text}"))
}

pub struct TestGw {
    pub gw: Gateway,
    pub cfg: GatewayConfig,
    pub t: f64,
    tx: UdpSocket,
}

impl TestGw {
    pub async fn start(cfg: GatewayConfig) -> TestGw {
        let gw = Gateway::start(cfg.clone(), ClockMode::Manual { t0: T0 })
            .await
            .expect("gateway starts");
        let tx = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        TestGw { gw, cfg, t: T0, tx }
    }

    pub fn http(&self) -> SocketAddr {
        self.gw.http_addr
    }

    /// One datagram to the gateway's UDP port (loopback: queued on return).
    pub async fn send(&self, data: &[u8]) {
        self.tx.send_to(data, self.gw.udp_addr).await.unwrap();
    }

    pub async fn send_frames(&self, frames: &[FeatureFrame]) {
        for f in frames {
            self.send(&f.encode()).await;
        }
    }

    /// Advance the gateway clock by `dt`; datagrams already sent are
    /// ingested at the current time first.
    pub async fn advance(&mut self, dt: f64) -> u64 {
        self.t += dt;
        self.gw.advance_to(self.t).await
    }

    /// `seconds` of 2 Hz operation: each step sends `frames(step)` and
    /// then advances the clock 0.5 s.
    pub async fn run(&mut self, seconds: f64, frames: impl Fn(u32) -> Vec<FeatureFrame>) {
        let steps = (seconds / 0.5).round() as u32;
        for i in 0..steps {
            self.send_frames(&frames(i)).await;
            self.advance(0.5).await;
        }
    }

    pub async fn get(&self, path: &str) -> Resp {
        get(self.http(), path).await
    }

    /// For the fonts, which are not text.
    pub async fn get_bytes(&self, path: &str) -> (u16, Vec<(String, String)>, Vec<u8>) {
        get_bytes(self.http(), path).await
    }

    pub async fn json(&self, path: &str) -> Value {
        let r = self.get(path).await;
        assert_eq!(r.status, 200, "GET {path}: {}", r.body);
        r.json()
    }

    /// Events from `/api/events` (optionally for one space) of one type.
    pub async fn events_of(&self, space: Option<&str>, ty: &str) -> Vec<Value> {
        let path = match space {
            Some(s) => format!("/api/events?space={s}&limit=1000"),
            None => "/api/events?limit=1000".to_string(),
        };
        let all = self.json(&path).await;
        all.as_array()
            .unwrap()
            .iter()
            .filter(|e| e["type"] == ty)
            .cloned()
            .collect()
    }

    pub async fn stop(self) {
        timeout(Duration::from_secs(15), self.gw.shutdown())
            .await
            .expect("gateway shuts down");
    }
}

// ----------------------------------------------------------------- frames

/// A calibrated link seeing an empty room, 20 Hz CSI, no loss.
pub fn empty_frame(rx: u8, tx: u8, seq: u32) -> FeatureFrame {
    FeatureFrame {
        node_id: rx,
        tx_id: tx,
        flags: FLAG_CALIBRATED,
        seq,
        uptime_ms: seq.wrapping_mul(500),
        csi_rate_chz: 2000,
        loss_permille: 0,
        motion_energy: 0.02,
        presence_threshold: 0.05,
        gross_motion: 0.01,
        best_bnr: 3.0,
        calib_age_s: 3600,
        rssi_dbm: -55,
        noise_floor_dbm: -92,
        n_subcarriers: 56,
        ..Default::default()
    }
}

/// Occupied by a still person; with `vitals = Some((bpm, conf))` the
/// node also reports valid breathing (as a consented node would).
pub fn occupied_frame(rx: u8, tx: u8, seq: u32, vitals: Option<(f32, f32)>) -> FeatureFrame {
    let mut f = empty_frame(rx, tx, seq);
    f.flags |= FLAG_OCCUPIED;
    f.motion_energy = 0.2;
    f.gross_motion = 0.03;
    if let Some((bpm, conf)) = vitals {
        f.flags |= FLAG_VITALS_ENABLED | FLAG_BREATHING_VALID;
        f.breath_rms = 0.1;
        f.breathing_bpm = bpm;
        f.breathing_conf = conf;
    }
    f
}

/// One raw-CSI datagram, single antenna, as `esp32_udp.py` builds it.
pub fn raw_datagram(node: u8, seq: u32, t: f64, n_sub: usize) -> Vec<u8> {
    let iq: Vec<(i8, i8)> = (0..n_sub)
        .map(|k| {
            (
                ((t * 0.3 + k as f64).sin() * 20.0) as i8,
                40 + ((t * 1.3).sin() * 5.0) as i8,
            )
        })
        .collect();
    saans_gateway::raw::build(&iq, node, seq, -50, -95)
}

// ------------------------------------------------------------------- HTTP

#[derive(Debug)]
pub struct Resp {
    pub status: u16,
    /// Lower-case names.
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl Resp {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    pub fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or_else(|e| panic!("not JSON ({e}): {}", self.body))
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn parse_head(head: &str) -> (u16, Vec<(String, String)>) {
    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap();
    let mut parts = status_line.splitn(3, ' ');
    assert_eq!(
        parts.next(),
        Some("HTTP/1.1"),
        "status line {status_line:?}"
    );
    let status = parts
        .next()
        .and_then(|s| s.parse().ok())
        .expect("status code");
    let headers = lines
        .map(|l| {
            let (k, v) = l
                .split_once(':')
                .unwrap_or_else(|| panic!("bad header line {l:?}"));
            (k.trim().to_ascii_lowercase(), v.trim().to_string())
        })
        .collect();
    (status, headers)
}

fn dechunk(mut b: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let eol = find(b, b"\r\n").expect("chunk size line");
        let size_hex = std::str::from_utf8(&b[..eol])
            .unwrap()
            .split(';')
            .next()
            .unwrap()
            .trim();
        let size = usize::from_str_radix(size_hex, 16).expect("chunk size");
        b = &b[eol + 2..];
        if size == 0 {
            return out;
        }
        out.extend_from_slice(&b[..size]);
        assert_eq!(&b[size..size + 2], b"\r\n", "chunk terminator");
        b = &b[size + 2..];
    }
}

/// `GET path` over a fresh connection, body left as bytes.
///
/// The site serves woff2, so not every response is text any more and the
/// UTF-8 assumption in [`get`] would panic on one.
pub async fn get_bytes(addr: SocketAddr, path: &str) -> (u16, Vec<(String, String)>, Vec<u8>) {
    let fut = async {
        let mut s = TcpStream::connect(addr).await.expect("connect");
        let req = format!(
            "GET {path} HTTP/1.1\r\nHost: {addr}\r\nAccept: */*\r\nConnection: close\r\n\r\n"
        );
        s.write_all(req.as_bytes()).await.unwrap();
        let mut raw = Vec::new();
        s.read_to_end(&mut raw).await.unwrap();
        let end = find(&raw, b"\r\n\r\n").expect("response has a header terminator");
        let (status, headers) = parse_head(std::str::from_utf8(&raw[..end]).unwrap());
        let rest = &raw[end + 4..];
        let chunked = headers
            .iter()
            .any(|(k, v)| k == "transfer-encoding" && v.contains("chunked"));
        let body = if chunked {
            dechunk(rest)
        } else if let Some((_, n)) = headers.iter().find(|(k, _)| k == "content-length") {
            let n: usize = n.parse().unwrap();
            assert_eq!(rest.len(), n, "body length matches Content-Length");
            rest.to_vec()
        } else {
            rest.to_vec()
        };
        (status, headers, body)
    };
    timeout(IO_TIMEOUT, fut)
        .await
        .unwrap_or_else(|_| panic!("GET {path} timed out"))
}

/// `GET path`, body decoded as text.
pub async fn get(addr: SocketAddr, path: &str) -> Resp {
    let (status, headers, body) = get_bytes(addr, path).await;
    Resp {
        status,
        headers,
        body: String::from_utf8(body).expect("UTF-8 body"),
    }
}

// -------------------------------------------------------------- WebSocket

pub struct Ws {
    s: TcpStream,
    buf: Vec<u8>,
}

/// RFC 6455 section 1.3's example key and its accept value.
const WS_KEY: &str = "dGhlIHNhbXBsZSBub25jZQ==";
const WS_ACCEPT: &str = "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=";

impl Ws {
    pub async fn connect(addr: SocketAddr, path: &str) -> Ws {
        let fut = async {
            let mut s = TcpStream::connect(addr).await.expect("connect");
            let req = format!(
                "GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: Upgrade\r\nUpgrade: websocket\r\n\
                 Sec-WebSocket-Version: 13\r\nSec-WebSocket-Key: {WS_KEY}\r\n\r\n"
            );
            s.write_all(req.as_bytes()).await.unwrap();
            let mut buf = Vec::new();
            let end = loop {
                if let Some(e) = find(&buf, b"\r\n\r\n") {
                    break e;
                }
                let mut chunk = [0u8; 1024];
                let n = s.read(&mut chunk).await.unwrap();
                assert!(n > 0, "connection closed during the handshake");
                buf.extend_from_slice(&chunk[..n]);
            };
            let (status, headers) = parse_head(std::str::from_utf8(&buf[..end]).unwrap());
            assert_eq!(status, 101, "{headers:?}");
            let h = |n: &str| headers.iter().find(|(k, _)| k == n).map(|(_, v)| v.clone());
            assert_eq!(
                h("upgrade").map(|v| v.to_ascii_lowercase()).as_deref(),
                Some("websocket")
            );
            assert_eq!(h("sec-websocket-accept").as_deref(), Some(WS_ACCEPT));
            buf.drain(..end + 4);
            Ws { s, buf }
        };
        timeout(IO_TIMEOUT, fut)
            .await
            .expect("WebSocket handshake timed out")
    }

    async fn fill(&mut self, n: usize) {
        while self.buf.len() < n {
            let mut chunk = [0u8; 8192];
            let k = self.s.read(&mut chunk).await.unwrap();
            assert!(k > 0, "WebSocket closed by the server");
            self.buf.extend_from_slice(&chunk[..k]);
        }
    }

    /// The next complete frame: (opcode, payload).
    async fn frame(&mut self) -> (u8, Vec<u8>) {
        self.fill(2).await;
        let (b0, b1) = (self.buf[0], self.buf[1]);
        assert!(b0 & 0x80 != 0, "fragmented messages are not expected");
        assert_eq!(b0 & 0x70, 0, "no extensions were negotiated");
        assert_eq!(b1 & 0x80, 0, "server frames must not be masked");
        let (len, mut off) = match b1 & 0x7F {
            126 => {
                self.fill(4).await;
                (u16::from_be_bytes([self.buf[2], self.buf[3]]) as usize, 4)
            }
            127 => {
                self.fill(10).await;
                (
                    u64::from_be_bytes(self.buf[2..10].try_into().unwrap()) as usize,
                    10,
                )
            }
            n => (n as usize, 2),
        };
        self.fill(off + len).await;
        let payload = self.buf[off..off + len].to_vec();
        off += len;
        self.buf.drain(..off);
        (b0 & 0x0F, payload)
    }

    /// The next text message, skipping control frames; panics on close.
    pub async fn next_text(&mut self) -> String {
        let fut = async {
            loop {
                match self.frame().await {
                    (0x1, p) => return String::from_utf8(p).expect("UTF-8 text frame"),
                    (0x8, p) => panic!("server closed the WebSocket: {p:?}"),
                    (0x9 | 0xA, _) => continue,
                    (op, _) => panic!("unexpected opcode {op}"),
                }
            }
        };
        timeout(IO_TIMEOUT, fut)
            .await
            .expect("no WebSocket message within 10 s")
    }

    pub async fn next_json(&mut self) -> Value {
        let t = self.next_text().await;
        serde_json::from_str(&t).unwrap_or_else(|e| panic!("not JSON ({e}): {t}"))
    }

    /// Messages until one satisfies `pred` (at most `max`).
    pub async fn until(&mut self, max: usize, pred: impl Fn(&Value) -> bool) -> Value {
        for _ in 0..max {
            let v = self.next_json().await;
            if pred(&v) {
                return v;
            }
        }
        panic!("no matching WebSocket message in {max}");
    }

    /// Send a (masked) close frame; the server must answer with its own
    /// close frame (RFC 6455 section 5.5.1).
    pub async fn close(mut self) {
        let mask = [0x12u8, 0x34, 0x56, 0x78];
        let payload = 1000u16.to_be_bytes();
        let mut f = vec![0x88, 0x80 | payload.len() as u8];
        f.extend_from_slice(&mask);
        f.extend(payload.iter().enumerate().map(|(i, b)| b ^ mask[i % 4]));
        self.s.write_all(&f).await.unwrap();
        // Messages still in flight may come first.
        let fut = async { while self.frame().await.0 != 0x8 {} };
        timeout(IO_TIMEOUT, fut)
            .await
            .expect("server answers a close frame");
    }
}

// ------------------------------------------------------------- Prometheus

#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub name: String,
    pub labels: BTreeMap<String, String>,
    pub value: f64,
}

fn valid_metric_name(s: &str) -> bool {
    let mut c = s.chars();
    c.next()
        .is_some_and(|f| f.is_ascii_alphabetic() || f == '_' || f == ':')
        && c.all(|x| x.is_ascii_alphanumeric() || x == '_' || x == ':')
}

fn valid_label_name(s: &str) -> bool {
    let mut c = s.chars();
    !s.starts_with("__")
        && c.next()
            .is_some_and(|f| f.is_ascii_alphabetic() || f == '_')
        && c.all(|x| x.is_ascii_alphanumeric() || x == '_')
}

fn parse_value(s: &str) -> Option<f64> {
    match s {
        "NaN" => Some(f64::NAN),
        "+Inf" | "Inf" => Some(f64::INFINITY),
        "-Inf" => Some(f64::NEG_INFINITY),
        _ if s
            .chars()
            .all(|c| c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E')) =>
        {
            s.parse().ok()
        }
        _ => None,
    }
}

/// Parse `{a="x",b="y"}` starting at `s[0] == '{'`; returns labels and
/// the rest after `}`.
fn parse_labels(s: &str) -> Result<(BTreeMap<String, String>, &str), String> {
    let mut labels = BTreeMap::new();
    let mut rest = &s[1..];
    loop {
        rest = rest.trim_start();
        if let Some(r) = rest.strip_prefix('}') {
            return Ok((labels, r));
        }
        let eq = rest.find('=').ok_or("label without '='")?;
        let name = rest[..eq].trim();
        if !valid_label_name(name) {
            return Err(format!("invalid label name {name:?}"));
        }
        rest = rest[eq + 1..].trim_start();
        let mut chars = rest.char_indices();
        if chars.next().map(|c| c.1) != Some('"') {
            return Err(format!("label {name} value is not quoted"));
        }
        let mut value = String::new();
        let mut end = None;
        while let Some((i, c)) = chars.next() {
            match c {
                '\\' => match chars.next().map(|c| c.1) {
                    Some('\\') => value.push('\\'),
                    Some('"') => value.push('"'),
                    Some('n') => value.push('\n'),
                    other => return Err(format!("invalid escape \\{other:?} in label {name}")),
                },
                '"' => {
                    end = Some(i);
                    break;
                }
                '\n' => return Err("raw newline in a label value".into()),
                c => value.push(c),
            }
        }
        let end = end.ok_or(format!("unterminated value of label {name}"))?;
        if labels.insert(name.to_string(), value).is_some() {
            return Err(format!("label {name} repeated"));
        }
        rest = rest[end + 1..].trim_start();
        if let Some(r) = rest.strip_prefix(',') {
            rest = r;
        } else if !rest.starts_with('}') {
            return Err(format!("expected ',' or '}}' after label {name}"));
        }
    }
}

/// Strict parser for the Prometheus text exposition format 0.0.4: valid
/// names, escapes and values; at most one HELP and one TYPE per family,
/// before its samples; every family's lines in one contiguous group; no
/// duplicate series.
pub fn parse_prometheus(text: &str) -> Result<Vec<Sample>, String> {
    if !text.is_empty() && !text.ends_with('\n') {
        return Err("exposition must end with a newline".into());
    }
    let mut samples = Vec::new();
    let mut types: BTreeMap<String, String> = BTreeMap::new();
    let mut helps: BTreeSet<String> = BTreeSet::new();
    let mut with_samples: BTreeSet<String> = BTreeSet::new();
    let mut closed: BTreeSet<String> = BTreeSet::new();
    let mut current: Option<String> = None;
    let mut series: BTreeSet<String> = BTreeSet::new();
    let mut enter = |family: &str, current: &mut Option<String>| -> Result<(), String> {
        if current.as_deref() == Some(family) {
            return Ok(());
        }
        if closed.contains(family) {
            return Err(format!("lines for {family} are not one contiguous group"));
        }
        if let Some(prev) = current.take() {
            closed.insert(prev);
        }
        *current = Some(family.to_string());
        Ok(())
    };
    for (n, line) in text.lines().enumerate() {
        let at = |e: String| format!("line {}: {e}: {line:?}", n + 1);
        if line.contains('\r') {
            return Err(at("carriage return".into()));
        }
        if line.trim().is_empty() {
            continue;
        }
        if let Some(c) = line.strip_prefix('#') {
            let mut parts = c.trim_start().splitn(3, ' ');
            let kw = parts.next().unwrap_or("");
            if kw != "HELP" && kw != "TYPE" {
                continue; // plain comment
            }
            let name = parts.next().unwrap_or("");
            let arg = parts.next().unwrap_or("");
            if !valid_metric_name(name) {
                return Err(at(format!("invalid metric name {name:?}")));
            }
            enter(name, &mut current).map_err(at)?;
            if kw == "HELP" {
                if !helps.insert(name.to_string()) {
                    return Err(at("second HELP".into()));
                }
            } else {
                if !["counter", "gauge", "histogram", "summary", "untyped"].contains(&arg) {
                    return Err(at(format!("unknown type {arg:?}")));
                }
                if with_samples.contains(name) {
                    return Err(at("TYPE after samples".into()));
                }
                if types.insert(name.to_string(), arg.to_string()).is_some() {
                    return Err(at("second TYPE".into()));
                }
            }
            continue;
        }
        let name_end = line
            .find(['{', ' '])
            .ok_or_else(|| at("sample without value".into()))?;
        let name = &line[..name_end];
        if !valid_metric_name(name) {
            return Err(at(format!("invalid metric name {name:?}")));
        }
        let (labels, rest) = if line[name_end..].starts_with('{') {
            parse_labels(&line[name_end..]).map_err(at)?
        } else {
            (BTreeMap::new(), &line[name_end..])
        };
        let mut fields = rest.split_whitespace();
        let value = fields.next().ok_or_else(|| at("no value".into()))?;
        let value = parse_value(value).ok_or_else(|| at(format!("invalid value {value:?}")))?;
        if let Some(ts) = fields.next() {
            ts.parse::<i64>()
                .map_err(|_| at(format!("invalid timestamp {ts:?}")))?;
        }
        if fields.next().is_some() {
            return Err(at("trailing fields".into()));
        }
        let family = ["_bucket", "_sum", "_count"]
            .iter()
            .find_map(|suf| {
                let base = name.strip_suffix(suf)?;
                matches!(
                    types.get(base).map(String::as_str),
                    Some("histogram" | "summary")
                )
                .then(|| base.to_string())
            })
            .unwrap_or_else(|| name.to_string());
        enter(&family, &mut current).map_err(at)?;
        with_samples.insert(family);
        if !series.insert(format!("{name}{labels:?}")) {
            return Err(at("duplicate series".into()));
        }
        samples.push(Sample {
            name: name.to_string(),
            labels,
            value,
        });
    }
    Ok(samples)
}

/// The value of the one sample matching `name` and all of `labels`.
pub fn metric(samples: &[Sample], name: &str, labels: &[(&str, &str)]) -> Option<f64> {
    let found: Vec<&Sample> = samples
        .iter()
        .filter(|s| {
            s.name == name
                && labels
                    .iter()
                    .all(|(k, v)| s.labels.get(*k).map(String::as_str) == Some(*v))
        })
        .collect();
    assert!(
        found.len() <= 1,
        "{name}{labels:?} matches {} samples",
        found.len()
    );
    found.first().map(|s| s.value)
}

// ---------------------------------------------------------------- privacy

/// Every object key in `v`, recursively, with its JSON path.
pub fn keys(v: &Value, path: &str, out: &mut Vec<String>) {
    match v {
        Value::Object(m) => {
            for (k, x) in m {
                out.push(format!("{path}.{k}"));
                keys(x, &format!("{path}.{k}"), out);
            }
        }
        Value::Array(a) => {
            for (i, x) in a.iter().enumerate() {
                keys(x, &format!("{path}[{i}]"), out);
            }
        }
        _ => {}
    }
}

/// Paths of keys that name a vital sign.
pub fn vital_keys(v: &Value) -> Vec<String> {
    let mut all = Vec::new();
    keys(v, "$", &mut all);
    all.into_iter()
        .filter(|p| {
            let k = p.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
            ["breath", "bpm", "heart", "pause", "resp"]
                .iter()
                .any(|w| k.contains(w))
        })
        .collect()
}
