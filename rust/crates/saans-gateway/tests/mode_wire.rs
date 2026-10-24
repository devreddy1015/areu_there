//! The gateway's mode on the wire and the position scope in `/api/me`
//! (spec/livedemo.md §1, §2.3; ADR-008 d10, d15), against real gateways
//! over real sockets: the enum keeps the lower-case spelling `hello`,
//! `/api/me` and `/api/health` always had, every view maps it with no
//! default branch, and `can.position` answers per role. Also the e2e
//! harness's own header, refusal and frame paths (spec/livedemo.md §11),
//! checked from the other end of the wire.

mod support;

use saans_gateway::pcap::PcapWriter;
use saans_gateway::shared::Mode;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use support::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const LAB: &str = "[[space]]\nid = \"lab\"\nlinks = [{ rx = 1, tx = 2 }]\n";

/// `"mode":"<m>"` exactly as the gateway writes it.
fn wire(m: &str) -> String {
    format!("\"mode\":\"{m}\"")
}

/// Every `Mode` variant's wire name, from serde itself: deserialising an
/// unknown name fails with "expected one of `sim`, ..." naming each. A
/// fourth mode changes this list without anything here being edited.
fn serde_variants() -> Vec<String> {
    let e = serde_json::from_value::<Mode>("?".into())
        .unwrap_err()
        .to_string();
    let (_, list) = e
        .split_once(", expected ")
        .unwrap_or_else(|| panic!("no variant list in {e:?}"));
    list.split('`')
        .skip(1)
        .step_by(2)
        .map(String::from)
        .collect()
}

/// The `hello` a gateway in `mode` sends the local admin. Its full shape
/// is pinned: the enum changed the type, not one byte of the message.
fn hello_for(mode: &str) -> Value {
    json!({
        "type": "hello",
        "version": env!("CARGO_PKG_VERSION"),
        "mode": mode,
        "name": "local",
        "role": "admin",
        "can": { "system": true },
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_live_gateway_says_live_as_before() {
    let mut g = TestGw::start(config("mode-live", LAB)).await;
    assert_eq!(g.gw.shared.mode, Mode::Live);

    let mut ws = Ws::connect(g.http(), "/api/live").await;
    let hello = ws.next_text().await;
    assert!(hello.contains(&wire("live")), "{hello}");
    assert_eq!(
        serde_json::from_str::<Value>(&hello).unwrap(),
        hello_for("live")
    );

    // The mode is in the health published on every tick, not only the
    // first one.
    g.advance(0.5).await;
    let h = g.get("/api/health").await;
    assert_eq!(h.status, 200);
    assert!(h.body.contains(&wire("live")), "{}", h.body);
    assert_eq!(h.json()["t_s"], g.t);

    let me = g.get("/api/me").await;
    assert!(me.body.contains(&wire("live")), "{}", me.body);
    ws.close().await;
    g.stop().await;
}

/// A short recording: link 1<-2 for 5 s.
fn write_recording(path: &std::path::Path) {
    let file = std::fs::File::create(path).unwrap();
    let mut w = PcapWriter::new(std::io::BufWriter::new(file)).unwrap();
    let gw = SocketAddrV4::new(Ipv4Addr::new(10, 0, 0, 1), 5566);
    let node = SocketAddrV4::new(Ipv4Addr::new(10, 0, 0, 101), 5566);
    let base_ns: i64 = 1_790_000_000 * 1_000_000_000;
    for i in 0..10u32 {
        w.write(
            base_ns + i as i64 * 500_000_000,
            node,
            gw,
            &empty_frame(1, 2, i).encode(),
        )
        .unwrap();
    }
    w.into_inner().into_inner().unwrap();
}

/// A child process, killed and reaped on drop, so a failed assertion
/// leaves no gateway behind.
struct Running(std::process::Child);

impl Drop for Running {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// `saans-gateway --replay F --serve` as a process: main.rs builds the
/// `Shared` itself, so this fails if it ever passes another mode.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_replay_gateway_says_replay_as_before() {
    let dir = temp_dir("mode-replay");
    let pcap = dir.join("rec.pcap");
    write_recording(&pcap);
    let cfg = dir.join("saans.toml");
    std::fs::write(&cfg, config_toml(&dir.join("data"), LAB)).unwrap();
    let mut gw = Running(
        Command::new(env!("CARGO_BIN_EXE_saans-gateway"))
            .arg("--config")
            .arg(&cfg)
            .arg("--replay")
            .arg(&pcap)
            .arg("--serve")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    // Its log is drained to the end on a thread of its own: the gateway
    // logs after the lines waited for here and must not meet a closed
    // pipe.
    let (tx, log) = std::sync::mpsc::channel();
    let stderr = gw.0.stderr.take().unwrap();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            let _ = tx.send(line);
        }
    });
    // The address it bound (`http_addr` port 0), then the end of the
    // replay, so what is read below is the final state.
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut seen: Vec<String> = Vec::new();
    let mut addr: Option<SocketAddr> = None;
    while !seen.last().is_some_and(|l| l.contains("replay finished")) {
        let line = log
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .unwrap_or_else(|e| panic!("{e}: no end of replay in its log:\n{}", seen.join("\n")));
        if let Some((_, at)) = line.split_once("replay dashboard on http://") {
            addr = Some(at.trim_end_matches('/').parse().unwrap());
        }
        seen.push(line);
    }
    let addr = addr.unwrap_or_else(|| panic!("no address in its log:\n{}", seen.join("\n")));

    let mut ws = Ws::connect(addr, "/api/live").await;
    let hello = ws.next_text().await;
    assert!(hello.contains(&wire("replay")), "{hello}");
    assert_eq!(
        serde_json::from_str::<Value>(&hello).unwrap(),
        hello_for("replay")
    );
    let h = get(addr, "/api/health").await;
    assert_eq!(h.status, 200);
    assert!(h.body.contains(&wire("replay")), "{}", h.body);
    let me = get(addr, "/api/me").await;
    assert!(me.body.contains(&wire("replay")), "{}", me.body);
    ws.close().await;
    drop(gw);
}

/// `can.position` is `can_see_position(None)`: admin and safety, never
/// ops or occupancy, and false for a caller with no valid credential
/// (spec/livedemo.md §2.3). The rest of `can` is as before.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn me_reports_the_position_scope_per_role() {
    let g = TestGw::start(config("mode-roles", &format!("{}{LAB}", auth_toml()))).await;
    let expect = [
        (
            "admin",
            json!({ "system": true, "audit": true, "position": true }),
        ),
        (
            "ops",
            json!({ "system": true, "audit": false, "position": false }),
        ),
        (
            "safety",
            json!({ "system": false, "audit": false, "position": true }),
        ),
        (
            "occupancy",
            json!({ "system": false, "audit": false, "position": false }),
        ),
    ];
    for (role, can) in expect {
        let me = g
            .json_with("/api/me", &[("Authorization", &bearer(role))])
            .await;
        assert_eq!(me["authenticated"], true, "{role}: {me}");
        assert_eq!(me["role"], role);
        assert_eq!(me["can"], can, "{role}");
        assert_eq!(me["mode"], "live");
    }
    let anonymous = json!({ "system": false, "audit": false, "position": false });
    for headers in [
        &[][..],
        &[("Authorization", "Bearer not-a-token")][..],
        &[("Cookie", "saans_session=00")][..],
    ] {
        let me = g.json_with("/api/me", headers).await;
        assert_eq!(me["authenticated"], false, "{headers:?}: {me}");
        assert_eq!(me["can"], anonymous, "{headers:?}");
        assert_eq!(me["mode"], "live");
    }
    g.stop().await;

    // With authentication off every caller is the local admin.
    let g = TestGw::start(config("mode-local", LAB)).await;
    assert_eq!(g.json("/api/me").await["can"]["position"], true);
    g.stop().await;
}

/// The harness's credential, refusal and close paths against the
/// gateway's endpoints that exist today, so a later test can rely on
/// them.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_harness_sends_headers_and_observes_refusals_and_closes() {
    let mut g = TestGw::start(config("mode-harness", &format!("{}{LAB}", auth_toml()))).await;
    let origin = same_origin(g.http());

    // A refused handshake is an answer, not a panic.
    let refused = Ws::try_connect(g.http(), "/api/live", &[("Origin", &origin)])
        .await
        .err()
        .expect("no credential, no stream");
    assert_eq!(refused.status, 401);
    assert_eq!(refused.json()["error"], "authentication required");
    assert_eq!(
        get_with(
            g.http(),
            "/api/health",
            &[("Authorization", &bearer("safety"))]
        )
        .await
        .status,
        403
    );

    let auth = bearer("occupancy");
    let mut ws = Ws::connect_with(
        g.http(),
        "/api/live",
        &[("Origin", &origin), ("Authorization", &auth)],
    )
    .await;
    let hello = ws.next_json().await;
    assert_eq!(
        (
            hello["type"].as_str(),
            hello["mode"].as_str(),
            hello["role"].as_str()
        ),
        (Some("hello"), Some("live"), Some("occupancy"))
    );
    assert_eq!(
        hello["can"],
        json!({ "system": false }),
        "hello's `can` is unchanged"
    );
    assert_eq!(ws.next_json().await["type"], "state");

    // /api/live ignores client text but takes a frame of each length
    // encoding and stays up. What the frames carry is checked byte for
    // byte in `the_harness_masks_what_it_sends`.
    for n in [5, 300, 70_000] {
        ws.send_text(&"x".repeat(n)).await;
    }
    g.advance(0.5).await;
    match ws.next_message().await {
        WsMsg::Text(t) => assert!(t.contains("\"type\":\"state\""), "{t}"),
        other => panic!("{other:?}"),
    }

    // The gateway stopping sends a close frame with no code; it is
    // returned, not a panic.
    let stopping = tokio::spawn(g.stop());
    let close = loop {
        match ws.next_message().await {
            WsMsg::Text(_) => continue,
            c @ WsMsg::Close { .. } => break c,
        }
    };
    assert_eq!(
        close,
        WsMsg::Close {
            code: None,
            reason: String::new()
        }
    );
    stopping.await.unwrap();
}

/// What a server of our own reported, within the harness's 10 s: a
/// server task that panics or waits for bytes never sent fails the test
/// instead of hanging it.
async fn within<T>(next: impl std::future::Future<Output = Option<T>>) -> T {
    tokio::time::timeout(Duration::from_secs(10), next)
        .await
        .expect("the server saw nothing within 10 s")
        .expect("the server stopped")
}

/// One request head, through its blank line, read a byte at a time so
/// nothing after it is consumed.
async fn read_head(s: &mut TcpStream) -> String {
    let mut head = Vec::new();
    while !head.ends_with(b"\r\n\r\n") {
        head.push(s.read_u8().await.expect("a complete request head"));
    }
    String::from_utf8(head).unwrap()
}

/// `Host` lines in a request head, whatever their case.
fn host_lines(head: &str) -> Vec<&str> {
    head.split("\r\n")
        .filter(|l| l.to_ascii_lowercase().starts_with("host:"))
        .collect()
}

/// What the harness puts on the wire for a `Host` override, seen by a
/// server of our own: the gateway answers a duplicate `Host` like any
/// other request, so only this end can tell one line from two. The
/// refusal is answered on a connection left open, so its body must be
/// read by its `Content-Length`, not to the end of the stream.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_harness_sends_one_host_line() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let at = listener.local_addr().unwrap();
    let (heads_tx, mut heads) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(async move {
        loop {
            let (mut s, _) = listener.accept().await.unwrap();
            let heads_tx = heads_tx.clone();
            tokio::spawn(async move {
                let head = read_head(&mut s).await;
                let upgrade = head.contains("\r\nUpgrade: websocket\r\n");
                s.write_all(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 2\r\n\r\nno")
                    .await
                    .unwrap();
                heads_tx.send(head).unwrap();
                if upgrade {
                    // Open until the client lets go.
                    let _ = s.read_to_end(&mut Vec::new()).await;
                }
            });
        }
    });

    let forged = [
        ("Host", "evil.example:1"),
        ("Origin", "http://evil.example:1"),
    ];
    let r = Ws::try_connect(at, "/x", &forged)
        .await
        .err()
        .expect("refused");
    assert_eq!((r.status, r.body.as_str()), (403, "no"));
    let head = within(heads.recv()).await;
    assert!(head.starts_with("GET /x HTTP/1.1\r\n"), "{head}");
    assert_eq!(host_lines(&head), ["Host: evil.example:1"], "{head}");
    assert!(
        head.contains("\r\nOrigin: http://evil.example:1\r\n"),
        "{head}"
    );

    // A plain request, the override's name in any case, and no override.
    let r = get_with(at, "/y", &[("host", "evil.example:1")]).await;
    assert_eq!((r.status, r.body.as_str()), (403, "no"));
    let head = within(heads.recv()).await;
    assert_eq!(host_lines(&head), ["Host: evil.example:1"], "{head}");
    get(at, "/z").await;
    let head = within(heads.recv()).await;
    assert_eq!(host_lines(&head), [format!("Host: {at}")], "{head}");
}

/// One client frame as a server reads it: opcode, the 7-bit length
/// field, and the payload unmasked. Panics unless FIN is set and the
/// frame is masked, as RFC 6455 section 5.1 requires of a client.
async fn read_client_frame(s: &mut TcpStream) -> (u8, u8, Vec<u8>) {
    let (b0, b1) = (s.read_u8().await.unwrap(), s.read_u8().await.unwrap());
    assert_eq!(b0 & 0xF0, 0x80, "FIN and no reserved bits");
    assert_eq!(b1 & 0x80, 0x80, "client frames are masked");
    let field = b1 & 0x7F;
    let len = match field {
        126 => s.read_u16().await.unwrap() as usize,
        127 => s.read_u64().await.unwrap() as usize,
        n => n as usize,
    };
    let mut mask = [0u8; 4];
    s.read_exact(&mut mask).await.unwrap();
    let mut p = vec![0u8; len];
    s.read_exact(&mut p).await.unwrap();
    for (i, b) in p.iter_mut().enumerate() {
        *b ^= mask[i % 4];
    }
    (b0 & 0x0F, field, p)
}

/// One short unmasked server frame.
fn server_frame(opcode: u8, payload: &[u8]) -> Vec<u8> {
    assert!(payload.len() < 126);
    [&[0x80 | opcode, payload.len() as u8][..], payload].concat()
}

/// Text of `n` bytes that differs from byte to byte, so a mask applied
/// at the wrong offset cannot come out right.
fn varied(n: usize) -> String {
    (0..n).map(|i| (b'!' + (i % 94) as u8) as char).collect()
}

/// `Ws::send_text` and `Ws::close` as a server of our own sees them, and
/// a server's close with a code and reason as `next_message` returns it.
/// `/api/live` ignores client text, so the gateway cannot show what a
/// frame carried.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_harness_masks_what_it_sends() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let at = listener.local_addr().unwrap();
    let (frames_tx, mut frames) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(async move {
        loop {
            let (mut s, _) = listener.accept().await.unwrap();
            let frames_tx = frames_tx.clone();
            tokio::spawn(async move {
                let head = read_head(&mut s).await;
                assert!(
                    head.contains("\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"),
                    "{head}"
                );
                s.write_all(
                    b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n\
                      Connection: Upgrade\r\nSec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n",
                )
                .await
                .unwrap();
                if head.starts_with("GET /bye ") {
                    let mut out = server_frame(0x1, "hi ✓".as_bytes());
                    out.extend(server_frame(0x9, b"ping"));
                    out.extend(server_frame(0x8, b"\x03\xE9bye"));
                    s.write_all(&out).await.unwrap();
                    let _ = s.read_to_end(&mut Vec::new()).await;
                    return;
                }
                // Each client frame goes to the test; a close is answered
                // after one more message, as a server still sending may.
                loop {
                    let f = read_client_frame(&mut s).await;
                    let close = f.0 == 0x8;
                    frames_tx.send(f).unwrap();
                    if close {
                        let mut out = server_frame(0x1, b"late");
                        out.extend(server_frame(0x8, b"\x03\xE8"));
                        s.write_all(&out).await.unwrap();
                        return;
                    }
                }
            });
        }
    });

    let mut ws = Ws::connect(at, "/echo").await;
    let mut sent: Vec<String> = [0, 5, 125, 126, 300, 65_535, 65_536, 70_000]
        .map(varied)
        .into();
    sent.push("mode ✓ é".into());
    for text in &sent {
        ws.send_text(text).await;
        let (opcode, field, payload) = within(frames.recv()).await;
        let n = text.len();
        let want = match n {
            0..=125 => n as u8,
            126..=65_535 => 126,
            _ => 127,
        };
        assert_eq!((opcode, field), (0x1, want), "{n} bytes");
        assert!(payload == text.as_bytes(), "{n} bytes arrive as sent");
    }
    ws.close().await;
    assert_eq!(
        within(frames.recv()).await,
        (0x8, 2, 1000u16.to_be_bytes().to_vec()),
        "close says 1000"
    );

    let mut ws = Ws::connect(at, "/bye").await;
    assert_eq!(ws.next_message().await, WsMsg::Text("hi ✓".into()));
    assert_eq!(
        ws.next_message().await,
        WsMsg::Close {
            code: Some(1001),
            reason: "bye".into()
        },
        "the ping between them is skipped"
    );
}

/// The body of the dashboard's function `name`, through its closing
/// brace at the start of a line.
fn js_function<'a>(body: &'a str, name: &str) -> &'a str {
    let start = body
        .find(&format!("function {name}("))
        .unwrap_or_else(|| panic!("no function {name}"));
    let end = body[start..]
        .find("\n}\n")
        .unwrap_or_else(|| panic!("no end of {name}"));
    &body[start..start + end + 2]
}

/// The dashboard renders its badge by an exhaustive mapping over the
/// three wire values with no default branch, and shows an unknown value
/// as an error (spec/livedemo.md §1).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_dashboard_maps_every_mode_and_no_other() {
    let g = TestGw::start(config("mode-dashboard", LAB)).await;
    let r = g.get("/").await;
    assert_eq!(r.status, 200);
    let body = r.body;
    let mapping = js_function(&body, "modeBadge");
    let cases: Vec<&str> = mapping
        .match_indices("case '")
        .map(|(i, _)| mapping[i + 6..].split('\'').next().unwrap())
        .collect();
    // Against serde's own variant list, so a fourth mode fails here until
    // the dashboard maps it too.
    assert_eq!(cases, serde_variants(), "{mapping}");
    assert_eq!(cases, ["sim", "replay", "live"]);
    assert!(!mapping.contains("default"), "no default branch: {mapping}");
    assert!(
        mapping.contains(
            "'Replaying a capture file — not live sensors; a .sim.pcap is simulated (SIM)'"
        ),
        "{mapping}"
    );

    // Each view shows the mode through one function, which shows an
    // unknown value as an error pill.
    let shown = js_function(&body, "modePill");
    assert!(shown.contains("modeBadge(mode)"), "{shown}");
    assert!(shown.contains("pill('unknown mode', 'bad'"), "{shown}");
    for view in ["renderShell", "renderSystem"] {
        let f = js_function(&body, view);
        assert!(f.contains("modePill("), "{view} shows the mode: {f}");
    }
    // The raw value reaches nothing but the mapping: every `.mode` read
    // is an argument to it, or `hello` copying the field into `S.me`.
    for (i, _) in body.match_indices(".mode") {
        let before = &body[..i];
        assert!(
            ["modeBadge(me", "modePill(me", "modePill(h", "mode: m"]
                .iter()
                .any(|p| before.ends_with(p)),
            "a raw mode read at byte {i}: {}",
            &body[i.saturating_sub(60)..i + 20]
        );
    }
    g.stop().await;
}
