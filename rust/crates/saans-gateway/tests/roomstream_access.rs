//! The room stream's handshake (spec/livedemo.md §2), against real
//! gateways over real sockets: Origin and Host first and unaudited, then
//! the principal, then the position scope with its audit record, then the
//! session cap; and the refusals that come after the upgrade as an
//! `error` message and a close code. Every number on the stream is SIM.

mod support;

use support::*;

const LAB: &str = "[[space]]\nid = \"lab\"\nlinks = [{ rx = 1, tx = 2 }]\n";

/// Without the `roomsim` feature the path is like any unknown one.
#[cfg(not(feature = "roomsim"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_room_stream_is_absent_from_a_default_build() {
    let g = TestGw::start(config("room-absent", LAB)).await;
    let origin = same_origin(g.http());
    let r = Ws::try_connect(
        g.http(),
        "/api/roomsim?scene=bedroom",
        &[("Origin", &origin)],
    )
    .await;
    assert_eq!(r.err().map(|r| r.status), Some(404));
    assert_eq!(g.get("/api/roomsim").await.status, 404);
    g.stop().await;
}

#[cfg(feature = "roomsim")]
mod roomsim {
    use super::*;
    use serde_json::{json, Value};
    use std::path::Path;
    use std::time::{Duration, Instant};

    const ROOM: &str = "/api/roomsim?scene=bedroom";

    /// Every line of every audit file in `dir`.
    fn audit_lines(dir: &Path) -> Vec<Value> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with("audit-") && name.ends_with(".jsonl") {
                let text = std::fs::read_to_string(entry.path()).unwrap();
                out.extend(
                    text.lines()
                        .map(|l| serde_json::from_str::<Value>(l).unwrap()),
                );
            }
        }
        out
    }

    /// The handshake's status: 101 when it upgraded.
    async fn status(g: &TestGw, path: &str, headers: &[(&str, &str)]) -> u16 {
        match Ws::try_connect(g.http(), path, headers).await {
            Ok(ws) => {
                drop(ws);
                101
            }
            Err(r) => r.status,
        }
    }

    /// The first message of an accepted session is its `state`.
    async fn opens(g: &TestGw, path: &str, headers: &[(&str, &str)]) -> Ws {
        let mut ws = Ws::connect_with(g.http(), path, headers).await;
        let first = ws.next_json().await;
        assert_eq!(first["type"], "state", "{first}");
        assert_eq!(first["state"], "calibrating");
        assert_eq!(
            (first["mode"].as_str(), first["label"].as_str()),
            (Some("sim"), Some("SIM"))
        );
        ws
    }

    /// An `error` message naming `param`, then a close with `code`.
    async fn refused_after_upgrade(ws: &mut Ws, param: &str, code: u16) -> Value {
        let e = ws.next_json().await;
        assert_eq!(e["type"], "error", "{e}");
        assert_eq!(e["param"], param, "{e}");
        assert_eq!(e["label"], "SIM");
        match ws.next_message().await {
            WsMsg::Close { code: c, .. } => assert_eq!(c, Some(code), "{e}"),
            other => panic!("{other:?} after {e}"),
        }
        e
    }

    /// §2.2: checked first, refused with 403 and not audited, on an
    /// auth-off gateway including DNS rebinding (Origin and Host both
    /// carry the attacker's name).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn origin_and_host_are_checked_first_and_not_audited() {
        let g = TestGw::start(config("room-origin", LAB)).await;
        let addr = g.http();
        let port = addr.port();
        let origin = same_origin(addr);
        let evil_host = format!("evil.example:{port}");
        let evil_origin = format!("http://{evil_host}");
        let other_port = format!("http://127.0.0.1:{}", port.wrapping_add(1).max(1));
        let with_path = format!("http://127.0.0.1:{port}/x");
        let refused: [&[(&str, &str)]; 6] = [
            &[],
            &[("Origin", "null")],
            &[("Origin", "http://evil.example")],
            &[("Origin", &other_port)],
            &[("Origin", &evil_origin), ("Host", &evil_host)],
            &[("Origin", &with_path)],
        ];
        for headers in refused {
            assert_eq!(status(&g, ROOM, headers).await, 403, "{headers:?}");
        }
        // Refused before anything else: a valid-looking query does not
        // matter, nor does a missing one.
        assert_eq!(status(&g, "/api/roomsim", &[]).await, 403);

        // Same origin, by address or by name, upgrades.
        opens(&g, ROOM, &[("Origin", &origin)]).await.close().await;
        let local_host = format!("localhost:{port}");
        let local_origin = format!("http://LOCALHOST:{port}");
        opens(
            &g,
            ROOM,
            &[("Origin", &local_origin), ("Host", &local_host)],
        )
        .await
        .close()
        .await;
        assert!(
            audit_lines(&g.cfg.data_dir).is_empty(),
            "Origin and Host refusals are not audited"
        );
        g.stop().await;
    }

    /// §2.3: a missing credential gets 401 (audited by `principal()`),
    /// ops and occupancy get 403 and an audit `Denied` with their name and
    /// role, safety and admin get the stream. `/api/me` says which.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_stream_needs_a_principal_with_the_position_scope() {
        let g = TestGw::start(config("room-roles", &format!("{}{LAB}", auth_toml()))).await;
        let port = g.http().port();
        let origin = same_origin(g.http());
        let with = |role: &str| vec![("Origin", origin.clone()), ("Authorization", bearer(role))];

        // Origin is still first: a valid admin token without it is a 403,
        // and nothing is audited.
        let admin = bearer("admin");
        assert_eq!(status(&g, ROOM, &[("Authorization", &admin)]).await, 403);
        assert!(audit_lines(&g.cfg.data_dir).is_empty());

        let r = Ws::try_connect(g.http(), ROOM, &[("Origin", &origin)])
            .await
            .err()
            .expect("refused");
        assert_eq!(r.status, 401);
        assert_eq!(r.header("www-authenticate"), Some("Bearer realm=\"saans\""));
        let r = Ws::try_connect(
            g.http(),
            ROOM,
            &[("Origin", &origin), ("Authorization", "Bearer nope")],
        )
        .await
        .err();
        assert_eq!(r.map(|r| r.status), Some(401));

        for role in ["ops", "occupancy"] {
            let h = with(role);
            let h: Vec<(&str, &str)> = h.iter().map(|(k, v)| (*k, v.as_str())).collect();
            let r = Ws::try_connect(g.http(), ROOM, &h)
                .await
                .err()
                .expect("refused");
            assert_eq!(r.status, 403, "{role}");
        }
        for role in ["safety", "admin"] {
            let h = with(role);
            let h: Vec<(&str, &str)> = h.iter().map(|(k, v)| (*k, v.as_str())).collect();
            opens(&g, ROOM, &h).await.close().await;
        }
        // DNS rebinding is closed by the loopback rule only where every
        // caller is the local admin; with authentication on, a token is the
        // gate and a proxy's Host is accepted.
        let host = format!("gw.example:{port}");
        let named = format!("http://{host}");
        opens(
            &g,
            ROOM,
            &[
                ("Origin", &named),
                ("Host", &host),
                ("Authorization", &admin),
            ],
        )
        .await
        .close()
        .await;

        let lines = audit_lines(&g.cfg.data_dir);
        let denied: Vec<&Value> = lines.iter().filter(|l| l["action"] == "denied").collect();
        let position: Vec<(Value, Value)> = denied
            .iter()
            .filter(|l| l["detail"]["why"] == "position")
            .map(|l| (l["who"].clone(), l["role"].clone()))
            .collect();
        assert_eq!(
            position,
            [
                (json!("ops"), json!("ops")),
                (json!("occupancy"), json!("occupancy"))
            ]
        );
        for l in denied.iter().filter(|l| l["detail"]["why"] == "position") {
            assert_eq!(
                l["detail"],
                json!({ "path": "/api/roomsim", "why": "position" })
            );
        }
        let unauthenticated: Vec<&Value> = denied
            .iter()
            .copied()
            .filter(|l| l["detail"]["path"] == "/api/roomsim" && l["who"] == "-")
            .collect();
        assert_eq!(
            unauthenticated.len(),
            2,
            "both 401s are audited: {unauthenticated:?}"
        );

        for (role, can) in [
            ("admin", true),
            ("safety", true),
            ("ops", false),
            ("occupancy", false),
        ] {
            let me = g
                .json_with("/api/me", &[("Authorization", &bearer(role))])
                .await;
            assert_eq!(me["can"]["position"], can, "{role}");
        }
        assert_eq!(
            g.json("/api/me").await["can"]["position"],
            false,
            "anonymous"
        );
        g.stop().await;
    }

    /// §2.4, §3.1: a bad query, an unknown key, a scene that is not
    /// embedded and a replay with no recorded run are refused after the
    /// upgrade, naming the parameter and never the value; a binary
    /// message closes with 1008.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn refusals_after_the_upgrade_name_the_parameter_then_close_1008() {
        let g = TestGw::start(config("room-params", LAB)).await;
        let origin = same_origin(g.http());
        let h = [("Origin", origin.as_str())];
        for (path, param, mode) in [
            (
                "/api/roomsim?scene=bedroom&variant=sleeper",
                "variant",
                "sim",
            ),
            ("/api/roomsim?scene=bedroom&seed=sleeper", "seed", "sim"),
            ("/api/roomsim?scene=bedroom&sleeper=1", "query", "sim"),
            ("/api/roomsim?scene=sleeper", "scene", "sim"),
            ("/api/roomsim", "scene", "sim"),
            ("/api/roomsim?scene=bedroom&replay=1", "replay", "replay"),
            (
                "/api/roomsim?scene=bedroom&replay=1&seed=3",
                "seed",
                "replay",
            ),
        ] {
            let mut ws = Ws::connect_with(g.http(), path, &h).await;
            let e = refused_after_upgrade(&mut ws, param, 1008).await;
            assert_eq!(e["mode"], mode, "{path}");
            assert!(
                !e["message"].as_str().unwrap().contains("sleeper"),
                "{path}: the value is echoed: {e}"
            );
        }
        let mut ws = Ws::connect_with(g.http(), "/api/roomsim?scene=bedroom&replay=1", &h).await;
        let e = refused_after_upgrade(&mut ws, "replay", 1008).await;
        assert_eq!(e["message"], "no recorded run for this scene");

        let mut ws = opens(&g, ROOM, &h).await;
        ws.send_binary(b"\x00\x01").await;
        loop {
            match ws.next_message().await {
                WsMsg::Close { code, .. } => {
                    assert_eq!(code, Some(1008));
                    break;
                }
                WsMsg::Text(t) => assert!(t.contains("\"type\":\"frame\""), "{t}"),
            }
        }
        g.stop().await;
    }

    /// §2.4: four sessions at once, calibrating ones included; a fifth
    /// gets an error and 1013, and a slot is free again once a session's
    /// thread has exited.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_fifth_session_is_told_to_try_later() {
        let g = TestGw::start(config("room-cap", LAB)).await;
        let origin = same_origin(g.http());
        let h = [("Origin", origin.as_str())];
        let mut open = Vec::new();
        for _ in 0..4 {
            open.push(opens(&g, ROOM, &h).await);
        }
        let mut fifth = Ws::connect_with(g.http(), ROOM, &h).await;
        let e = refused_after_upgrade(&mut fifth, "session", 1013).await;
        assert_eq!(
            e["message"],
            "4 room simulations are running; try again later"
        );
        assert_eq!(e["mode"], "sim");
        // REPLAY sessions count too: the refusal names the requested mode.
        let mut replay =
            Ws::connect_with(g.http(), "/api/roomsim?scene=bedroom&replay=1", &h).await;
        assert_eq!(
            refused_after_upgrade(&mut replay, "session", 1013).await["mode"],
            "replay"
        );

        // Closing one frees its slot when its thread exits (within 0.5 s).
        open.pop().unwrap().close().await;
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            let mut ws = Ws::connect_with(g.http(), ROOM, &h).await;
            let first = ws.next_json().await;
            if first["type"] == "state" {
                ws.close().await;
                break;
            }
            assert_eq!(first["param"], "session", "{first}");
            assert!(Instant::now() < deadline, "the slot was not released");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        for ws in open {
            ws.close().await;
        }
        g.stop().await;
    }
}
