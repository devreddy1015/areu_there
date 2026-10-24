//! REPLAY on the room stream (spec/livedemo.md §6): the binary started
//! with `--room-replay` on a file written here by saans-locate's writer
//! plays it with no truth and no error figures, and `--check` refuses a
//! bad file with exit code 2, naming the problem. Every record is SIM.

mod support;

use std::process::Command;
use support::*;

const LAB: &str = "[[space]]\nid = \"lab\"\nlinks = [{ rx = 1, tx = 2 }]\n";

/// `--room-replay` exists only in a `roomsim` build.
#[cfg(not(feature = "roomsim"))]
#[test]
fn room_replay_is_not_a_flag_of_a_default_build() {
    let dir = temp_dir("room-replay-absent");
    let cfg = dir.join("saans.toml");
    std::fs::write(&cfg, config_toml(&dir.join("data"), LAB)).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_saans-gateway"))
        .arg("--check")
        .arg("--config")
        .arg(&cfg)
        .arg("--room-replay")
        .arg(dir.join("run.jsonl"))
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--room-replay"));
}

#[cfg(feature = "roomsim")]
mod roomsim {
    use super::*;
    use saans_locate::phone::Config;
    use saans_locate::records::{write_records, Manifest, Records};
    use saans_locate::score::{false_alarms, score, PersonEstimator};
    use saans_locate::stream::RoomSimSetup;
    use saans_locate::variant::Variant;
    use saans_roomsim::Config as RoomSimConfig;
    use saans_scene::Scene;
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::io::{BufRead, BufReader};
    use std::net::SocketAddr;
    use std::path::Path;
    use std::process::Stdio;
    use std::time::{Duration, Instant};

    /// The shipped bedroom: its file's text and the scene.
    fn bedroom() -> (String, Scene) {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../spec/scenes/bedroom.json");
        let text = std::fs::read_to_string(path).unwrap();
        let scene = Scene::from_json_str("bedroom", &text).unwrap();
        (text, scene)
    }

    /// A `with_walker` run of the bedroom, seed 0, as a records file
    /// (§6.1), with its empty run.
    fn bedroom_records(duration_s: f64) -> Records {
        let (text, scene) = bedroom();
        let base = RoomSimConfig {
            seed: 0,
            duration_s,
            ..RoomSimConfig::default()
        };
        let rs = Variant::WithWalker.apply(&scene, &base);
        let setup = RoomSimSetup::new(&scene, &rs, &Config::default(), "SIM", &mut || true)
            .unwrap()
            .unwrap();
        let all = setup.stream().unwrap().advance_to(f64::INFINITY).unwrap();
        let empty_setup = setup.empty_room().unwrap();
        let empty = empty_setup
            .stream()
            .unwrap()
            .advance_to(f64::INFINITY)
            .unwrap();
        let hz = setup.tick_hz();
        Records {
            manifest: Manifest {
                scene_name: scene.name.clone(),
                scene_sha256: Sha256::digest(text.as_bytes())
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect(),
                provenance: scene.provenance,
                variant: Variant::WithWalker,
                seed: 0,
                duration_s,
                roomsim_config: rs,
                locate_config: setup.config().clone(),
                links: setup.links().to_vec(),
                has_phone: setup.has_phone(),
                empty_run: Some((
                    false_alarms(&empty.ticks, PersonEstimator::Wlm, hz),
                    false_alarms(&empty.ticks, PersonEstimator::Rti, hz),
                )),
            },
            ticks: all.ticks,
            links: all.links,
            epochs: all.epochs,
        }
    }

    fn write(path: &Path, records: &Records) {
        let mut w = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        write_records(&mut w, records).unwrap();
    }

    /// A child process, killed and reaped on drop.
    struct Running(std::process::Child);

    impl Drop for Running {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    /// `saans-gateway --config CFG ARGS…` serving live; its HTTP address
    /// and its log up to the line that names it.
    fn serve(cfg: &Path, args: &[&Path]) -> (Running, SocketAddr, Vec<String>) {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_saans-gateway"));
        cmd.arg("--config").arg(cfg);
        for a in args {
            cmd.arg("--room-replay").arg(a);
        }
        let mut gw = Running(
            cmd.stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
        let (tx, log) = std::sync::mpsc::channel();
        let stderr = gw.0.stderr.take().unwrap();
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                let _ = tx.send(line);
            }
        });
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut seen = Vec::new();
        loop {
            let line = log
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .unwrap_or_else(|e| panic!("{e}: no address in its log:\n{}", seen.join("\n")));
            let addr = line
                .split_once("dashboard on http://")
                .map(|(_, at)| at.split('/').next().unwrap().to_string());
            seen.push(line);
            if let Some(addr) = addr {
                return (gw, addr.parse().unwrap(), seen);
            }
        }
    }

    fn check(dir: &Path, args: &[&Path]) -> (Option<i32>, String, String) {
        let cfg = dir.join("saans.toml");
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_saans-gateway"));
        cmd.arg("--check").arg("--config").arg(&cfg);
        for a in args {
            cmd.arg("--room-replay").arg(a);
        }
        let out = cmd.output().unwrap();
        (
            out.status.code(),
            String::from_utf8(out.stdout).unwrap(),
            String::from_utf8(out.stderr).unwrap(),
        )
    }

    /// Every key named `key`, as JSON paths.
    fn paths_of(v: &Value, key: &str, at: &str, out: &mut Vec<String>) {
        match v {
            Value::Object(m) => {
                for (k, x) in m {
                    let p = format!("{at}.{k}");
                    if k == key {
                        out.push(p.clone());
                    }
                    paths_of(x, key, &p, out);
                }
            }
            Value::Array(a) => a
                .iter()
                .enumerate()
                .for_each(|(i, x)| paths_of(x, key, &format!("{at}[{i}]"), out)),
            _ => {}
        }
    }

    /// §6.3 and §11: REPLAY from a gateway started with `--room-replay`:
    /// `ready` at once, frames on the same clock, mode `replay`, no `truth`
    /// or `blocked` key anywhere, and `stats` with no error figures, equal
    /// to the score of the file's records without them.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_recorded_run_replays_without_truth_or_errors() {
        let dir = temp_dir("room-replay");
        let records = bedroom_records(20.0);
        let file = dir.join("bedroom.jsonl");
        write(&file, &records);
        let cfg = dir.join("saans.toml");
        std::fs::write(&cfg, config_toml(&dir.join("data"), LAB)).unwrap();
        let (gw, addr, log) = serve(&cfg, &[&file]);
        assert!(
            log.iter()
                .any(|l| l.contains("room stream: recorded runs (REPLAY of SIM) for bedroom")),
            "{log:?}"
        );

        let origin = format!("http://{addr}");
        let h = [("Origin", origin.as_str())];
        let mut ws = Ws::connect_with(addr, "/api/roomsim?scene=bedroom&replay=1", &h).await;
        let ready = ws.next_json().await;
        assert_eq!(ready["type"], "state");
        assert_eq!(ready["state"], "ready");
        assert_eq!(
            (ready["mode"].as_str(), ready["label"].as_str()),
            (Some("replay"), Some("SIM"))
        );
        assert_eq!(
            ready["run"],
            json!({ "index": 0, "variant": "with_walker", "seed": 0, "duration_s": 20.0 })
        );
        assert_eq!(ready["scene"]["sha256"], records.manifest.scene_sha256);

        ws.send_text(r#"{"cmd":"speed","value":8}"#).await;
        let mut frames: Vec<Value> = Vec::new();
        loop {
            let f = ws.next_json().await;
            assert_eq!(f["type"], "frame", "{f}");
            assert_eq!(f["seq"], frames.len() as u64);
            assert_eq!(
                (f["mode"].as_str(), f["label"].as_str()),
                (Some("replay"), Some("SIM"))
            );
            assert!(f.get("truth").is_none(), "no truth in REPLAY");
            let mut blocked = Vec::new();
            paths_of(&f, "blocked", "$", &mut blocked);
            assert!(blocked.is_empty(), "{blocked:?}");
            for block in [
                &f["stats"]["wlm"]["occupied"],
                &f["stats"]["rti"]["occupied"],
                &f["stats"]["rtt"],
                &f["stats"]["rssi"],
            ] {
                for key in ["median_m", "p90_m", "inside_radius", "ref"] {
                    assert!(block.get(key).is_none(), "REPLAY keeps no {key}");
                }
                assert!(block.get("coverage").is_some() && block.get("radius_median_m").is_some());
            }
            assert!(
                f["stats"]["wlm"].get("empty_run").is_some(),
                "the manifest's empty run"
            );
            let ended = f["ended"] == true;
            frames.push(f);
            if ended {
                break;
            }
        }
        let last = frames.last().unwrap();
        assert_eq!(last["t"], 20.0);
        let (_, scene) = bedroom();
        let hz = records.manifest.locate_config.tick_hz;
        let mut want = score(&scene, &records.ticks, &records.epochs, true, hz).to_json_truthless();
        let (wlm, rti) = records.manifest.empty_run.unwrap();
        want["wlm"]["empty_run"] = serde_json::to_value(wlm).unwrap();
        want["rti"]["empty_run"] = serde_json::to_value(rti).unwrap();
        assert_eq!(last["stats"], want);
        let n_new: usize = frames
            .iter()
            .map(|f| f["new"]["persons"].as_array().unwrap().len())
            .sum();
        assert_eq!(
            n_new,
            records.ticks.len(),
            "every recorded tick is played once"
        );

        // Restart rewinds to the first record.
        ws.send_text(r#"{"cmd":"restart"}"#).await;
        let f = ws.next_json().await;
        assert_eq!(
            (f["run"]["index"].as_u64(), f["t"].as_f64()),
            (Some(1), Some(0.0))
        );
        ws.close().await;

        // The file fixes the variant and the seed; another scene has none.
        for (path, param) in [
            (
                "/api/roomsim?scene=bedroom&replay=1&variant=declared",
                "variant",
            ),
            ("/api/roomsim?scene=office&replay=1", "replay"),
        ] {
            let mut ws = Ws::connect_with(addr, path, &h).await;
            let e = ws.next_json().await;
            assert_eq!(
                (e["type"].as_str(), e["param"].as_str(), e["mode"].as_str()),
                (Some("error"), Some(param), Some("replay"))
            );
            assert!(matches!(
                ws.next_message().await,
                WsMsg::Close {
                    code: Some(1008),
                    ..
                }
            ));
        }
        // SIM still runs beside it.
        let mut ws = Ws::connect_with(addr, "/api/roomsim?scene=bedroom", &h).await;
        assert_eq!(ws.next_json().await["state"], "calibrating");
        ws.close().await;
        drop(gw);
    }

    /// §6.2: `--check --room-replay` reads and checks the files, and a bad
    /// one exits 2 naming the file, the line and the problem.
    #[test]
    fn check_refuses_a_bad_room_replay_file_with_exit_code_2() {
        let dir = temp_dir("room-replay-check");
        std::fs::write(dir.join("saans.toml"), config_toml(&dir.join("data"), LAB)).unwrap();
        let mut records = bedroom_records(5.0);
        let good = dir.join("good.jsonl");
        write(&good, &records);
        let (code, out, err) = check(&dir, &[&good]);
        assert_eq!(code, Some(0), "{err}");
        assert!(
            out.contains("--room-replay: OK (1 recorded runs, SIM: bedroom)"),
            "{out}"
        );

        let not_a_manifest = dir.join("not-a-manifest.jsonl");
        std::fs::write(&not_a_manifest, "{\"type\":\"tick\",\"label\":\"SIM\"}\n").unwrap();
        records.manifest.scene_sha256 = "0".repeat(64);
        let unknown_scene = dir.join("unknown-scene.jsonl");
        write(&unknown_scene, &records);
        let missing = dir.join("missing.jsonl");
        let cases: [(&[&Path], &str); 4] = [
            (
                &[&not_a_manifest],
                "not-a-manifest.jsonl: line 1: the first line is not a manifest",
            ),
            (
                &[&unknown_scene],
                "unknown-scene.jsonl: line 1: scene.sha256 matches no embedded scene",
            ),
            (&[&good, &good], "line 1: scene: "),
            (&[&missing], "missing.jsonl: cannot be read"),
        ];
        for (args, why) in cases {
            let (code, out, err) = check(&dir, args);
            assert_eq!(code, Some(2), "{args:?}: {out}{err}");
            assert!(err.starts_with("saans-gateway: --room-replay "), "{err}");
            assert!(err.contains(why), "{args:?}: {err}");
            assert!(!out.contains("OK"), "--check does not pass: {out}");
        }
        let (_, _, err) = check(&dir, &[&good, &good]);
        assert!(
            err.contains("already covers this scene; one file per scene"),
            "{err}"
        );
    }
}
