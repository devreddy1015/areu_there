//! `saans roomsim-export` end to end (spec/roomsim.md §13): run the binary
//! on a declared scene and check the three files it writes against the
//! recorder's pcap format, the firmware's datagram header and an
//! independent run of the `saans-roomsim` library. Everything checked here
//! is SIM output.

use saans_roomsim::numerics::round_half_even;
use saans_roomsim::{Config, Simulator};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const T0_NS: u64 = 1_790_000_000 * 1_000_000_000;
const CSI_MAGIC: u32 = 0xC511_0001;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn scene_path(name: &str) -> PathBuf {
    repo().join("spec/scenes").join(name)
}

/// A fresh, empty directory for one test.
fn temp_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "saans-roomsim-export-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn saans(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_saans"))
        .args(args)
        .output()
        .expect("run saans")
}

fn u16_le(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}
fn u32_le(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}
fn u16_be(b: &[u8], at: usize) -> u16 {
    u16::from_be_bytes([b[at], b[at + 1]])
}

/// RFC 791: the one's complement sum of a header with a correct checksum is 0xFFFF.
fn ipv4_header_ok(h: &[u8]) -> bool {
    let mut sum: u32 = h
        .chunks(2)
        .map(|c| u16::from_be_bytes([c[0], c[1]]) as u32)
        .sum();
    while sum >> 16 != 0 {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    sum == 0xFFFF
}

/// One pcap record: its timestamp in ns, source IPv4 and port, destination
/// IPv4 and port, IPv4 identification and flags, and the UDP payload. Every
/// header field the recorder writes is checked on the way.
struct Record {
    t_ns: u64,
    src: ([u8; 4], u16),
    dst: ([u8; 4], u16),
    ip_id: u16,
    payload: Vec<u8>,
}

fn read_pcap(bytes: &[u8]) -> Vec<Record> {
    // Global header: "<IHHiIII" = ns magic, v2.4, thiszone 0, sigfigs 0, snaplen 65535, LINKTYPE_RAW.
    assert!(bytes.len() >= 24, "no pcap global header");
    assert_eq!(
        u32_le(bytes, 0),
        0xA1B2_3C4D,
        "nanosecond little-endian magic"
    );
    assert_eq!((u16_le(bytes, 4), u16_le(bytes, 6)), (2, 4), "pcap v2.4");
    assert_eq!(u32_le(bytes, 8), 0, "thiszone");
    assert_eq!(u32_le(bytes, 12), 0, "sigfigs");
    assert_eq!(u32_le(bytes, 16), 65535, "snaplen");
    assert_eq!(u32_le(bytes, 20), 101, "LINKTYPE_RAW");
    let mut out = Vec::new();
    let mut at = 24;
    while at < bytes.len() {
        assert!(
            at + 16 <= bytes.len(),
            "truncated record header at byte {at}"
        );
        let (sec, nsec, incl, orig) = (
            u32_le(bytes, at),
            u32_le(bytes, at + 4),
            u32_le(bytes, at + 8) as usize,
            u32_le(bytes, at + 12) as usize,
        );
        assert!(nsec < 1_000_000_000, "nanoseconds {nsec}");
        assert_eq!(incl, orig, "nothing is cut by the snaplen");
        let data = &bytes[at + 16..at + 16 + incl];
        let ip = &data[..20];
        assert_eq!(ip[0], 0x45, "IPv4, 20-byte header");
        assert_eq!(u16_be(ip, 2) as usize, incl, "IPv4 total length");
        assert_eq!(u16_be(ip, 6), 0x4000, "don't-fragment, offset 0");
        assert_eq!((ip[8], ip[9]), (64, 17), "TTL 64, UDP");
        assert!(ipv4_header_ok(ip), "IPv4 header checksum");
        let udp = &data[20..28];
        assert_eq!(u16_be(udp, 4) as usize, incl - 20, "UDP length");
        assert_eq!(u16_be(udp, 6), 0, "UDP checksum not computed");
        out.push(Record {
            t_ns: sec as u64 * 1_000_000_000 + nsec as u64,
            src: (ip[12..16].try_into().unwrap(), u16_be(udp, 0)),
            dst: (ip[16..20].try_into().unwrap(), u16_be(udp, 2)),
            ip_id: u16_be(ip, 4),
            payload: data[28..].to_vec(),
        });
        at += 16 + incl;
    }
    out
}

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The warning of an export with the bedroom's four transmitters (§12.4).
const GATEWAY_WARNING_4: &str = "warning: SIM export with 4 transmitters: each receiver's datagrams interleave them and carry no transmitter id, so this pcap cannot be split per link; for a gateway-faithful export use roles with one tx and slot_s 0.05 (spec/roomsim.md §10)";

/// `saans roomsim-export SCENE --out PREFIX` plus `extra`.
fn export_to(scene: &Path, prefix: &Path, extra: &[&str]) -> Output {
    let mut args = vec![
        "roomsim-export",
        scene.to_str().unwrap(),
        "--out",
        prefix.to_str().unwrap(),
    ];
    args.extend_from_slice(extra);
    saans(&args)
}

#[test]
fn a_short_export_is_the_recorder_format_and_the_library_run() {
    let dir = temp_dir("bedroom");
    let cfg_path = dir.join("cfg.json");
    // Loss high enough that a 0.5 s run surely loses packets; a narrower
    // firmware buffer than the default, so the length check means something.
    std::fs::write(
        &cfg_path,
        r#"{"loss_prob": 0.25, "fw_subcarriers": 40, "seed": 1}"#,
    )
    .unwrap();
    let scene = scene_path("bedroom.json");
    let prefix = dir.join("run");
    let out = saans(&[
        "roomsim-export",
        scene.to_str().unwrap(),
        "--out",
        prefix.to_str().unwrap(),
        "--seed",
        "11",
        "--duration",
        "0.5",
        "--config",
        cfg_path.to_str().unwrap(),
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "saans roomsim-export failed: {stderr}"
    );
    assert!(
        stderr.starts_with("SIM "),
        "the summary is labelled SIM: {stderr}"
    );

    // The same run through the library: --seed overrides the file's seed.
    let fw = 40usize;
    let cfg = Config::from_json_str(
        r#"{"loss_prob": 0.25, "fw_subcarriers": 40, "seed": 11, "duration_s": 0.5}"#,
    )
    .unwrap();
    let sc = saans_scene::Scene::load(&scene).unwrap();
    let sim = Simulator::new(&sc, cfg).unwrap();
    let packets: Vec<_> = sim.packets().collect();
    let received: Vec<_> = packets.iter().filter(|p| !p.lost).collect();
    let n_lost = packets.len() - received.len();
    assert!(n_lost > 0, "the check below needs lost packets");

    // PREFIX.sim.pcap: one record per received packet, in run order.
    let recs = read_pcap(&std::fs::read(dir.join("run.sim.pcap")).unwrap());
    assert_eq!(
        recs.len(),
        received.len(),
        "one datagram per received packet, none for a lost one"
    );
    for (i, (r, p)) in recs.iter().zip(&received).enumerate() {
        assert_eq!(
            r.ip_id, i as u16,
            "the recorder counts the IPv4 identification per file"
        );
        assert_eq!(
            r.src,
            ([10, 1, p.rx, 1], 5566),
            "10.1.<rx>.1:5566, a convention"
        );
        assert_eq!(r.dst, ([10, 0, 0, 1], 5566));
        assert_eq!(
            r.t_ns,
            T0_NS + round_half_even(p.t * 1e9) as u64,
            "1_790_000_000 s + t"
        );
        let d = &r.payload;
        assert_eq!(
            d.len(),
            20 + 2 * fw,
            "20-byte header + fw_subcarriers (imag, real) pairs"
        );
        assert_eq!(u32_le(d, 0), CSI_MAGIC, "raw CSI magic");
        assert_eq!(d[4], p.rx, "node id is the receiver");
        assert_eq!(d[5], 1, "one antenna");
        assert_eq!(u16_le(d, 6) as usize, fw, "n_subcarriers");
        assert_eq!(u32_le(d, 8), 2437, "freq_mhz of channel 6");
        let rcv = p.received.as_ref().unwrap();
        assert_eq!(u32_le(d, 12), rcv.seq);
        assert_eq!(d[16] as i8 as i32, rcv.rssi);
        assert_eq!(
            Some(d),
            rcv.datagram.as_ref(),
            "the library's datagram, byte for byte"
        );
    }

    // PREFIX.manifest.json.
    let m = read_json(&dir.join("run.manifest.json"));
    assert_eq!(m["label"], "SIM");
    assert_eq!(m["spec"], "roomsim v1");
    assert_eq!(m["scene_name"], "bedroom-4.2x3.6");
    assert_eq!(m["provenance"], "hand_written");
    let sha: String = Sha256::digest(std::fs::read(&scene).unwrap())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        m["scene_sha256"],
        sha.as_str(),
        "the digest of the scene file's bytes"
    );
    let c = &m["config"];
    assert_eq!(c["seed"], 11, "--seed wins over the file");
    assert_eq!(c["duration_s"], 0.5);
    assert_eq!(c["loss_prob"], 0.25);
    assert_eq!(c["fw_subcarriers"], 40);
    assert_eq!(c["smoothing"], false, "defaults are written too");
    let all = serde_json::to_value(Config::default()).unwrap();
    assert_eq!(
        c.as_object().unwrap().len(),
        all.as_object().unwrap().len(),
        "every §3.2 value"
    );
    assert_eq!(c.as_object().unwrap().len(), 75);
    assert_eq!(m["per_link_rate_hz"]["0,1"], 20.0);
    assert_eq!(m["per_link_rate_hz"].as_object().unwrap().len(), 12);
    let links = m["links"].as_array().unwrap();
    assert_eq!(links.len(), 12);
    for (i, l) in links.iter().enumerate() {
        let want: Vec<_> = packets.iter().filter(|p| p.link_index == i).collect();
        assert_eq!(l["link_index"], i);
        assert_eq!(l["packets"], want.len());
        assert_eq!(l["lost"], want.iter().filter(|p| p.lost).count());
        assert_eq!(l["datagrams"], want.iter().filter(|p| !p.lost).count());
    }
    let counts = &m["counts"];
    assert_eq!(counts["packets"], packets.len());
    assert_eq!(counts["lost"], n_lost);
    assert_eq!(counts["datagrams"], recs.len());
    assert!(
        m.get("assumptions").is_none(),
        "the bedroom lists its walls"
    );
    // Four txrx nodes: every receiver hears the other three at 20 Hz each,
    // and the stream of one receiver interleaves three transmitters.
    assert_eq!(
        m["receiver_rate_hz"],
        serde_json::json!({"0": 60.0, "1": 60.0, "2": 60.0, "3": 60.0})
    );
    assert_eq!(m["gateway_faithful"], false);
    assert!(
        stderr.contains(GATEWAY_WARNING_4),
        "the CLI warns that this pcap cannot be split per link: {stderr}"
    );

    // PREFIX.truth.jsonl: one line per transmission, lost or not.
    let text = std::fs::read_to_string(dir.join("run.truth.jsonl")).unwrap();
    let lines: Vec<serde_json::Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let transmissions: Vec<_> = sim.run().collect();
    assert_eq!(lines.len(), transmissions.len());
    assert_eq!(counts["transmissions"], lines.len());
    for (l, tr) in lines.iter().zip(&transmissions) {
        assert_eq!(l["label"], "SIM");
        assert_eq!(l["t"], tr.t);
        assert_eq!(l["tx"], tr.tx);
        let people = l["people"].as_array().unwrap();
        assert_eq!(people.len(), tr.truth.people.len());
        for (i, (p, want)) in people.iter().zip(&tr.truth.people).enumerate() {
            assert_eq!(p["name"], format!("people[{i}]"));
            assert_eq!(
                (p["x"].as_f64(), p["y"].as_f64()),
                (Some(want.x), Some(want.y))
            );
            assert_eq!(p["heading"].as_f64(), Some(want.heading));
            assert_eq!(p["walking"], want.walking);
            assert_eq!(p["posture"], want.posture.name());
        }
        let phone = tr.truth.phone.expect("the bedroom has a phone");
        assert_eq!(l["phone"], serde_json::json!([phone.x, phone.y, phone.z]));
        let lost: Vec<bool> = l["packets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["lost"].as_bool().unwrap())
            .collect();
        assert_eq!(lost, tr.packets.iter().map(|p| p.lost).collect::<Vec<_>>());
    }
    // A person is people[i] in every output, never their scene id (§3.1).
    assert!(!text.contains("\"sleeper\""));
    assert!(!std::fs::read_to_string(dir.join("run.manifest.json"))
        .unwrap()
        .contains("sleeper"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn implied_walls_are_recorded_as_sim_assumptions() {
    let dir = temp_dir("specular");
    let scene = repo().join("spec/scenes/fixtures/roomsim_specular.json");
    let prefix = dir.join("spec");
    let out = saans(&[
        "roomsim-export",
        scene.to_str().unwrap(),
        "--out",
        prefix.to_str().unwrap(),
        "--duration",
        "0.1",
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    let m = read_json(&dir.join("spec.manifest.json"));
    let a = m["assumptions"]
        .as_array()
        .expect("implied walls are an assumption");
    assert!(!a.is_empty());
    for s in a {
        assert!(s.as_str().unwrap().starts_with("SIM: "), "{s}");
    }
    assert!(
        stderr.contains("assumed: SIM: "),
        "the summary says so too: {stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn bad_invocations_are_refused_and_write_nothing() {
    let dir = temp_dir("refused");
    let scene = scene_path("bedroom.json");
    let scene = scene.to_str().unwrap();
    let prefix = dir.join("x");
    let prefix = prefix.to_str().unwrap();

    // Usage errors: exit 2, with the usage text.
    for args in [
        vec!["roomsim-export", scene],
        vec!["roomsim-export", "--out", prefix],
        vec!["roomsim-export", scene, "--out", prefix, "--seed", "abc"],
        vec!["roomsim-export", scene, "--out", prefix, "--seed", "-1"],
        vec!["roomsim-export", scene, "--out", prefix, "--duration"],
        vec!["roomsim-export", scene, "--out", prefix, "--sead", "3"],
        vec![
            "roomsim-export",
            scene,
            "--out",
            prefix,
            "--seed",
            "1",
            "--seed",
            "2",
        ],
        vec!["roomsim-export", scene, scene, "--out", prefix],
    ] {
        let out = saans(&args);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{args:?}: {stderr}");
        assert!(
            stderr.contains("saans roomsim-export SCENE.json --out PREFIX"),
            "{args:?}: {stderr}"
        );
    }

    // Rejected runs: exit 1, naming what is wrong.
    let cfg = |name: &str, text: &str| {
        let p = dir.join(name);
        std::fs::write(&p, text).unwrap();
        p.to_str().unwrap().to_string()
    };
    for (args, needle) in [
        (
            vec![
                "--config".to_string(),
                cfg("q.json", r#"{"quantize": false}"#),
            ],
            "quantize",
        ),
        (
            vec!["--config".to_string(), cfg("u.json", r#"{"seeed": 1}"#)],
            "seeed",
        ),
        (
            vec!["--config".to_string(), cfg("c.json", r#"{"channel": 14}"#)],
            "channel",
        ),
        (
            vec!["--duration".to_string(), "-1".to_string()],
            "duration_s",
        ),
        (
            vec!["--duration".to_string(), "nan".to_string()],
            "duration_s",
        ),
    ] {
        let mut all = vec![
            "roomsim-export".to_string(),
            scene.to_string(),
            "--out".to_string(),
            prefix.to_string(),
        ];
        all.extend(args.clone());
        let all: Vec<&str> = all.iter().map(String::as_str).collect();
        let out = saans(&all);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(1), "{args:?}: {stderr}");
        assert!(
            stderr.contains(needle),
            "{args:?} should name {needle}: {stderr}"
        );
    }
    let missing = saans(&["roomsim-export", "/nonexistent/scene.json", "--out", prefix]);
    assert_eq!(missing.status.code(), Some(1));

    for ext in ["sim.pcap", "truth.jsonl", "manifest.json"] {
        assert!(
            !dir.join(format!("x.{ext}")).exists(),
            "a refused run wrote x.{ext}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// §10, §13: today's firmware, one transmitter beaconing at 20 Hz (`roles`
/// one `tx`, `slot_s` 0.05), is the gateway-faithful export: no warning,
/// and each receiver hears one link at 20 Hz.
#[test]
fn one_transmitter_is_gateway_faithful() {
    let dir = temp_dir("faithful");
    let cfg_path = dir.join("cfg.json");
    let roles = r#""roles": {"0": "tx", "1": "rx", "2": "rx", "3": "rx"}, "slot_s": 0.05"#;
    std::fs::write(&cfg_path, format!("{{{roles}}}")).unwrap();
    let scene = scene_path("bedroom.json");
    let out = export_to(
        &scene,
        &dir.join("run"),
        &["--duration", "0.5", "--config", cfg_path.to_str().unwrap()],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(!stderr.contains("warning"), "{stderr}");
    assert!(!stderr.contains("note:"), "{stderr}");
    assert!(
        stderr.contains("on 3 link(s) at 20 Hz per link"),
        "{stderr}"
    );

    let cfg = Config::from_json_str(&format!("{{{roles}, \"duration_s\": 0.5}}")).unwrap();
    let sim = Simulator::new(&saans_scene::Scene::load(&scene).unwrap(), cfg).unwrap();
    let rate = sim.per_link_rate_hz(0);
    assert!((rate - 20.0).abs() < 1e-9, "{rate}");
    let m = read_json(&dir.join("run.manifest.json"));
    assert_eq!(m["gateway_faithful"], true);
    // Node 0 only transmits: it is no receiver.
    assert_eq!(
        m["receiver_rate_hz"],
        serde_json::json!({"1": rate, "2": rate, "3": rate})
    );
    assert_eq!(
        m["per_link_rate_hz"],
        serde_json::json!({"0,1": rate, "0,2": rate, "0,3": rate})
    );
    for r in read_pcap(&std::fs::read(dir.join("run.sim.pcap")).unwrap()) {
        let rx = r.src.0[2];
        assert!((1..=3).contains(&rx), "{:?}", r.src);
        assert_eq!(r.src, ([10, 1, rx, 1], 5566));
        assert_eq!(r.payload[4], rx, "the datagram's node id");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// §12.4, §10: a lone transmitter is gateway-faithful only with the
/// gateway's `raw.sample_rate_hz` at its per-link rate, so at any rate but
/// the gateway's default 20 Hz the CLI says so after the summary, with the
/// rate spelled as the summary spells it; the manifest's flag stays
/// `nT == 1`.
#[test]
fn a_lone_transmitter_off_20_hz_gets_a_note() {
    let dir = temp_dir("lone-rate");
    let scene = scene_path("bedroom.json");
    let roles = r#""roles": {"0": "tx", "1": "rx", "2": "rx", "3": "rx"}"#;
    for (i, (slot, rate)) in [("", "80"), (r#", "slot_s": 0.03"#, "33.333")]
        .into_iter()
        .enumerate()
    {
        let cfg_path = dir.join(format!("cfg{i}.json"));
        std::fs::write(&cfg_path, format!("{{{roles}{slot}}}")).unwrap();
        let prefix = dir.join(format!("run{i}"));
        let out = export_to(
            &scene,
            &prefix,
            &["--duration", "0.1", "--config", cfg_path.to_str().unwrap()],
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{stderr}");
        assert!(!stderr.contains("warning"), "{stderr}");
        let summary = format!("on 3 link(s) at {rate} Hz per link");
        let note = format!(
            "note: one transmitter at {rate} Hz per link; set the gateway's raw.sample_rate_hz to {rate}, or use slot_s 0.05 for its default 20 Hz (spec/roomsim.md §10)"
        );
        let lines: Vec<&str> = stderr.lines().collect();
        let at = |s: &str| lines.iter().position(|l| l.contains(s));
        let (Some(sum_at), Some(note_at)) = (at(&summary), at(&note)) else {
            panic!("want {summary:?} and {note:?}: {stderr}");
        };
        assert!(sum_at < note_at, "the note follows the summary: {stderr}");
        assert_eq!(lines.last(), Some(&note.as_str()), "{stderr}");
        let m = read_json(&dir.join(format!("run{i}.manifest.json")));
        assert_eq!(m["gateway_faithful"], true);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// §13: an export refuses to start when any of its three files exists,
/// naming it and touching nothing, unless given `--overwrite`, which first
/// deletes the old manifest; the manifest is written last.
#[test]
fn an_existing_export_is_never_replaced_without_overwrite() {
    let dir = temp_dir("overwrite");
    let scene = scene_path("bedroom.json");
    let prefix = dir.join("run");
    let short = ["--duration", "0.1"];
    let ext = ["sim.pcap", "truth.jsonl", "manifest.json"];
    let path = |e: &str| dir.join(format!("run.{e}"));

    // Each file alone is enough to refuse the run, and it is left as it was.
    for e in ext {
        std::fs::write(path(e), b"earlier").unwrap();
        let out = export_to(&scene, &prefix, &short);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(1), "{e}: {stderr}");
        assert!(
            stderr.contains(&format!("run.{e}: already exists")) && stderr.contains("--overwrite"),
            "{e}: {stderr}"
        );
        assert_eq!(std::fs::read(path(e)).unwrap(), b"earlier");
        for other in ext.iter().filter(|&&o| o != e) {
            assert!(!path(other).exists(), "{e}: a refused run wrote {other}");
        }
        std::fs::remove_file(path(e)).unwrap();
    }

    // A first export, then the same again: refused, naming the pcap.
    let first = export_to(&scene, &prefix, &short);
    assert!(first.status.success());
    let before: Vec<Vec<u8>> = ext
        .iter()
        .map(|e| std::fs::read(path(e)).unwrap())
        .collect();
    let again = export_to(&scene, &prefix, &short);
    let stderr = String::from_utf8_lossy(&again.stderr);
    assert_eq!(again.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("run.sim.pcap: already exists"), "{stderr}");
    let after: Vec<Vec<u8>> = ext
        .iter()
        .map(|e| std::fs::read(path(e)).unwrap())
        .collect();
    assert_eq!(before, after, "a refused run changes nothing");

    // --overwrite replaces all three.
    let out = export_to(&scene, &prefix, &["--duration", "0.2", "--overwrite"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let m = read_json(&path("manifest.json"));
    assert_eq!(m["config"]["duration_s"], 0.2);
    let pcap = std::fs::read(path("sim.pcap")).unwrap();
    assert!(pcap.len() > before[0].len(), "the longer run's pcap");

    // --overwrite deletes the old manifest before anything else: a run
    // that then fails (its truth file cannot be created) leaves none.
    std::fs::remove_file(path("truth.jsonl")).unwrap();
    std::fs::create_dir(path("truth.jsonl")).unwrap();
    let out = export_to(&scene, &prefix, &["--duration", "0.2", "--overwrite"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        !path("manifest.json").exists(),
        "no manifest describing other files"
    );

    let twice = export_to(&scene, &prefix, &["--overwrite", "--overwrite"]);
    assert_eq!(twice.status.code(), Some(2));
    let _ = std::fs::remove_dir_all(&dir);
}

/// §3.1: no `Person.id` or `Phone.id` appears in any output, error messages
/// included. An id serde would reject is reported by position only, before
/// serde (whose messages quote the value) sees it; a valid one never
/// reaches the files.
#[test]
fn people_and_phone_ids_never_appear_in_a_message() {
    let dir = temp_dir("ids");
    let text = std::fs::read_to_string(scene_path("bedroom.json")).unwrap();
    let base: serde_json::Value = serde_json::from_str(&text).unwrap();
    let person = r#"{ "id": "sleeper", "start": [1.1, 2.6] }"#;
    assert!(text.contains(person), "the bedroom's sleeper");
    let phone = r#"{ "id": "phone", "start": [1.1, 2.6] }"#;
    assert!(text.contains(phone), "the bedroom's phone");
    let with = |f: &dyn Fn(&mut serde_json::Value)| {
        let mut v = base.clone();
        f(&mut v);
        serde_json::to_string_pretty(&v).unwrap()
    };
    let cases: Vec<(String, &str)> = vec![
        (
            with(&|v| v["people"][0]["id"] = serde_json::json!(4242)),
            "people[0].id: expected a string",
        ),
        (
            with(&|v| v["people"][0]["id"] = serde_json::json!(["Asha 4242"])),
            "people[0].id: expected a string",
        ),
        (
            with(&|v| {
                v["people"]
                    .as_array_mut()
                    .unwrap()
                    .push(serde_json::json!({"id": true, "start": [2.0, 1.5]}))
            }),
            "people[1].id: expected a string",
        ),
        (
            with(&|v| v["people"] = serde_json::json!({"id": "Asha 4242"})),
            "people: expected an array",
        ),
        (
            with(&|v| v["people"] = serde_json::Value::Null),
            "people: expected an array",
        ),
        (
            with(&|v| v["people"][0] = serde_json::json!("Asha 4242")),
            "people[0]: expected an object",
        ),
        (
            with(&|v| v["phone"] = serde_json::json!(["Asha 4242"])),
            "phone: expected an object",
        ),
        (
            with(&|v| v["phone"]["id"] = serde_json::json!(4242.5)),
            "phone.id: expected a string",
        ),
        // A key given twice is checked every time, not only the last.
        (
            text.replace(
                person,
                r#"{ "id": 4242, "id": "sleeper", "start": [1.1, 2.6] }"#,
            ),
            "people[0].id: expected a string",
        ),
        (
            text.replace(
                phone,
                r#"{ "id": 4242, "id": "phone", "start": [1.1, 2.6] }"#,
            ),
            "phone.id: expected a string",
        ),
    ];
    for (i, (scene_text, want)) in cases.iter().enumerate() {
        let scene = dir.join(format!("scene{i}.json"));
        std::fs::write(&scene, scene_text).unwrap();
        let out = export_to(&scene, &dir.join(format!("run{i}")), &["--duration", "0.1"]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(1), "case {i}: {stderr}");
        assert!(
            stderr.trim_end().ends_with(want),
            "case {i}: want {want:?}: {stderr}"
        );
        assert!(
            !stderr.contains("4242") && !stderr.contains("Asha"),
            "case {i}: an id leaked: {stderr}"
        );
    }

    // A null id is no id (serde reads it so); a string id is accepted and
    // never written.
    let scene = dir.join("null.json");
    std::fs::write(
        &scene,
        with(&|v| {
            v["people"][0]["id"] = serde_json::Value::Null;
            v["phone"]["id"] = serde_json::json!("Asha 4242");
        }),
    )
    .unwrap();
    let out = export_to(&scene, &dir.join("null"), &["--duration", "0.1"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(!stderr.contains("Asha"), "{stderr}");
    for e in ["truth.jsonl", "manifest.json"] {
        let written = std::fs::read_to_string(dir.join(format!("null.{e}"))).unwrap();
        assert!(!written.contains("Asha"), "the phone's id is in null.{e}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// §3.1: the id check reports the first id problem even when the text
/// fails after it (a syntax error, nesting deeper than serde_json's limit,
/// a number out of range), and skips subtrees it does not care about as
/// serde's derive does, so neither a deep nor an out-of-range unknown key
/// stops it before the scene parser, which would quote the value.
#[test]
fn an_id_problem_is_reported_even_when_the_text_fails_later() {
    let dir = temp_dir("ids-later");
    let text = std::fs::read_to_string(scene_path("bedroom.json")).unwrap();
    let person = r#"{ "id": "sleeper", "start": [1.1, 2.6] }"#;
    let phone = r#"{ "id": "phone", "start": [1.1, 2.6] }"#;
    let people_key = r#""people": ["#;
    let phone_key = r#""phone": {"#;
    for s in [person, phone, people_key, phone_key] {
        assert_eq!(text.matches(s).count(), 1, "the bedroom has one {s}");
    }
    let deep = format!("{}{}", "[".repeat(200), "]".repeat(200));
    // A trailing comma before the final brace: a syntax error after
    // everything else.
    let trailing = |t: String| {
        let end = t
            .trim_end()
            .strip_suffix('}')
            .unwrap()
            .trim_end()
            .to_string();
        format!("{end},\n}}\n")
    };
    let cases: Vec<(String, &str, &str)> = vec![
        (
            trailing(text.replace(person, r#""Aman Gupta""#)),
            "people[0]: expected an object",
            "Aman",
        ),
        (
            trailing(text.replace(phone, r#""Aman phone""#)),
            "phone: expected an object",
            "Aman",
        ),
        (
            text.replace(person, r#"{ "id": 9876543210, "start": [1.1, 2.6] }"#)
                .replace(phone_key, &format!(r#""zz": {deep}, {phone_key}"#)),
            "people[0].id: expected a string",
            "9876543210",
        ),
        (
            text.replace(person, r#""Aman Gupta""#)
                .replace(people_key, &format!(r#""zz": 1e400, {people_key}"#)),
            "people[0]: expected an object",
            "Aman",
        ),
        (
            text.replace(phone, r#"{ "id": 4242, "start": [1.1, 2.6] }"#)
                .replace(people_key, &format!(r#""zz": {deep}, {people_key}"#)),
            "phone.id: expected a string",
            "4242",
        ),
        (
            text.replace(phone, r#"{ "id": 9876543210, "start": [1.1, 2.6], }"#),
            "phone.id: expected a string",
            "9876543210",
        ),
        // Inside a person too: a deep or out-of-range unknown key before
        // the id does not stop the check.
        (
            text.replace(
                person,
                &format!(r#"{{ "zz": {deep}, "yy": -1e400, "id": 4242, "start": [1.1, 2.6] }}"#),
            ),
            "people[0].id: expected a string",
            "4242",
        ),
    ];
    for (i, (scene_text, want, value)) in cases.iter().enumerate() {
        let scene = dir.join(format!("scene{i}.json"));
        std::fs::write(&scene, scene_text).unwrap();
        let prefix = dir.join(format!("run{i}"));
        let out = export_to(&scene, &prefix, &["--duration", "0.1"]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(1), "case {i}: {stderr}");
        assert!(
            stderr.trim_end().ends_with(want),
            "case {i}: want {want:?}: {stderr}"
        );
        assert!(
            !stderr.contains(value) && !stderr.contains("invalid type"),
            "case {i}: a value leaked: {stderr}"
        );
        assert!(
            !dir.join(format!("run{i}.sim.pcap")).exists(),
            "case {i} wrote a file"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// §3.2: a run needs a link; a one-node scene has none and is refused,
/// naming `roles`, before anything is written.
#[test]
fn a_one_node_scene_is_refused() {
    let dir = temp_dir("one-node");
    let mut v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(scene_path("bedroom.json")).unwrap())
            .unwrap();
    v["nodes"].as_array_mut().unwrap().truncate(1);
    let scene = dir.join("one.json");
    std::fs::write(&scene, serde_json::to_string(&v).unwrap()).unwrap();
    let out = export_to(&scene, &dir.join("run"), &["--duration", "0.1"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(
        stderr.contains("`roles`") && stderr.contains("no link"),
        "{stderr}"
    );
    for e in ["sim.pcap", "truth.jsonl", "manifest.json"] {
        assert!(!dir.join(format!("run.{e}")).exists());
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// One golden set in `spec/golden` (`NAME.kv` scalars, `NAME__ARRAY.f64`
/// little-endian f64 arrays), written by the Python reference on CPython
/// 3.14 (spec/roomsim.md §14). SIM data.
struct Golden {
    name: String,
    kv: std::collections::BTreeMap<String, String>,
}

impl Golden {
    fn load(name: &str) -> Self {
        let path = repo().join(format!("spec/golden/{name}.kv"));
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{}: {e} (run spec/golden/generate.py)", path.display()));
        let kv = text
            .lines()
            .filter_map(|l| l.split_once(' '))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Self {
            name: name.to_string(),
            kv,
        }
    }

    fn s(&self, k: &str) -> &str {
        self.kv
            .get(k)
            .unwrap_or_else(|| panic!("{}: no scalar {k}", self.name))
    }

    fn f(&self, k: &str) -> f64 {
        self.s(k)
            .parse()
            .unwrap_or_else(|_| panic!("{}: {k} not a number", self.name))
    }

    fn a(&self, k: &str) -> Vec<f64> {
        let n: usize = self.s(&format!("array.{k}")).parse().unwrap();
        let path = repo().join(format!("spec/golden/{}__{k}.f64", self.name));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(bytes.len(), n * 8, "{}: array {k} length", self.name);
        bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect()
    }
}

/// A JSON object's members in file order (serde_json's `Map` sorts them).
struct Ordered(Vec<(String, serde_json::Value)>);

impl<'de> serde::Deserialize<'de> for Ordered {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = Ordered;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut m: A,
            ) -> Result<Ordered, A::Error> {
                let mut out = Vec::new();
                while let Some(kv) = m.next_entry()? {
                    out.push(kv);
                }
                Ok(Ordered(out))
            }
        }
        d.deserialize_map(V)
    }
}

/// The §14 float rule, `M` the largest `|b_j|` of `want`; returns the worst
/// `|a - b| / |b|` over values with `|b| >= 1e-6*M`.
fn close(what: &str, got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let m = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let mut worst = 0.0f64;
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        let err = (a - b).abs();
        assert!(
            a == b || err <= 1e-9 * b.abs() + 1e-12 * m,
            "{what}[{i}]: got {a:e}, want {b:e}"
        );
        if err > 0.0 && b.abs() >= 1e-6 * m {
            worst = worst.max(err / b.abs());
        }
    }
    worst
}

/// §14 `roomsim_export`: the Python reference's `export_run` on the
/// bedroom (0.25 s, `loss_prob` 0.25) and `saans roomsim-export` on the
/// same scene and configuration write the same pcap, byte for byte, the
/// same truth values and the same manifest counts and rates.
#[test]
fn the_export_is_the_python_references() {
    let g = Golden::load("roomsim_export");
    assert_eq!(g.s("label"), "SIM");
    let scene = repo().join(g.s("scene"));
    let sha: String = Sha256::digest(std::fs::read(&scene).unwrap())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        g.s("scene_sha256"),
        sha,
        "{} changed since the set was generated (rerun spec/golden/generate.py)",
        g.s("scene")
    );

    // The configuration: every cfg.* scalar (each value that differs from §3.2).
    let mut cfg = serde_json::Map::new();
    for (k, text) in &g.kv {
        let Some(name) = k.strip_prefix("cfg.") else {
            continue;
        };
        assert!(!name.contains('.'), "{k}: only plain values are expected");
        let value: serde_json::Value = match text.parse::<i64>() {
            Ok(i) => i.into(),
            Err(_) => text.parse::<f64>().expect("a number").into(),
        };
        cfg.insert(name.to_string(), value);
    }
    assert!(cfg.contains_key("duration_s") && cfg.contains_key("loss_prob"));
    let dir = temp_dir("golden");
    let cfg_path = dir.join("cfg.json");
    std::fs::write(&cfg_path, serde_json::Value::Object(cfg).to_string()).unwrap();
    let prefix = dir.join("export");
    let out = export_to(&scene, &prefix, &["--config", cfg_path.to_str().unwrap()]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");

    // PREFIX.sim.pcap: byte for byte.
    let pcap = std::fs::read(dir.join("export.sim.pcap")).unwrap();
    let want: Vec<u8> = g
        .a("pcap")
        .iter()
        .map(|&b| {
            assert!(b.fract() == 0.0 && (0.0..=255.0).contains(&b));
            b as u8
        })
        .collect();
    assert_eq!(pcap.len() as f64, g.f("pcap_len"));
    if let Some(i) = pcap.iter().zip(&want).position(|(a, b)| a != b) {
        panic!("pcap byte {i}: got {}, want {}", pcap[i], want[i]);
    }
    assert_eq!(pcap, want);

    // PREFIX.truth.jsonl, value for value, as the generator lays it out.
    let text = std::fs::read_to_string(dir.join("export.truth.jsonl")).unwrap();
    let mut got: std::collections::BTreeMap<&str, Vec<f64>> = Default::default();
    let mut push = |k: &'static str, x: f64| got.entry(k).or_default().push(x);
    let num = |v: &serde_json::Value| v.as_f64().unwrap_or_else(|| panic!("{v}: not a number"));
    let mut n_people = None;
    for line in text.lines() {
        let rec: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(rec["label"], "SIM");
        for (k, key) in [
            ("tr_m", "m"),
            ("tr_slot", "slot"),
            ("tr_t", "t"),
            ("tr_tx", "tx"),
        ] {
            push(k, num(&rec[key]));
        }
        let packets = rec["packets"].as_array().unwrap();
        push("tr_npackets", packets.len() as f64);
        match &rec["phone"] {
            serde_json::Value::Null => {
                push("has_phone", 0.0);
                for k in ["phone_x", "phone_y", "phone_z"] {
                    push(k, 0.0);
                }
            }
            ph => {
                push("has_phone", 1.0);
                for (i, k) in ["phone_x", "phone_y", "phone_z"].into_iter().enumerate() {
                    push(k, num(&ph[i]));
                }
            }
        }
        let people = rec["people"].as_array().unwrap();
        assert_eq!(*n_people.get_or_insert(people.len()), people.len());
        for (i, p) in people.iter().enumerate() {
            assert_eq!(p["name"], format!("people[{i}]"));
            push("p_x", num(&p["x"]));
            push("p_y", num(&p["y"]));
            push("p_heading", num(&p["heading"]));
            push(
                "p_walking",
                f64::from(u8::from(p["walking"].as_bool().unwrap())),
            );
            let code = match p["posture"].as_str().unwrap() {
                "stand" => 0.0,
                "sit" => 1.0,
                "lie" => 2.0,
                other => panic!("posture {other}"),
            };
            push("p_posture", code);
        }
        for pk in packets {
            let lost = pk["lost"].as_bool().unwrap();
            push("pk_rx", num(&pk["rx"]));
            push("pk_link_index", num(&pk["link_index"]));
            push("pk_lost", f64::from(u8::from(lost)));
            for (k, sentinel) in [("seq", -1.0), ("rssi", 0.0), ("rssi_true", 0.0)] {
                let v = &pk[k];
                assert_eq!(v.is_null(), lost, "{k} is null exactly when lost");
                push(
                    match k {
                        "seq" => "pk_seq",
                        "rssi" => "pk_rssi",
                        _ => "pk_rssi_true",
                    },
                    if lost { sentinel } else { num(v) },
                );
            }
        }
    }
    assert_eq!(text.lines().count() as f64, g.f("n_transmissions"));
    assert_eq!(n_people.unwrap_or(0) as f64, g.f("n_people"));
    assert_eq!(got["pk_rx"].len() as f64, g.f("n_packets"));
    for k in [
        "tr_m",
        "tr_slot",
        "tr_tx",
        "tr_npackets",
        "has_phone",
        "p_walking",
        "p_posture",
        "pk_rx",
        "pk_link_index",
        "pk_lost",
        "pk_seq",
        "pk_rssi",
    ] {
        assert_eq!(got[k], g.a(k), "{k}: integers are identical");
    }
    let mut worst = 0.0f64;
    for k in [
        "tr_t",
        "phone_x",
        "phone_y",
        "phone_z",
        "p_heading",
        "pk_rssi_true",
    ] {
        worst = worst.max(close(k, &got[k], &g.a(k)));
    }
    // One trajectory's coordinates share M (§14).
    let mut xy = got["p_x"].clone();
    xy.extend(&got["p_y"]);
    let mut xy_want = g.a("p_x");
    xy_want.extend(g.a("p_y"));
    worst = worst.max(close("p_xy", &xy, &xy_want));

    // PREFIX.manifest.json: counts, rates and the gateway verdict.
    let man_text = std::fs::read_to_string(dir.join("export.manifest.json")).unwrap();
    let m: serde_json::Value = serde_json::from_str(&man_text).unwrap();
    assert_eq!(m["label"], "SIM");
    assert_eq!(m["scene_sha256"], sha.as_str());
    for k in ["transmissions", "packets", "lost", "datagrams"] {
        assert_eq!(
            num(&m["counts"][k]),
            g.f(&format!("counts_{k}")),
            "counts.{k}"
        );
    }
    assert_eq!(
        f64::from(u8::from(m["gateway_faithful"].as_bool().unwrap())),
        g.f("gateway_faithful")
    );
    if g.f("gateway_faithful") == 0.0 {
        assert!(stderr.contains(GATEWAY_WARNING_4), "{stderr}");
    }
    #[derive(serde::Deserialize)]
    struct Rates {
        receiver_rate_hz: Ordered,
    }
    let rates: Rates = serde_json::from_str(&man_text).unwrap();
    let keys: Vec<&str> = rates
        .receiver_rate_hz
        .0
        .iter()
        .map(|(k, _)| k.as_str())
        .collect();
    assert_eq!(
        keys.join(","),
        g.s("receivers"),
        "receivers ascending, in file order"
    );
    assert_eq!(keys.len() as f64, g.f("n_receivers"));
    let r_got: Vec<f64> = rates
        .receiver_rate_hz
        .0
        .iter()
        .map(|(_, v)| num(v))
        .collect();
    let r_want: Vec<f64> = keys
        .iter()
        .map(|k| g.f(&format!("receiver_rate_hz.{k}")))
        .collect();
    worst = worst.max(close("receiver_rate_hz", &r_got, &r_want));
    let links = m["links"].as_array().unwrap();
    let col = |key: &str| -> Vec<f64> { links.iter().map(|l| num(&l[key])).collect() };
    for (k, key) in [
        ("link_tx", "tx"),
        ("link_rx", "rx"),
        ("link_packets", "packets"),
        ("link_lost", "lost"),
        ("link_datagrams", "datagrams"),
    ] {
        assert_eq!(col(key), g.a(k), "manifest {k}");
    }
    worst = worst.max(close(
        "link_rate_hz",
        &col("per_link_rate_hz"),
        &g.a("link_rate_hz"),
    ));
    println!(
        "roomsim_export: pcap {} bytes identical; worst rel {worst:e}",
        pcap.len()
    );
    let _ = std::fs::remove_dir_all(&dir);
}
