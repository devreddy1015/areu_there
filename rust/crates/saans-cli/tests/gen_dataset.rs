//! `saans gen-dataset` end to end (spec/dataset.md §11): run the binary on
//! small datasets (a few scenes of 2-4 s, so the dev profile stays quick)
//! and check its files against the Room Sim and Locate libraries, with a
//! zip/npy reader written here independently of the writer. Everything
//! checked here is SIM.

use saans_locate::phone::Config as LocateConfig;
use saans_locate::pipeline::{calibration_config, resolve_config, run_roomsim};
use saans_locate::radius::{Outcome, Reason};
use saans_locate::records::locate_config_json;
use saans_roomsim::{Buffer, Config as RoomSimConfig, Posture, Simulator};
use saans_scene::{Provenance, Scene};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// A fresh, empty directory for one test (not created: the command makes
/// it), removed when the test ends.
struct TempDir(PathBuf);

impl std::ops::Deref for TempDir {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for TempDir {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn temp_dir(name: &str) -> TempDir {
    let d = std::env::temp_dir().join(format!("saans-gen-dataset-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    TempDir(d)
}

fn saans(args: &[&str], env: &[(&str, &str)]) -> Output {
    let mut c = Command::new(env!("CARGO_BIN_EXE_saans"));
    c.args(args);
    for (k, v) in env {
        c.env(k, v);
    }
    c.output().expect("run saans")
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// `saans gen-dataset --out DIR` plus `extra`, which must succeed.
fn generate(dir: &Path, extra: &[&str]) -> Output {
    let d = dir.to_str().unwrap();
    let mut args = vec!["gen-dataset", "--out", d];
    args.extend_from_slice(extra);
    let out = saans(&args, &[]);
    assert!(out.status.success(), "{args:?}: {}", stderr(&out));
    for l in stdout(&out).lines().chain(stderr(&out).lines()) {
        assert!(l.starts_with("SIM  "), "an unlabelled line: {l}");
    }
    out
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

// ---------------------------------------------------------------------
// An independent reader of §7.2's bytes
// ---------------------------------------------------------------------

/// The IEEE CRC-32, bit by bit (the writer uses a table).
fn crc32(bytes: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in bytes {
        c ^= u32::from(b);
        for _ in 0..8 {
            c = if c & 1 == 1 {
                (c >> 1) ^ 0xEDB8_8320
            } else {
                c >> 1
            };
        }
    }
    !c
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

/// One `.npy` member.
#[derive(Debug, Clone, PartialEq)]
struct Npy {
    descr: String,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl Npy {
    fn len(&self) -> usize {
        self.shape.iter().product()
    }
    fn u8(&self) -> Vec<u8> {
        assert_eq!(self.descr, "|u1");
        self.data.clone()
    }
    fn i8(&self) -> Vec<i8> {
        assert_eq!(self.descr, "|i1");
        self.data.iter().map(|&b| b as i8).collect()
    }
    fn i32(&self) -> Vec<i32> {
        assert_eq!(self.descr, "<i4");
        self.data
            .chunks(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
    fn f32(&self) -> Vec<f32> {
        assert_eq!(self.descr, "<f4");
        self.data
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
    fn f64(&self) -> Vec<f64> {
        assert_eq!(self.descr, "<f8");
        self.data
            .chunks(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}

/// Parse an npy v1.0 member: magic, version, header padded to 64, a dict
/// with `descr`, `fortran_order` False and `shape`; then the data.
fn parse_npy(b: &[u8]) -> Npy {
    assert_eq!(&b[..8], b"\x93NUMPY\x01\x00");
    let hl = usize::from(u16_at(b, 8));
    let start = 10 + hl;
    assert_eq!(start % 64, 0, "data at a multiple of 64");
    let header = std::str::from_utf8(&b[10..start]).unwrap();
    assert!(header.ends_with('\n'));
    let dict = header.trim_end_matches('\n').trim_end_matches(' ');
    let rest = dict.strip_prefix("{'descr': '").unwrap();
    let (descr, rest) = rest.split_once('\'').unwrap();
    let rest = rest
        .strip_prefix(", 'fortran_order': False, 'shape': (")
        .unwrap();
    let (dims, rest) = rest.split_once(')').unwrap();
    assert_eq!(rest, ", }");
    let shape: Vec<usize> = dims
        .split(',')
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .map(|d| d.parse().unwrap())
        .collect();
    if shape.len() == 1 {
        assert!(dims.ends_with(','), "a 1-tuple keeps its comma: ({dims})");
    }
    let itemsize = match descr {
        "|u1" | "|i1" => 1,
        "<i4" | "<f4" => 4,
        "<f8" => 8,
        other => panic!("descr {other} is not one of §7.2's"),
    };
    let data = b[start..].to_vec();
    assert_eq!(data.len(), shape.iter().product::<usize>() * itemsize);
    Npy {
        descr: descr.to_string(),
        shape,
        data,
    }
}

/// Read a shard: every local header (version 20, flags 0, STORED, time 0,
/// date 0x0021, CRC, sizes, no extra), then the central directory and end
/// record, which must agree. Members in file order.
fn read_npz(bytes: &[u8]) -> Vec<(String, Npy)> {
    let mut at = 0;
    let mut members = Vec::new();
    let mut offsets = Vec::new();
    while u32_at(bytes, at) == 0x0403_4b50 {
        assert_eq!(&bytes[at + 4..at + 14], &[20, 0, 0, 0, 0, 0, 0, 0, 0x21, 0]);
        let crc = u32_at(bytes, at + 14);
        let size = u32_at(bytes, at + 18) as usize;
        assert_eq!(u32_at(bytes, at + 22) as usize, size, "STORED: sizes equal");
        let name_len = usize::from(u16_at(bytes, at + 26));
        assert_eq!(u16_at(bytes, at + 28), 0, "no extra field");
        let name = std::str::from_utf8(&bytes[at + 30..at + 30 + name_len])
            .unwrap()
            .to_string();
        let body = &bytes[at + 30 + name_len..at + 30 + name_len + size];
        assert_eq!(crc32(body), crc, "{name}: CRC-32");
        offsets.push((name.clone(), crc, size, at));
        let stem = name.strip_suffix(".npy").unwrap().to_string();
        members.push((stem, parse_npy(body)));
        at += 30 + name_len + size;
    }
    let cd = at;
    for (name, crc, size, offset) in &offsets {
        assert_eq!(u32_at(bytes, at), 0x0201_4b50);
        assert_eq!(
            &bytes[at + 4..at + 16],
            &[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0x21, 0]
        );
        assert_eq!(u32_at(bytes, at + 16), *crc);
        assert_eq!(u32_at(bytes, at + 20) as usize, *size);
        assert_eq!(u32_at(bytes, at + 24) as usize, *size);
        let name_len = usize::from(u16_at(bytes, at + 28));
        assert_eq!(
            &bytes[at + 30..at + 42],
            &[0; 12],
            "no extra, comment, disk or attributes"
        );
        assert_eq!(u32_at(bytes, at + 42) as usize, *offset);
        assert_eq!(&bytes[at + 46..at + 46 + name_len], name.as_bytes());
        at += 46 + name_len;
    }
    assert_eq!(u32_at(bytes, at), 0x0605_4b50);
    assert_eq!(u16_at(bytes, at + 4), 0);
    assert_eq!(u16_at(bytes, at + 6), 0);
    assert_eq!(usize::from(u16_at(bytes, at + 8)), offsets.len());
    assert_eq!(usize::from(u16_at(bytes, at + 10)), offsets.len());
    assert_eq!(u32_at(bytes, at + 12) as usize, at - cd);
    assert_eq!(u32_at(bytes, at + 16) as usize, cd);
    assert_eq!(u16_at(bytes, at + 20), 0, "no comment");
    assert_eq!(at + 22, bytes.len());
    members
}

/// §7.2's members, in order, with their dtypes.
const MEMBERS: [(&str, &str); 37] = [
    ("csi_iq", "|i1"),
    ("csi_mask", "|u1"),
    ("rssi", "|i1"),
    ("tx_slot", "|u1"),
    ("subcarrier_k", "|i1"),
    ("subcarrier_active", "|u1"),
    ("count", "|u1"),
    ("presence", "|u1"),
    ("xy", "<f4"),
    ("person_mask", "|u1"),
    ("walking", "|u1"),
    ("posture", "|u1"),
    ("zone", "|i1"),
    ("zone_occupied", "|u1"),
    ("zone_single", "|i1"),
    ("t_label", "<f8"),
    ("zone_floor_frac", "<f4"),
    ("zone_reachable", "|u1"),
    ("wlm_xy", "<f4"),
    ("rti_xy", "<f4"),
    ("wlm_radius", "<f4"),
    ("rti_radius", "<f4"),
    ("wlm_reason", "|u1"),
    ("rti_reason", "|u1"),
    ("calib_baseline", "<f8"),
    ("calib_null_sigma", "<f8"),
    ("calib_mask", "|u1"),
    ("calib_iq", "|i1"),
    ("calib_mask_m", "|u1"),
    ("calib_rssi", "|i1"),
    ("links", "|u1"),
    ("node_ids", "|u1"),
    ("node_pos", "<f4"),
    ("room_size", "<f4"),
    ("footprint", "<f4"),
    ("window_end_tick", "<i4"),
    ("meta_json", "|u1"),
];

/// The §5 label members: identical with and without impairments (§8).
const LABEL_MEMBERS: [&str; 12] = [
    "count",
    "presence",
    "xy",
    "person_mask",
    "walking",
    "posture",
    "zone",
    "zone_occupied",
    "zone_single",
    "t_label",
    "zone_floor_frac",
    "zone_reachable",
];

fn shard(dir: &Path, i: usize) -> BTreeMap<String, Npy> {
    let bytes = std::fs::read(dir.join(format!("shards/gen-{i:06}.npz"))).unwrap();
    let members = read_npz(&bytes);
    let names: Vec<&str> = members.iter().map(|(n, _)| n.as_str()).collect();
    let want: Vec<&str> = MEMBERS.iter().map(|(n, _)| *n).collect();
    assert_eq!(names, want, "the members in §7.2's order");
    for ((name, npy), (_, descr)) in members.iter().zip(MEMBERS) {
        assert_eq!(npy.descr, descr, "{name}");
    }
    members.into_iter().collect()
}

/// Every file under `dir/sub`, by name.
fn files(dir: &Path, sub: &str) -> BTreeMap<String, Vec<u8>> {
    std::fs::read_dir(dir.join(sub))
        .unwrap()
        .map(|e| {
            let p = e.unwrap().path();
            (
                p.file_name().unwrap().to_string_lossy().into_owned(),
                std::fs::read(&p).unwrap(),
            )
        })
        .collect()
}

/// The impairment switches of §8 in a config (and Room Sim's `impair`),
/// as the manifest's `impairment_switches` records them.
fn switches(rs: &RoomSimConfig) -> Value {
    serde_json::json!({
        "impair": rs.impair,
        "loss_prob": rs.loss_prob,
        "loss_prob_link": rs.loss_prob_link,
        "agc_jump_prob": rs.agc_jump_prob,
        "device_gain_db": [rs.device_gain_db.0, rs.device_gain_db.1],
        "ripple": rs.ripple,
        "cpe": rs.cpe,
        "timing_offset_s": rs.timing_offset_s,
        "timing_offset_dev_s": [rs.timing_offset_dev_s.0, rs.timing_offset_dev_s.1],
    })
}

/// The keys of the eight hardware switches §8's `--no-impair` sets.
const HARDWARE: [&str; 8] = [
    "loss_prob",
    "loss_prob_link",
    "agc_jump_prob",
    "device_gain_db",
    "ripple",
    "cpe",
    "timing_offset_s",
    "timing_offset_dev_s",
];

/// The manifest without the keys that legitimately differ.
fn manifest_without(dir: &Path, keys: &[&str]) -> Value {
    let mut m = read_json(&dir.join("manifest.json"));
    for k in keys {
        m.as_object_mut().unwrap().remove(*k);
    }
    m
}

// ---------------------------------------------------------------------
// §11 Stream, Labels, Baseline, Reproduction, Format
// ---------------------------------------------------------------------

/// A small dataset against the libraries, scene by scene: the stream is
/// the packets of `Simulator::new` on the written scene and config, the
/// labels its truth, the baseline `run_roomsim`'s ticks and calibration.
#[test]
fn a_small_dataset_is_the_library_run() {
    const SECONDS: usize = 4;
    let dir = temp_dir("library");
    let out = generate(
        &dir,
        &[
            "--scenes",
            "3",
            "--seconds",
            "4",
            "--seed",
            "11",
            "--threads",
            "3",
        ],
    );
    assert!(stdout(&out).contains("3 generated scenes x 4 s, seed 11, impairments on"));

    let m = read_json(&dir.join("manifest.json"));
    assert_eq!(m["label"], "SIM");
    assert_eq!(m["spec"], "dataset v1");
    assert_eq!(m["n_scenes"], 3);
    assert_eq!(m["seconds"], 4);
    assert_eq!(m["seed"], 11);
    assert_eq!(m["label_hz"], 2);
    assert_eq!(m["impair"], true);
    assert_eq!(m["impairments"], "on");
    assert_eq!(
        m["impairment_switches"],
        switches(&RoomSimConfig::default())
    );
    assert_eq!(m["baseline_valid"], true);
    assert_eq!(m["from"], Value::Null);
    assert_eq!(
        m["command"],
        "saans gen-dataset --scenes 3 --seconds 4 --seed 11"
    );
    assert_eq!(
        m["roomsim_defaults"],
        serde_json::to_value(RoomSimConfig::default()).unwrap()
    );
    assert_eq!(m["totals"]["labels"], 3 * 2 * SECONDS);
    let mut listed: Vec<u64> = ["train", "val", "test"]
        .iter()
        .flat_map(|s| {
            m["splits"][*s]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
        })
        .collect();
    listed.sort_unstable();
    assert_eq!(listed, vec![0, 1, 2], "every scene in exactly one split");

    for i in 0..3 {
        let id = format!("gen-{i:06}");
        let e = &m["scenes"][i];
        assert_eq!(e["index"], i);
        assert_eq!(e["id"], id.as_str());
        assert_eq!(e["shard"], format!("shards/{id}.npz"));
        let shard_bytes = std::fs::read(dir.join(format!("shards/{id}.npz"))).unwrap();
        assert_eq!(e["shard_sha256"], sha256_hex(&shard_bytes));
        let scene_bytes = std::fs::read(dir.join(format!("scenes/{id}.json"))).unwrap();
        assert_eq!(e["scene_sha256"], sha256_hex(&scene_bytes));
        assert!(m["splits"][e["split"].as_str().unwrap()]
            .as_array()
            .unwrap()
            .contains(&Value::from(i)));

        // Reproduction (§7.1): the written scene and config.
        let scene = Scene::from_json_str(&id, std::str::from_utf8(&scene_bytes).unwrap()).unwrap();
        assert_eq!(scene.name, id);
        assert_eq!(scene.provenance, Provenance::Generated);
        let rs_text =
            std::fs::read_to_string(dir.join(format!("scenes/{id}.roomsim.json"))).unwrap();
        let rs = RoomSimConfig::from_json_str(&rs_text).unwrap();
        assert_eq!(rs.duration_s, SECONDS as f64 + 0.5);
        assert!(rs.impair);
        assert_eq!(e["n_nodes"], scene.nodes.len());
        assert_eq!(e["n_people"], scene.people.len());
        let locate = resolve_config(&rs, &LocateConfig::default()).unwrap();
        assert_eq!(
            read_json(&dir.join(format!("scenes/{id}.locate.json"))),
            locate_config_json(&locate)
        );
        let a = shard(&dir, i);
        check_scene(&scene, &rs, &a, i, &scene_bytes);
    }
}

fn check_scene(
    scene: &Scene,
    rs: &RoomSimConfig,
    a: &BTreeMap<String, Npy>,
    index: usize,
    scene_bytes: &[u8],
) {
    let sim = Simulator::new(scene, rs.clone()).unwrap();
    let n = sim.transmitters().len();
    let links = sim.links();
    let l = links.len();
    assert_eq!(l, n * (n - 1));
    let seconds = (rs.duration_s - 0.5) as usize;
    let k_labels = 2 * seconds;
    let n_store = 80 * seconds;
    let columns = n_store.div_ceil(n);

    // §4.2 shapes and the grid's constants.
    assert_eq!(a["csi_iq"].shape, vec![l, 56, columns, 2]);
    assert_eq!(a["csi_mask"].shape, vec![l, columns]);
    assert_eq!(a["rssi"].shape, vec![l, columns]);
    let tx_slot = a["tx_slot"].u8();
    for (li, lk) in links.iter().enumerate() {
        assert_eq!(
            usize::from(tx_slot[li]),
            sim.transmitters().iter().position(|&t| t == lk.tx).unwrap()
        );
    }
    let bk = sim.buffer_k();
    assert_eq!(
        a["subcarrier_k"].i8(),
        bk[..56].iter().map(|&k| k as i8).collect::<Vec<_>>()
    );
    let active = a["subcarrier_active"].u8();
    assert_eq!(
        active,
        sim.buffer_active()[..56]
            .iter()
            .map(|&b| u8::from(b))
            .collect::<Vec<_>>()
    );
    assert_eq!(active.iter().filter(|&&b| b == 1).count(), 44);

    // Stream: every received packet with n < 80 S in exactly one (l, m).
    let iq = a["csi_iq"].i8();
    let mask = a["csi_mask"].u8();
    let rssi = a["rssi"].i8();
    let xy = a["xy"].f32();
    let walking = a["walking"].u8();
    let posture = a["posture"].u8();
    let zone = a["zone"].i8();
    let person_mask = a["person_mask"].u8();
    let (w, d) = (scene.room.w_m, scene.room.d_m);
    let mut received = 0;
    let mut labelled = 0;
    for tr in sim.run() {
        let nn = tr.m as usize * n + tr.slot;
        if nn >= n_store {
            break;
        }
        // `% == 0`, not `is_multiple_of`: the workspace's rust-version is 1.80.
        #[allow(clippy::manual_is_multiple_of)]
        if (nn + 1) % 40 == 0 {
            // Labels: Room Sim's truth at transmission 40(k+1) - 1.
            let k = (nn + 1) / 40 - 1;
            labelled += 1;
            for (j, p) in tr.truth.people.iter().enumerate() {
                let at = k * 3 + j;
                assert_eq!((xy[2 * at], xy[2 * at + 1]), (p.x as f32, p.y as f32));
                assert_eq!(walking[at], u8::from(p.walking));
                let code = match p.posture {
                    Posture::Stand => 0,
                    Posture::Sit => 1,
                    Posture::Lie => 2,
                };
                assert_eq!(posture[at], code);
                assert_eq!(person_mask[at], 1);
                let col = ((3.0 * p.x / w).floor() as i64).clamp(0, 2);
                let row = ((3.0 * p.y / d).floor() as i64).clamp(0, 2);
                assert_eq!(i64::from(zone[at]), 3 * row + col);
            }
            for j in tr.truth.people.len()..3 {
                let at = k * 3 + j;
                assert!(
                    xy[2 * at].is_nan() && xy[2 * at + 1].is_nan(),
                    "absent people are NaN"
                );
                assert_eq!(
                    (person_mask[at], posture[at], zone[at], walking[at]),
                    (0, 255, -1, 0)
                );
            }
        }
        for p in &tr.packets {
            let cell = p.link_index * columns + tr.m as usize;
            match &p.received {
                None => assert_eq!(mask[cell], 0),
                Some(r) => {
                    received += 1;
                    assert_eq!(mask[cell], 1);
                    assert_eq!(i32::from(rssi[cell]), r.rssi);
                    let Buffer::Int8(entries) = &r.buffer else {
                        panic!("quantized")
                    };
                    for (s, &(re, im)) in entries[..56].iter().enumerate() {
                        let at = ((p.link_index * 56 + s) * columns + tr.m as usize) * 2;
                        assert_eq!(
                            (iq[at], iq[at + 1]),
                            (im, re),
                            "(imag, real) as the datagram"
                        );
                    }
                    // The datagram carries the same pairs.
                    let dg = r.datagram.as_ref().unwrap();
                    let at = ((p.link_index * 56) * columns + tr.m as usize) * 2;
                    assert_eq!((dg[20] as i8, dg[21] as i8), (iq[at], iq[at + 1]));
                }
            }
        }
    }
    assert_eq!(labelled, k_labels);
    // Masks consistent: as many cells set as packets received; masked
    // cells hold zeros.
    assert_eq!(mask.iter().filter(|&&b| b == 1).count(), received);
    for (cell, &mk) in mask.iter().enumerate() {
        if mk == 0 {
            assert_eq!(rssi[cell], 0);
            let (li, m) = (cell / columns, cell % columns);
            for s in 0..56 {
                let at = ((li * 56 + s) * columns + m) * 2;
                assert_eq!((iq[at], iq[at + 1]), (0, 0));
            }
        }
    }

    // Labels per tick.
    let p = scene.people.len();
    assert_eq!(a["count"].u8(), vec![p as u8; k_labels]);
    assert_eq!(a["presence"].u8(), vec![u8::from(p > 0); k_labels]);
    let t_label = a["t_label"].f64();
    let occupied = a["zone_occupied"].u8();
    let single = a["zone_single"].i8();
    for k in 0..k_labels {
        assert_eq!(t_label[k], 0.5 * (k + 1) as f64);
        for z in 0..9 {
            let held = (0..p).any(|j| zone[k * 3 + j] == z as i8);
            assert_eq!(occupied[k * 9 + z], u8::from(held));
        }
        assert_eq!(single[k], if p == 1 { zone[k * 3] } else { -1 });
    }
    // Every cell a person is labelled in is reachable (§5).
    let reachable = a["zone_reachable"].u8();
    for (at, &z) in zone.iter().enumerate() {
        if z >= 0 {
            assert_eq!(
                reachable[z as usize], 1,
                "label {at}: zone {z} is not reachable"
            );
        }
    }
    let frac = a["zone_floor_frac"].f32();
    assert_eq!(
        a["zone_reachable"].u8(),
        frac.iter().map(|&f| u8::from(f > 0.0)).collect::<Vec<_>>()
    );
    if scene.room.polygon.is_none() {
        assert_eq!(frac, vec![1.0; 9]);
    }

    // Baseline: run_roomsim's ticks on the label grid; k = 0 warming up.
    let run = run_roomsim(scene, rs, &LocateConfig::default(), "SIM").unwrap();
    assert_eq!(run.ticks.len() + 1, k_labels);
    for (key_xy, key_r, key_reason, pick) in [
        ("wlm_xy", "wlm_radius", "wlm_reason", 0),
        ("rti_xy", "rti_radius", "rti_reason", 1),
    ] {
        let bxy = a[key_xy].f32();
        let br = a[key_r].f32();
        let reason = a[key_reason].u8();
        assert_eq!(reason[0], 4, "warming_up");
        assert!(bxy[0].is_nan() && bxy[1].is_nan() && br[0].is_nan());
        for (j, tick) in run.ticks.iter().enumerate() {
            let k = j + 1;
            assert_eq!(tick.t, 0.5 * (k + 1) as f64);
            let o = if pick == 0 { &tick.wlm } else { &tick.rti };
            let code = match o {
                Outcome::Estimate(_) => 0,
                Outcome::Abstain(Reason::NoDisturbance) => 1,
                Outcome::Abstain(Reason::NotSignificant) => 2,
                Outcome::Abstain(Reason::ImplausibleRadius) => 3,
                Outcome::Abstain(r) => panic!("{r}"),
            };
            assert_eq!(reason[k], code);
            match o.estimate() {
                Some(e) => {
                    assert_eq!(
                        (bxy[2 * k], bxy[2 * k + 1], br[k]),
                        (e.x as f32, e.y as f32, e.radius_m as f32)
                    );
                }
                None => assert!(bxy[2 * k].is_nan() && bxy[2 * k + 1].is_nan() && br[k].is_nan()),
            }
        }
    }
    assert_eq!(a["calib_baseline"].f64(), run.calibration.baseline);
    assert_eq!(a["calib_null_sigma"].f64(), run.calibration.null_sigma);
    assert_eq!(
        a["calib_mask"].u8(),
        run.calibration
            .calibrated_mask
            .iter()
            .map(|&b| u8::from(b))
            .collect::<Vec<_>>()
    );

    // The calibration packets are the ones Locate counted: the calibration
    // run's received packets with t >= duration_s, on the §4.1 grid.
    let meta: Value = serde_json::from_slice(&a["meta_json"].u8()).unwrap();
    let m0 = meta["calib_m0"].as_u64().unwrap() as usize;
    let cmask = a["calib_mask_m"].u8();
    let mc = a["calib_mask_m"].shape[1];
    assert_eq!(
        cmask.iter().filter(|&&b| b == 1).count(),
        run.calibration.packets
    );
    let ciq = a["calib_iq"].i8();
    let crssi = a["calib_rssi"].i8();
    let csim = Simulator::new(scene, calibration_config(rs, 15.0)).unwrap();
    let mut first = None;
    let mut seen = 0;
    for tr in csim.run() {
        if tr.t < rs.duration_s {
            continue;
        }
        first.get_or_insert(tr.m as usize);
        for p in &tr.packets {
            let cell = p.link_index * mc + (tr.m as usize - m0);
            match &p.received {
                None => assert_eq!(cmask[cell], 0),
                Some(r) => {
                    seen += 1;
                    assert_eq!(cmask[cell], 1);
                    assert_eq!(i32::from(crssi[cell]), r.rssi);
                    let Buffer::Int8(entries) = &r.buffer else {
                        panic!("quantized")
                    };
                    let at = ((p.link_index * 56 + 7) * mc + (tr.m as usize - m0)) * 2;
                    assert_eq!((ciq[at], ciq[at + 1]), (entries[7].1, entries[7].0));
                }
            }
        }
    }
    assert_eq!(
        first,
        Some(m0),
        "calib_m0 is the first calibration superframe"
    );
    assert_eq!(seen, run.calibration.packets);

    // §7.2's tail.
    let lk = a["links"].u8();
    for (li, link) in links.iter().enumerate() {
        assert_eq!((lk[2 * li], lk[2 * li + 1]), (link.tx, link.rx));
    }
    assert_eq!(a["node_ids"].u8(), (0..n as u8).collect::<Vec<_>>());
    let pos = a["node_pos"].f32();
    for (j, node) in scene.nodes.iter().enumerate() {
        assert_eq!(
            &pos[3 * j..3 * j + 3],
            &[node.pos.0 as f32, node.pos.1 as f32, node.pos.2 as f32]
        );
    }
    assert_eq!(
        a["room_size"].f32(),
        vec![w as f32, d as f32, scene.room.h_m as f32]
    );
    let fp = scene.footprint();
    assert_eq!(a["footprint"].shape, vec![fp.len(), 2]);
    assert_eq!(
        a["window_end_tick"].i32(),
        (0..seconds / 2)
            .map(|w| 4 * w as i32 + 3)
            .collect::<Vec<_>>()
    );
    assert_eq!(meta["label"], "SIM");
    assert_eq!(meta["spec"], "dataset v1");
    assert_eq!(meta["scene_index"], index);
    assert_eq!(meta["scene_id"], format!("gen-{index:06}"));
    assert_eq!(meta["scene_sha256"], sha256_hex(scene_bytes));
    assert_eq!(meta["roomsim_seed"], rs.seed);
    assert_eq!(meta["n_nodes"], n);
    assert_eq!(meta["n_people"], p);
    assert_eq!(meta["slot_s"], 0.0125);
    assert_eq!(meta["n_tx"], n);
    assert_eq!(meta["columns"], columns);
    assert_eq!(meta["impair"], rs.impair);
    assert_eq!(meta.as_object().unwrap().len(), 13);
}

// ---------------------------------------------------------------------
// §11 Determinism, --no-impair, --from; §7.3 splits
// ---------------------------------------------------------------------

/// §7.3's splits rebuilt here from §9's split draw: by `(n_people,
/// l_shape)`, each stratum in the order of `uniform()` of
/// `Rng::stream(seed, (1 << 40) | i)` (ties by index), `round(0.1 n)`
/// half to even to test (at least 1 from 3 scenes), as many to val.
#[test]
fn the_splits_are_the_stratified_split_draws() {
    let dir = temp_dir("splits");
    let seed = 9u64;
    generate(
        &dir,
        &[
            "--scenes",
            "60",
            "--seconds",
            "2",
            "--seed",
            "9",
            "--threads",
            "16",
        ],
    );
    let m = read_json(&dir.join("manifest.json"));
    let entries = m["scenes"].as_array().unwrap();
    let mut strata: BTreeMap<(u64, bool), Vec<(f64, usize)>> = BTreeMap::new();
    for (i, e) in entries.iter().enumerate() {
        let u = saans_sim::rng::Rng::stream(seed, (1u64 << 40) | i as u64).uniform();
        let key = (
            e["n_people"].as_u64().unwrap(),
            e["l_shape"].as_bool().unwrap(),
        );
        strata.entry(key).or_default().push((u, i));
    }
    let mut want = vec![""; entries.len()];
    for members in strata.values_mut() {
        members.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let n = members.len();
        let tenth = (0.1 * n as f64).round_ties_even() as usize;
        let test = if n >= 3 { tenth.max(1) } else { tenth };
        for (rank, &(_, i)) in members.iter().enumerate() {
            want[i] = if rank < test {
                "test"
            } else if rank < test + tenth {
                "val"
            } else {
                "train"
            };
        }
    }
    assert!(
        strata.values().any(|m| m.len() >= 3),
        "a stratum big enough to split"
    );
    for (i, e) in entries.iter().enumerate() {
        assert_eq!(e["split"], want[i], "scene {i}");
    }
    for split in ["train", "val", "test"] {
        let listed: Vec<usize> = m["splits"][split]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let expected: Vec<usize> = (0..entries.len()).filter(|&i| want[i] == split).collect();
        assert_eq!(listed, expected, "{split}");
        assert_eq!(m["split_counts"][split]["n_scenes"], expected.len());
    }
    assert!(!m["splits"]["test"].as_array().unwrap().is_empty());
    // Every labelled cell is reachable, in every scene (§5), and no floor
    // fraction is a floating-point residue.
    for i in 0..entries.len() {
        let a = shard(&dir, i);
        let reachable = a["zone_reachable"].u8();
        for &z in a["zone"].i8().iter().filter(|&&z| z >= 0) {
            assert_eq!(reachable[z as usize], 1, "scene {i}: zone {z}");
        }
        for f in a["zone_floor_frac"].f32() {
            assert!(
                f == 0.0 || f == 1.0 || (1e-3..=0.999).contains(&f),
                "scene {i}: {f}"
            );
        }
    }
}

#[test]
fn one_and_four_threads_write_the_same_bytes() {
    let one = temp_dir("threads1");
    let four = temp_dir("threads4");
    let args = ["--scenes", "5", "--seconds", "2", "--seed", "3"];
    generate(&one, &[&args[..], &["--threads", "1"]].concat());
    generate(&four, &[&args[..], &["--threads", "4"]].concat());
    for sub in ["scenes", "shards"] {
        let (a, b) = (files(&one, sub), files(&four, sub));
        assert_eq!(
            a.len(),
            if sub == "scenes" { 15 } else { 5 },
            "no temporary file is left"
        );
        assert_eq!(a, b, "{sub}/ differs between 1 and 4 threads");
    }
    assert_eq!(
        manifest_without(&one, &["totals"]),
        manifest_without(&four, &["totals"])
    );
}

#[test]
fn no_impair_changes_only_the_radio_and_from_reruns_the_same_scenes() {
    let on = temp_dir("impair-on");
    let off = temp_dir("impair-off");
    let from_off = temp_dir("from-off");
    let from_on = temp_dir("from-on");
    let back_on = temp_dir("back-on");
    let args = ["--scenes", "3", "--seconds", "2", "--seed", "5"];
    generate(&on, &args);
    let out = generate(&off, &[&args[..], &["--no-impair"]].concat());
    assert!(stdout(&out).contains("impairments hardware_off"));
    assert!(stdout(&out).contains("baseline_valid true"));

    // The scene files, Locate's configs and every label are the
    // impairments-on dataset's; the Room Sim configs differ in the eight
    // hardware switches only, Room Sim's own `impair` true in both.
    let (son, soff) = (files(&on, "scenes"), files(&off, "scenes"));
    assert_eq!(
        son.keys().collect::<Vec<_>>(),
        soff.keys().collect::<Vec<_>>()
    );
    let hardware_off = serde_json::json!({
        "loss_prob": 0.0,
        "loss_prob_link": {},
        "agc_jump_prob": 0.0,
        "device_gain_db": [0.0, 0.0],
        "ripple": false,
        "cpe": false,
        "timing_offset_s": 0.0,
        "timing_offset_dev_s": [0.0, 0.0],
    });
    for (name, bytes) in &son {
        if name.ends_with(".roomsim.json") {
            let mut a: Value = serde_json::from_slice(bytes).unwrap();
            let mut b: Value = serde_json::from_slice(&soff[name]).unwrap();
            assert_eq!(
                (&a["impair"], &b["impair"]),
                (&Value::Bool(true), &Value::Bool(true))
            );
            let defaults = serde_json::to_value(RoomSimConfig::default()).unwrap();
            for key in HARDWARE {
                assert_eq!(a[key], defaults[key], "{name}: {key} on");
                assert_eq!(b[key], hardware_off[key], "{name}: {key} off");
                a.as_object_mut().unwrap().remove(key);
                b.as_object_mut().unwrap().remove(key);
            }
            assert_eq!(a, b, "{name}");
            // Noise kept.
            for key in [
                "snr_noise",
                "chip_noise_counts",
                "noise_floor_dbm",
                "rssi_jitter_db",
                "quantize",
                "ltf_merge",
            ] {
                assert_eq!(b[key], defaults[key], "{name}: {key}");
            }
        } else {
            assert_eq!(bytes, &soff[name], "{name}");
        }
    }
    let (mon, moff) = (
        read_json(&on.join("manifest.json")),
        read_json(&off.join("manifest.json")),
    );
    assert_eq!(
        (&mon["impair"], &mon["impairments"]),
        (&Value::Bool(true), &Value::from("on"))
    );
    assert_eq!(
        (&moff["impair"], &moff["impairments"]),
        (&Value::Bool(true), &Value::from("hardware_off"))
    );
    let mut off_rs = RoomSimConfig::default();
    for (key, v) in hardware_off.as_object().unwrap() {
        let mut all = serde_json::to_value(&off_rs).unwrap();
        all[key] = v.clone();
        off_rs = serde_json::from_value(all).unwrap();
    }
    assert_eq!(moff["impairment_switches"], switches(&off_rs));
    assert_eq!(mon["splits"], moff["splits"]);
    for (dir, m) in [(&on, &mon), (&off, &moff)] {
        let valid = (0..3).all(|i| {
            shard(dir, i)["calib_null_sigma"]
                .f64()
                .iter()
                .any(|&s| s > 0.0)
        });
        assert_eq!(m["baseline_valid"], valid);
        assert!(valid, "RSSI jitter keeps a null sigma above 0");
    }
    for i in 0..3 {
        let (a, b) = (shard(&on, i), shard(&off, i));
        for name in LABEL_MEMBERS {
            assert_eq!(a[name], b[name], "scene {i}: {name}");
        }
        assert_ne!(a["csi_iq"], b["csi_iq"], "scene {i}: the radio differs");
        // No packet is lost with the hardware off.
        let n = a["node_ids"].len();
        let columns = a["csi_mask"].shape[1];
        let mask = b["csi_mask"].u8();
        for (cell, &mk) in mask.iter().enumerate() {
            let (li, m) = (cell / columns, cell % columns);
            let n_tx = m * n + usize::from(b["tx_slot"].u8()[li]);
            assert_eq!(mk, u8::from(n_tx < 160), "scene {i}: link {li}, column {m}");
        }
        let meta: Value = serde_json::from_slice(&b["meta_json"].u8()).unwrap();
        assert_eq!(meta["impair"], true, "Room Sim's master switch stays on");
    }

    // --from reruns DIR's scenes: with --no-impair it is the
    // impairments-off dataset of the same seed, byte for byte.
    let src = on.to_str().unwrap();
    let out = generate(&from_off, &["--no-impair", "--from", src, "--threads", "2"]);
    assert!(stdout(&out).contains("scenes from"));
    for sub in ["scenes", "shards"] {
        assert_eq!(files(&from_off, sub), files(&off, sub), "{sub}/");
    }
    let skip = ["totals", "from", "command"];
    assert_eq!(
        manifest_without(&from_off, &skip),
        manifest_without(&off, &skip)
    );
    let m = read_json(&from_off.join("manifest.json"));
    assert_eq!(
        m["from"],
        std::fs::canonicalize(&on).unwrap().display().to_string()
    );
    // And without it, the source dataset again (the same --scenes is
    // accepted).
    generate(&from_on, &["--from", src, "--scenes", "3"]);
    assert_eq!(files(&from_on, "shards"), files(&on, "shards"));
    assert_eq!(files(&from_on, "scenes"), files(&on, "scenes"));
    // From the hardware-off copy without --no-impair: the impairments are
    // on again, the source dataset byte for byte (the checks and the run
    // use the ON profile, whatever DIR's configs say).
    generate(&back_on, &["--from", off.to_str().unwrap()]);
    assert_eq!(files(&back_on, "shards"), files(&on, "shards"));
    assert_eq!(files(&back_on, "scenes"), files(&on, "scenes"));

    // A different --seed, --seconds or --scenes than DIR's is refused.
    for (flag, value) in [("--seed", "6"), ("--seconds", "4"), ("--scenes", "2")] {
        let dir = temp_dir(&format!("from-refused-{}", &flag[2..]));
        let o = saans(
            &[
                "gen-dataset",
                "--out",
                dir.to_str().unwrap(),
                "--from",
                src,
                flag,
                value,
            ],
            &[],
        );
        assert_eq!(o.status.code(), Some(2), "{flag}: {}", stderr(&o));
        assert!(
            stderr(&o).contains(&format!("gen-dataset: {flag}: ")),
            "{}",
            stderr(&o)
        );
        assert!(!dir.exists(), "a refusal writes nothing");
    }
}

// ---------------------------------------------------------------------
// §11 CLI: refusals, --overwrite, the size check
// ---------------------------------------------------------------------

#[test]
fn bad_invocations_are_refused_naming_the_parameter() {
    let dir = temp_dir("refused");
    let d = dir.to_str().unwrap();
    for (args, param) in [
        (vec!["gen-dataset"], "--out"),
        (
            vec!["gen-dataset", "--out", d, "--seconds", "3"],
            "--seconds",
        ),
        (vec!["gen-dataset", "--out", d, "--scenes", "0"], "--scenes"),
        (vec!["gen-dataset", "--out", d, "--seed", "x"], "--seed"),
        (
            vec!["gen-dataset", "--out", d, "--threads", "0"],
            "--threads",
        ),
        (vec!["gen-dataset", "--out", d, "--bogus"], "--bogus"),
        (vec!["gen-dataset", "--out", d, "--from", d], "--from"),
        // An empty path would name the working directory's files.
        (vec!["gen-dataset", "--out", ""], "--out"),
        (vec!["gen-dataset", "--out", "", "--overwrite"], "--out"),
        (vec!["gen-dataset", "--out", d, "--from", ""], "--from"),
    ] {
        let o = saans(&args, &[]);
        assert_eq!(o.status.code(), Some(2), "{args:?}: {}", stderr(&o));
        assert!(
            stderr(&o).starts_with(&format!("gen-dataset: {param}: ")),
            "{args:?}: {}",
            stderr(&o)
        );
        assert!(!dir.exists(), "{args:?}: a refusal writes nothing");
    }
    // An --out that is a file.
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("f");
    std::fs::write(&file, b"x").unwrap();
    let o = saans(
        &[
            "gen-dataset",
            "--out",
            file.to_str().unwrap(),
            "--scenes",
            "1",
            "--seconds",
            "2",
        ],
        &[],
    );
    assert_eq!(o.status.code(), Some(2));
    assert!(
        stderr(&o).starts_with("gen-dataset: --out: "),
        "{}",
        stderr(&o)
    );
    // A --from that is a file, not a dataset directory.
    let elsewhere = temp_dir("refused-elsewhere");
    let o = saans(
        &[
            "gen-dataset",
            "--out",
            elsewhere.to_str().unwrap(),
            "--from",
            file.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(o.status.code(), Some(2));
    assert!(
        stderr(&o).starts_with("gen-dataset: --from: "),
        "{}",
        stderr(&o)
    );
    assert!(stderr(&o).contains("not a directory"), "{}", stderr(&o));
    assert!(!elsewhere.exists());
    // --from naming --out's own directory, or a dataset inside the
    // scenes/ or shards/ that --overwrite would remove.
    for sub in ["", "scenes/old", "shards/old/deeper"] {
        let from = dir.join(sub);
        std::fs::create_dir_all(&from).unwrap();
        std::fs::write(from.join("manifest.json"), b"{}").unwrap();
        let o = saans(
            &[
                "gen-dataset",
                "--out",
                d,
                "--from",
                from.to_str().unwrap(),
                "--overwrite",
            ],
            &[],
        );
        assert_eq!(o.status.code(), Some(2), "{sub}: {}", stderr(&o));
        assert!(
            stderr(&o).starts_with("gen-dataset: --from: "),
            "{sub}: {}",
            stderr(&o)
        );
        assert!(
            from.join("manifest.json").exists(),
            "{sub}: nothing removed"
        );
    }
    assert_eq!(std::fs::read(&file).unwrap(), b"x");
}

#[test]
fn a_non_empty_out_needs_overwrite_which_replaces_only_the_dataset() {
    let dir = temp_dir("overwrite");
    let d = dir.to_str().unwrap();
    generate(&dir, &["--scenes", "2", "--seconds", "2", "--seed", "1"]);
    std::fs::write(dir.join("notes.txt"), b"keep me").unwrap();
    std::fs::create_dir(dir.join("plots")).unwrap();
    std::fs::write(dir.join("plots/a.png"), b"png").unwrap();
    std::fs::write(dir.join("scenes/stale.json"), b"{}").unwrap();
    let before = std::fs::read(dir.join("manifest.json")).unwrap();

    let o = saans(
        &[
            "gen-dataset",
            "--out",
            d,
            "--scenes",
            "1",
            "--seconds",
            "2",
            "--seed",
            "2",
        ],
        &[],
    );
    assert_eq!(o.status.code(), Some(2), "{}", stderr(&o));
    assert!(
        stderr(&o).starts_with("gen-dataset: --out: "),
        "{}",
        stderr(&o)
    );
    assert_eq!(
        std::fs::read(dir.join("manifest.json")).unwrap(),
        before,
        "untouched"
    );

    generate(
        &dir,
        &[
            "--scenes",
            "1",
            "--seconds",
            "2",
            "--seed",
            "2",
            "--overwrite",
        ],
    );
    assert_eq!(std::fs::read(dir.join("notes.txt")).unwrap(), b"keep me");
    assert_eq!(std::fs::read(dir.join("plots/a.png")).unwrap(), b"png");
    assert!(
        !dir.join("scenes/stale.json").exists(),
        "scenes/ was replaced"
    );
    assert_eq!(files(&dir, "scenes").len(), 3);
    assert_eq!(files(&dir, "shards").len(), 1);
    let m = read_json(&dir.join("manifest.json"));
    assert_eq!(
        (m["seed"].as_u64(), m["n_scenes"].as_u64()),
        (Some(2), Some(1))
    );
}

#[test]
fn the_size_check_refuses_when_too_little_space_would_stay_free() {
    let dir = temp_dir("size");
    let d = dir.to_str().unwrap();
    let args = ["gen-dataset", "--out", d, "--scenes", "2", "--seconds", "2"];
    // 5 GB plus a few bytes: less than the estimate fits.
    let o = saans(&args, &[("SAANS_GEN_DATASET_FREE_BYTES", "5000001000")]);
    assert_eq!(o.status.code(), Some(2), "{}", stderr(&o));
    let err = stderr(&o);
    assert!(err.contains("SIM  gen-dataset: estimated size"), "{err}");
    assert!(
        err.contains("gen-dataset: --out: the dataset needs about"),
        "{err}"
    );
    assert!(!dir.exists(), "nothing written");
    // Plenty of room.
    let o = saans(&args, &[("SAANS_GEN_DATASET_FREE_BYTES", "100000000000")]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert!(dir.join("manifest.json").exists());
    // The estimate is close to what was written.
    let m = read_json(&dir.join("manifest.json"));
    let written = m["totals"]["bytes"].as_u64().unwrap() as f64;
    let estimate: f64 = stderr(&o)
        .lines()
        .find_map(|l| {
            l.split_once("GB (")
                .and_then(|(_, r)| r.split_once(" bytes"))
                .map(|(n, _)| n.parse().unwrap())
        })
        .unwrap();
    assert!(
        estimate >= written && estimate < 1.3 * written,
        "estimate {estimate}, written {written}"
    );
}

#[test]
fn a_from_that_is_not_a_finished_dataset_is_refused() {
    let src = temp_dir("not-a-dataset");
    std::fs::create_dir_all(src.join("scenes")).unwrap();
    let out = temp_dir("from-nothing");
    let o = saans(
        &[
            "gen-dataset",
            "--out",
            out.to_str().unwrap(),
            "--from",
            src.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(o.status.code(), Some(2), "{}", stderr(&o));
    assert!(
        stderr(&o).contains("gen-dataset: --from: ") && stderr(&o).contains("no manifest.json")
    );
    assert!(!out.exists());
}
