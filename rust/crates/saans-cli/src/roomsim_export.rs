//! `saans roomsim-export`: run Room Sim v1 (spec/roomsim.md) on a declared
//! scene and write the raw CSI datagrams its ESP32 nodes would have sent
//! in raw mode, with the truth behind them (§13).
//!
//! - `PREFIX.sim.pcap`: every received packet's raw CSI datagram (§12.4), in
//!   the `wifisense.sources.recorder` format (ns pcap, LINKTYPE_RAW,
//!   IPv4+UDP from `10.1.<rx>.1:5566` to `10.0.0.1:5566`, stamped
//!   `1_790_000_000 s + t`). The addresses and the source port are a
//!   convention, never the gateway's own or a network address (a real node
//!   sends from an ephemeral port). Byte-identical to the Python
//!   reference's `wifisense.roomsim.export_run` for the same scene and
//!   configuration. A lost packet produces no datagram. It holds the raw
//!   CSI datagrams only, not the FeatureFrames a raw-mode node also sends.
//! - `PREFIX.truth.jsonl`: one line per transmission, lost or not, with
//!   where everyone was (`people[i]`, never a scene id) and the phone.
//!   Truth never goes into the pcap.
//! - `PREFIX.manifest.json`: the scene's name, provenance and SHA-256 (of
//!   the file's bytes), every effective §3.2 value, the per-link and
//!   per-receiver rates, the links with their packet counts, whether the
//!   pcap is gateway-faithful (one transmitter, §12.4), and the SIM
//!   assumptions.
//!
//! An export refuses to start when any of the three files exists, unless
//! asked to overwrite, which first deletes the old manifest; the manifest
//! is written last, so a run that fails part-way never leaves a manifest
//! that describes other files.
//!
//! Everything written is **SIM**: a physically motivated model, not a
//! measurement. The room comes from the scene file, never from WiFi.
//!
//! The run is streamed: one transmission is in memory at a time.

use crate::pcap::RecorderPcapWriter;
use saans_roomsim::numerics::round_half_even;
use saans_roomsim::{Config, Simulator, LABEL, SPEC};
use saans_scene::{Provenance, Scene};
use serde::de::{DeserializeSeed, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, Serializer};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};

/// `1_790_000_000 s` in ns: simulated time 0 (§13), the anchor
/// `saans export` uses too.
pub const T0_NS: u64 = 1_790_000_000 * 1_000_000_000;
/// UDP port at both ends (§13): the gateway's default sensor port as the
/// destination; as the source port a convention (a real node sends from an
/// ephemeral port).
pub const PORT: u16 = 5566;
/// Every datagram's destination address (§13), a convention: never the
/// gateway's own address.
pub const GATEWAY_IP: [u8; 4] = [10, 0, 0, 1];

/// The source address of node `rx`'s datagrams, `10.1.<rx>.1` (§13): a
/// convention that is never the gateway's address or a network address.
pub fn node_ip(rx: u8) -> [u8; 4] {
    [10, 1, rx, 1]
}

/// What `saans roomsim-export` was asked to do.
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    pub scene: String,
    pub out: String,
    /// Overrides the configuration file's `seed`.
    pub seed: Option<u64>,
    /// Overrides the configuration file's `duration_s`.
    pub duration_s: Option<f64>,
    /// A JSON object of §3.2 parameters; absent keys take their defaults.
    pub config: Option<String>,
    /// Replace an earlier export at the same prefix (its manifest is
    /// deleted first) instead of refusing to start.
    pub overwrite: bool,
}

/// Counts of one link over the run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct LinkCounts {
    pub packets: u64,
    pub lost: u64,
    pub datagrams: u64,
}

/// What a finished export wrote.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    pub scene_name: String,
    pub pcap: String,
    pub truth: String,
    pub manifest: String,
    pub transmissions: u64,
    pub packets: u64,
    pub lost: u64,
    pub datagrams: u64,
    pub links: usize,
    /// Distinct per-link rates, Hz, ascending (empty with no links).
    pub rates_hz: Vec<f64>,
    /// The number of transmitters, `nT` (§10).
    pub transmitters: usize,
    /// `nT == 1`: each receiver's datagrams come from one transmitter, so
    /// the gateway, which keys raw CSI by rx alone, sees one link per
    /// receiver (§12.4).
    pub gateway_faithful: bool,
    pub assumptions: Vec<String>,
    pub seed: u64,
    pub duration_s: f64,
}

/// The `*.sim.pcap`, `*.truth.jsonl` and `*.manifest.json` paths of a prefix.
pub fn output_paths(prefix: &str) -> (String, String, String) {
    (
        format!("{prefix}.sim.pcap"),
        format!("{prefix}.truth.jsonl"),
        format!("{prefix}.manifest.json"),
    )
}

/// Python's `repr(float)`: the shortest round-trip digits, `1.0` for an
/// integral value, exponent form below 1e-4 and from 1e16 with a sign and
/// at least two exponent digits (`1e-05`, `1e+16`). Used where a value is
/// spelled inside a string that must read as the Python reference's does.
pub fn py_repr(x: f64) -> String {
    if !x.is_finite() {
        return if x.is_nan() {
            "nan".into()
        } else if x > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        };
    }
    // Rust's Debug for f64 is the shortest round-trip form, with the same
    // fixed/exponent thresholds as Python's repr; only the exponent is
    // spelled differently ("1e-5" against "1e-05").
    let s = format!("{x:?}");
    match s.split_once('e') {
        None => s,
        Some((mant, exp)) => {
            let (sign, digits) = match exp.strip_prefix('-') {
                Some(d) => ('-', d),
                None => ('+', exp),
            };
            format!("{mant}e{sign}{digits:0>2}")
        }
    }
}

fn provenance_name(p: Provenance) -> &'static str {
    match p {
        Provenance::Surveyed => "surveyed",
        Provenance::RoomplanImport => "roomplan_import",
        Provenance::HandWritten => "hand_written",
        Provenance::Generated => "generated",
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The SIM assumptions the run makes about the scene (§3.1, §13): today,
/// only that walls are implied by the footprint when the scene lists none.
/// Worded exactly as the Python reference's manifest.
fn assumptions(sim: &Simulator) -> Vec<String> {
    let room = sim.room();
    if !room.walls_implied {
        return Vec::new();
    }
    let cfg = sim.config();
    vec![format!(
        "SIM: the scene lists no walls; the {} footprint edges are walls of {}, {} m thick, full height",
        room.walls.len(),
        cfg.implied_wall_material.name(),
        py_repr(cfg.implied_wall_thickness_m)
    )]
}

// ---------------------------------------------------------------------------
// JSON records. Key order is the Python reference's.
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct PersonLine {
    name: String,
    x: f64,
    y: f64,
    heading: f64,
    walking: bool,
    posture: &'static str,
}

#[derive(Serialize)]
struct PacketLine {
    rx: u8,
    link_index: usize,
    lost: bool,
    seq: Option<u32>,
    rssi: Option<i32>,
    rssi_true: Option<f64>,
}

#[derive(Serialize)]
struct TruthLine {
    label: &'static str,
    m: u64,
    slot: usize,
    t: f64,
    tx: u8,
    people: Vec<PersonLine>,
    phone: Option<[f64; 3]>,
    packets: Vec<PacketLine>,
}

/// `{"key": rate}` in the given order (a map, so it is not sorted as text):
/// `"tx,rx"` in link order, or `"rx"` with receivers ascending.
struct Rates<'a>(&'a [(String, f64)]);

impl Serialize for Rates<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(self.0.len()))?;
        for (k, v) in self.0 {
            m.serialize_entry(k, v)?;
        }
        m.end()
    }
}

#[derive(Serialize)]
struct LinkLine {
    link_index: usize,
    tx: u8,
    rx: u8,
    per_link_rate_hz: f64,
    #[serde(flatten)]
    counts: LinkCounts,
}

#[derive(Serialize)]
struct Counts {
    transmissions: u64,
    packets: u64,
    lost: u64,
    datagrams: u64,
}

#[derive(Serialize)]
struct Manifest<'a> {
    label: &'static str,
    spec: &'static str,
    scene_name: &'a str,
    scene_sha256: &'a str,
    provenance: &'static str,
    config: &'a Config,
    per_link_rate_hz: Rates<'a>,
    links: Vec<LinkLine>,
    counts: Counts,
    receiver_rate_hz: Rates<'a>,
    gateway_faithful: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    assumptions: Vec<String>,
}

/// §13 `receiver_rate_hz`: each receiver, ascending, with the mean rate of
/// the transmissions it hears, the plain left-to-right sum over its links,
/// in link order, of their per-link rates (`0.0` for a receiver no
/// transmitter reaches).
fn receiver_rates(sim: &Simulator) -> Vec<(String, f64)> {
    let links = sim.links();
    let mut receivers = sim.receivers().to_vec();
    receivers.sort_unstable();
    receivers
        .iter()
        .map(|&rx| {
            let mut acc = 0.0;
            for l in links.iter().filter(|l| l.rx == rx) {
                acc += sim.per_link_rate_hz(l.index);
            }
            (rx.to_string(), acc)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// People and phone ids (§3.1: no `Person.id` or `Phone.id` in any output).
// ---------------------------------------------------------------------------

/// Where a value sits in a scene document, as far as the id check cares.
#[derive(Debug, Clone, Copy, PartialEq)]
enum At {
    Top,
    People,
    Person(usize),
    PersonId(usize),
    Phone,
    PhoneId,
    Other,
}

impl At {
    /// What is wrong with a value of this shape here, if anything: `people`
    /// is an array of objects, `phone` an object or null, and an `id` of
    /// either a string (or null, which serde reads as no id).
    fn problem(self, array: bool, object: bool, string: bool, null: bool) -> Option<String> {
        match self {
            Self::People if !array => Some("people: expected an array".into()),
            Self::Person(i) if !object => Some(format!("people[{i}]: expected an object")),
            Self::PersonId(i) if !(string || null) => {
                Some(format!("people[{i}].id: expected a string"))
            }
            Self::Phone if !(object || null) => Some("phone: expected an object".into()),
            Self::PhoneId if !(string || null) => Some("phone.id: expected a string".into()),
            _ => None,
        }
    }

    fn scalar(self, string: bool, null: bool) -> Option<String> {
        self.problem(false, false, string, null)
    }
}

/// Walks a scene document and records the first problem with its people's
/// or phone's ids in a sink that outlives the walk, so that a syntax, depth
/// or range error later in the text cannot discard it. It reads the text
/// itself rather than a parsed `serde_json::Value`, so a key given twice is
/// checked every time, and it skips every other value with `IgnoredAny`, as
/// the scene's derived parser does, so no subtree it does not care about
/// (however deep, or holding a number out of range) stops it before that
/// parser would stop.
struct IdWalk<'a>(At, &'a RefCell<Option<String>>);

impl IdWalk<'_> {
    /// Keep `problem` unless an earlier one was found: the first wins.
    fn record(&self, problem: Option<String>) {
        if let Some(p) = problem {
            self.1.borrow_mut().get_or_insert(p);
        }
    }

    fn scalar(&self, string: bool, null: bool) {
        self.record(self.0.scalar(string, null));
    }
}

impl<'de> DeserializeSeed<'de> for IdWalk<'_> {
    type Value = ();

    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
        d.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for IdWalk<'_> {
    type Value = ();

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a JSON value")
    }

    fn visit_bool<E>(self, _: bool) -> Result<(), E> {
        self.scalar(false, false);
        Ok(())
    }

    fn visit_i64<E>(self, _: i64) -> Result<(), E> {
        self.scalar(false, false);
        Ok(())
    }

    fn visit_u64<E>(self, _: u64) -> Result<(), E> {
        self.scalar(false, false);
        Ok(())
    }

    fn visit_f64<E>(self, _: f64) -> Result<(), E> {
        self.scalar(false, false);
        Ok(())
    }

    fn visit_str<E>(self, _: &str) -> Result<(), E> {
        self.scalar(true, false);
        Ok(())
    }

    fn visit_unit<E>(self) -> Result<(), E> {
        self.scalar(false, true);
        Ok(())
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        self.record(self.0.problem(true, false, false, false));
        if self.0 == At::People {
            let mut i = 0;
            while seq
                .next_element_seed(IdWalk(At::Person(i), self.1))?
                .is_some()
            {
                i += 1;
            }
        } else {
            while seq.next_element::<IgnoredAny>()?.is_some() {}
        }
        Ok(())
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        self.record(self.0.problem(false, true, false, false));
        while let Some(key) = map.next_key::<String>()? {
            let at = match (self.0, key.as_str()) {
                (At::Top, "people") => At::People,
                (At::Top, "phone") => At::Phone,
                (At::Person(i), "id") => At::PersonId(i),
                (At::Phone, "id") => At::PhoneId,
                _ => At::Other,
            };
            if at == At::Other {
                map.next_value::<IgnoredAny>()?;
            } else {
                map.next_value_seed(IdWalk(at, self.1))?;
            }
        }
        Ok(())
    }
}

/// The first problem with a scene's `people` and `phone` ids (§3.1), checked
/// before the scene is parsed so that serde, whose messages quote the value
/// it rejects, never sees an id it would refuse: `people` is an array of
/// objects whose `id`, if present, is a string; `phone`, if present, an
/// object whose `id`, if present, is a string. The first problem is
/// reported even when the text fails after it (a syntax error, nesting too
/// deep, a number out of range); text that fails before any problem is left
/// for the scene parser to report.
pub fn scene_id_problem(text: &str) -> Option<String> {
    let found = RefCell::new(None);
    let mut d = serde_json::Deserializer::from_str(text);
    // Whatever the walk found before an error stands; the error itself is
    // the scene parser's to report.
    let _ = IdWalk(At::Top, &found).deserialize(&mut d);
    found.into_inner()
}

/// §13: refuse to start when any of the three files exists, naming the
/// first (pcap, truth, manifest); with `overwrite`, delete the old manifest
/// first instead. A dangling symlink counts as existing.
fn prepare_outputs(paths: [&str; 3], overwrite: bool) -> Result<(), String> {
    if overwrite {
        let manifest = paths[2];
        return match std::fs::remove_file(manifest) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(format!("{manifest}: cannot delete the old manifest: {e}")),
        };
    }
    for p in paths {
        if std::fs::symlink_metadata(p).is_ok() {
            return Err(format!(
                "{p}: already exists (pass --overwrite to replace the export)"
            ));
        }
    }
    Ok(())
}

/// The configuration a run uses: the file's (or the defaults), then
/// `--seed` and `--duration`. Validated by `Simulator::new`.
fn effective_config(opts: &Options) -> Result<Config, String> {
    let mut cfg = match &opts.config {
        None => Config::default(),
        Some(path) => {
            let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
            Config::from_json_str(&text).map_err(|e| format!("{path}: {e}"))?
        }
    };
    if let Some(seed) = opts.seed {
        cfg.seed = seed;
    }
    if let Some(d) = opts.duration_s {
        cfg.duration_s = d;
    }
    Ok(cfg)
}

/// Run the export. Every file is written before this returns `Ok`.
pub fn export(opts: &Options) -> Result<Summary, String> {
    // Hash exactly the bytes that are parsed (§13 `scene_sha256`).
    let bytes = std::fs::read(&opts.scene).map_err(|e| format!("{}: {e}", opts.scene))?;
    let text =
        std::str::from_utf8(&bytes).map_err(|e| format!("{}: not UTF-8: {e}", opts.scene))?;
    if let Some(why) = scene_id_problem(text) {
        return Err(format!("{}: {why}", opts.scene));
    }
    let scene = Scene::from_json_str(&opts.scene, text)
        .map_err(|e| e.to_string().trim_end().to_string())?;
    let scene_sha256 = sha256_hex(&bytes);

    let cfg = effective_config(opts)?;
    let sim = Simulator::new(&scene, cfg).map_err(|e| format!("{}: {e}", opts.scene))?;
    let cfg = sim.config();
    if !cfg.quantize {
        return Err(
            "config `quantize`: an export needs datagrams, and quantize = false emits none".into(),
        );
    }

    let (pcap_path, truth_path, manifest_path) = output_paths(&opts.out);
    prepare_outputs([&pcap_path, &truth_path, &manifest_path], opts.overwrite)?;
    // Without --overwrite a file that appeared since the check is refused too.
    let create = |p: &str| {
        let file = if opts.overwrite {
            File::create(p)
        } else {
            OpenOptions::new().write(true).create_new(true).open(p)
        };
        file.map(BufWriter::new).map_err(|e| format!("{p}: {e}"))
    };
    let mut pcap =
        RecorderPcapWriter::new(create(&pcap_path)?).map_err(|e| format!("{pcap_path}: {e}"))?;
    let mut truth = create(&truth_path)?;

    let links = sim.links();
    let mut per_link = vec![LinkCounts::default(); links.len()];
    let mut transmissions = 0u64;
    for tr in sim.run() {
        transmissions += 1;
        let line = TruthLine {
            label: LABEL,
            m: tr.m,
            slot: tr.slot,
            t: tr.t,
            tx: tr.tx,
            people: tr
                .truth
                .people
                .iter()
                .enumerate()
                .map(|(i, p)| PersonLine {
                    name: format!("people[{i}]"),
                    x: p.x,
                    y: p.y,
                    heading: p.heading,
                    walking: p.walking,
                    posture: p.posture.name(),
                })
                .collect(),
            phone: tr.truth.phone.map(|p| [p.x, p.y, p.z]),
            packets: tr
                .packets
                .iter()
                .map(|p| PacketLine {
                    rx: p.rx,
                    link_index: p.link_index,
                    lost: p.lost,
                    seq: p.received.as_ref().map(|r| r.seq),
                    rssi: p.received.as_ref().map(|r| r.rssi),
                    rssi_true: p.received.as_ref().map(|r| r.rssi_true),
                })
                .collect(),
        };
        serde_json::to_writer(&mut truth, &line).map_err(|e| format!("{truth_path}: {e}"))?;
        truth
            .write_all(b"\n")
            .map_err(|e| format!("{truth_path}: {e}"))?;

        for p in &tr.packets {
            let c = &mut per_link[p.link_index];
            c.packets += 1;
            if p.lost {
                c.lost += 1;
            }
            let Some(datagram) = p.received.as_ref().and_then(|r| r.datagram.as_ref()) else {
                continue;
            };
            // T0 + round(t * 1e9), round half to even, as the reference.
            let ns = round_half_even(p.t * 1e9);
            if !(0.0..1.8e19).contains(&ns) {
                return Err(format!("t = {} s does not fit a pcap timestamp", p.t));
            }
            let t_ns = T0_NS
                .checked_add(ns as u64)
                .ok_or_else(|| format!("t = {} s does not fit a pcap timestamp", p.t))?;
            pcap.write(t_ns, (node_ip(p.rx), PORT), (GATEWAY_IP, PORT), datagram)
                .map_err(|e| format!("{pcap_path}: {e}"))?;
            c.datagrams += 1;
        }
    }
    pcap.flush().map_err(|e| format!("{pcap_path}: {e}"))?;
    truth.flush().map_err(|e| format!("{truth_path}: {e}"))?;

    let rates: Vec<(String, f64)> = links
        .iter()
        .map(|l| (format!("{},{}", l.tx, l.rx), sim.per_link_rate_hz(l.index)))
        .collect();
    let total = |f: fn(&LinkCounts) -> u64| per_link.iter().map(f).sum::<u64>();
    let (packets, lost, datagrams) = (
        total(|c| c.packets),
        total(|c| c.lost),
        total(|c| c.datagrams),
    );
    let assumptions = assumptions(&sim);
    let receiver_rates = receiver_rates(&sim);
    let transmitters = sim.transmitters().len();
    let gateway_faithful = transmitters == 1;
    let manifest = Manifest {
        label: LABEL,
        spec: SPEC,
        scene_name: &scene.name,
        scene_sha256: &scene_sha256,
        provenance: provenance_name(scene.provenance),
        config: cfg,
        per_link_rate_hz: Rates(&rates),
        links: links
            .iter()
            .map(|l| LinkLine {
                link_index: l.index,
                tx: l.tx,
                rx: l.rx,
                per_link_rate_hz: sim.per_link_rate_hz(l.index),
                counts: per_link[l.index],
            })
            .collect(),
        counts: Counts {
            transmissions,
            packets,
            lost,
            datagrams,
        },
        receiver_rate_hz: Rates(&receiver_rates),
        gateway_faithful,
        assumptions: assumptions.clone(),
    };
    // Last, so that a manifest always describes the files beside it.
    let mut mf = create(&manifest_path)?;
    let mut ser = serde_json::Serializer::with_formatter(
        &mut mf,
        serde_json::ser::PrettyFormatter::with_indent(b" "),
    );
    manifest
        .serialize(&mut ser)
        .map_err(|e| format!("{manifest_path}: {e}"))?;
    mf.write_all(b"\n")
        .and_then(|_| mf.flush())
        .map_err(|e| format!("{manifest_path}: {e}"))?;

    let mut rates_hz: Vec<f64> = rates.iter().map(|r| r.1).collect();
    rates_hz.sort_by(f64::total_cmp);
    rates_hz.dedup();
    Ok(Summary {
        scene_name: scene.name.clone(),
        pcap: pcap_path,
        truth: truth_path,
        manifest: manifest_path,
        transmissions,
        packets,
        lost,
        datagrams,
        links: links.len(),
        rates_hz,
        transmitters,
        gateway_faithful,
        assumptions,
        seed: cfg.seed,
        duration_s: cfg.duration_s,
    })
}

#[cfg(test)]
mod tests {
    use super::py_repr;

    #[test]
    fn floats_are_spelled_as_python_repr_spells_them() {
        for (x, want) in [
            (0.1, "0.1"),
            (1.0, "1.0"),
            (-0.0, "-0.0"),
            (0.23, "0.23"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.2345e100, "1.2345e+100"),
            (2.0 / 3.0, "0.6666666666666666"),
        ] {
            assert_eq!(py_repr(x), want, "{x}");
        }
    }
}
