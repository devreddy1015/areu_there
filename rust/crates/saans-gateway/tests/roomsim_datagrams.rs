//! spec/roomsim.md §12.4 and §15: every raw-CSI datagram Room Sim emits
//! parses with the gateway's `raw.rs`, unchanged. The datagrams are the
//! golden ones (`spec/golden/roomsim_impair` and `roomsim_switches`,
//! written by the Python reference, which `saans-roomsim` reproduces byte
//! for byte in its own golden test), so this check links nothing from the
//! simulator. They are SIM data, never a capture.

use saans_gateway::raw;
use std::collections::HashMap;
use std::path::PathBuf;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../spec/golden")
}

fn load(name: &str) -> (HashMap<String, String>, impl Fn(&str) -> Vec<f64>) {
    let text = std::fs::read_to_string(golden_dir().join(format!("{name}.kv"))).unwrap();
    let kv: HashMap<String, String> = text
        .lines()
        .filter_map(|l| l.split_once(' '))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let lens = kv.clone();
    let name = name.to_string();
    let array = move |k: &str| -> Vec<f64> {
        let n: usize = lens[&format!("array.{k}")].parse().unwrap();
        let bytes = std::fs::read(golden_dir().join(format!("{name}__{k}.f64"))).unwrap();
        assert_eq!(bytes.len(), n * 8);
        bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect()
    };
    (kv, array)
}

/// Every datagram of the golden set `name` parses, and every header field
/// and payload entry is the value the reference recorded for its packet.
/// Returns the number of datagrams.
fn check_set(name: &str) -> usize {
    let (kv, a) = load(name);
    assert_eq!(kv["label"], "SIM");
    // The firmware buffer width and channel of the run (§3.2 defaults 56, 6).
    let fw: usize = kv
        .get("cfg.fw_subcarriers")
        .map_or(56, |v| v.parse().unwrap());
    let freq_mhz = (kv["carrier_hz"].parse::<f64>().unwrap() / 1e6) as u32;
    let (dgram, lens) = (a("dgram"), a("dgram_len"));
    let (lost, rx, seq, rssi, nf) = (a("lost"), a("rx"), a("seq"), a("rssi"), a("noise_floor"));
    let (buf_re, buf_im) = (a("buffer_re"), a("buffer_im"));
    let received: Vec<usize> = (0..lost.len()).filter(|&i| lost[i] == 0.0).collect();
    assert_eq!(
        received.len(),
        lens.len(),
        "{name}: one datagram per received packet"
    );
    let mut off = 0;
    for (n, &i) in received.iter().enumerate() {
        let len = lens[n] as usize;
        assert_eq!(len, 20 + 2 * fw, "{name}: datagram {n} length");
        let bytes: Vec<u8> = dgram[off..off + len].iter().map(|&b| b as u8).collect();
        off += len;
        assert!(raw::sniff(&bytes));
        let f = raw::parse(&bytes).unwrap_or_else(|e| panic!("{name}: datagram {n}: {e}"));
        assert_eq!(f.node_id as f64, rx[i]);
        assert_eq!(f.n_antennas, 1);
        assert_eq!(f.n_subcarriers as usize, fw, "fw_subcarriers");
        assert_eq!(f.freq_mhz, freq_mhz);
        assert_eq!(f.sequence as f64, seq[i]);
        assert_eq!(f.rssi_dbm as f64, rssi[i]);
        assert_eq!(f.noise_floor_dbm as f64, nf[i]);
        assert_eq!((f.ppdu_type, f.flags), (0, 0));
        // The payload is the first fw buffer entries, imaginary first.
        assert_eq!(f.iq.len(), fw);
        for (e, &(im, re)) in f.iq.iter().enumerate() {
            assert_eq!(
                im as f64,
                buf_im[n * 128 + e],
                "{name}: datagram {n} entry {e}"
            );
            assert_eq!(
                re as f64,
                buf_re[n * 128 + e],
                "{name}: datagram {n} entry {e}"
            );
        }
    }
    assert_eq!(off, dgram.len());
    received.len()
}

#[test]
fn every_roomsim_datagram_parses_unchanged() {
    assert!(check_set("roomsim_impair") > 0);
}

/// `roomsim_switches`: 64 firmware subcarriers on channel 11, an rx-only
/// node 7, a link that loses every packet, smoothing on.
#[test]
fn every_roomsim_switches_datagram_parses_unchanged() {
    assert_eq!(check_set("roomsim_switches"), 211);
}
