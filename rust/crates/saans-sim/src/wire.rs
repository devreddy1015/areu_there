//! Raw-CSI datagrams: a simulated node putting bytes on the wire the way
//! a real one does.
//!
//! The format is not invented here, and nothing in this module may change
//! it. It is `spec/wire.md` section 1; it is written by `serialize_raw`
//! in `edge/firmware/main/csi_capture.cpp`; and it is read by
//! `saans-gateway/src/raw.rs`, itself a byte-exact port of the Python
//! reference's `wifisense/sources/esp32_udp.py`. This file is a fourth
//! writer of the same twenty bytes, and it exists only because the
//! dependency runs gateway -> sim: the simulator cannot call the
//! gateway's encoder. The tests on both sides pin the two together —
//! `tests` below against the documented offsets, and the gateway's
//! `tests/sim_raw.rs` against its own parser and its own encoder.
//!
//! # Why bytes and not floats
//!
//! Because then ONE parser serves simulated and real data. A gateway fed
//! from here runs the same `raw::parse`, the same `FeatureExtractor` and
//! the same fusion that a bench node will run, so a defect anywhere along
//! that path can be found before the hardware exists instead of during
//! the first bench session. Handing the gateway floats would exercise a
//! path no node will ever use.
//!
//! What this puts in those bytes is the output of [`crate::impair`],
//! whose realism is UNVERIFIED — see that module. The shape of the
//! datagram is right by construction; whether the numbers inside it look
//! like an ESP32's is exactly what nobody on this project has checked.

use crate::impair::{ImpairConfig, NodeImpairment};
use crate::scenario::{Scenario, SimOptions};
use crate::sim::Tick;
use saans_dsp::CsiFrame;
use std::collections::BTreeMap;

/// First four bytes of a raw-CSI datagram, little-endian on the wire.
/// `kRawMagic` in `csi_capture.h`, `CSI_MAGIC` in the gateway's `raw.rs`.
pub const RAW_MAGIC: u32 = 0xC511_0001;

/// Header bytes before the I/Q payload. `kRawHeaderSize`.
pub const RAW_HEADER_SIZE: usize = 20;

/// Antennas per datagram. The firmware writes 1 unconditionally — an
/// ESP32-S3 is 1T1R — and so does this, rather than offering a count no
/// node can produce.
pub const RAW_ANTENNAS: u8 = 1;

/// The format's "no noise floor measurement", and what a simulated node
/// reports by default.
///
/// A real node fills this field from `rx_ctrl.noise_floor`: a measurement
/// made by the radio, not derived from CSI. The simulator has nothing
/// equivalent. Its two noise terms are the scenario's channel noise, in
/// the arbitrary units the clean channel is expressed in, and the
/// receiver noise in [`crate::impair`], in converter counts; turning
/// either into dBm needs exactly the absolute calibration an AGC
/// destroys. Inventing a number here would be inventing a measurement, so
/// the field says "unknown", which the gateway already handles (`engine.rs`
/// only forwards a noise floor when it is not `i8::MIN`).
pub const NOISE_FLOOR_UNKNOWN: i8 = i8::MIN;

/// Centre frequency in MHz for the raw header, from a carrier in Hz.
///
/// The firmware gets this from `channel_to_mhz(rx_ctrl.channel)`; the
/// simulator has no channel number, only [`SimOptions::carrier_hz`]
/// (2.437 GHz by default, which is channel 6).
pub fn freq_mhz(carrier_hz: f64) -> u32 {
    (carrier_hz / 1e6).round().clamp(0.0, u32::MAX as f64) as u32
}

/// Write one raw-CSI datagram into `buf`, replacing its contents.
///
/// A transcription of the firmware's `serialize_raw`, field by field at
/// the documented offsets. Two differences, both because this is Rust and
/// neither visible on the wire: the payload length travels with the
/// slice instead of as `iq_len`, and the buffer grows instead of
/// returning 0 when it would not fit.
///
/// `iq` is already in wire order — `iq[2k]` imaginary, `iq[2k + 1]` real
/// — which is what [`NodeImpairment::apply_to_iq`] produces.
///
/// # Panics
/// If `iq` has an odd length (half a complex pair is a caller bug, not a
/// datagram), or if the subcarrier count would not fit the u16 field.
pub fn serialize_raw(
    iq: &[i8],
    node_id: u8,
    sequence: u32,
    rssi_dbm: i8,
    noise_floor_dbm: i8,
    freq_mhz: u32,
    buf: &mut Vec<u8>,
) {
    assert!(
        iq.len().is_multiple_of(2),
        "iq has {} bytes: not whole complex pairs",
        iq.len()
    );
    let n_subcarriers = iq.len() / 2;
    assert!(
        n_subcarriers <= u16::MAX as usize,
        "{n_subcarriers} subcarriers does not fit the u16 field"
    );

    buf.clear();
    buf.reserve(RAW_HEADER_SIZE + iq.len());
    // Little-endian throughout: the firmware writes these fields with
    // memcpy on a little-endian part, and the parser reads them with
    // from_le_bytes. Spelling it out here keeps this file honest on a
    // machine of either endianness.
    buf.extend_from_slice(&RAW_MAGIC.to_le_bytes()); // 0..4
    buf.push(node_id); // 4
    buf.push(RAW_ANTENNAS); // 5
    buf.extend_from_slice(&(n_subcarriers as u16).to_le_bytes()); // 6..8
    buf.extend_from_slice(&freq_mhz.to_le_bytes()); // 8..12
    buf.extend_from_slice(&sequence.to_le_bytes()); // 12..16
    buf.push(rssi_dbm as u8); // 16
    buf.push(noise_floor_dbm as u8); // 17
    buf.push(0); // 18: PPDU type, unused by the firmware
    buf.push(0); // 19: flags, unused by the firmware
    buf.extend(iq.iter().map(|&v| v as u8)); // 20..
}

/// One simulated node's raw-CSI output: a receiver model
/// ([`NodeImpairment`]) and the datagram counter in front of it.
///
/// A node, not a link. Two links into the same receiving node are one
/// radio, so they share one AGC, one static ripple and one sequence
/// counter — which is also how they arrive at the gateway, because a raw
/// datagram does not say who transmitted the frame and the gateway keys
/// every one of them on `(rx, 0xFF)`.
#[derive(Debug, Clone)]
pub struct RawNode {
    node_id: u8,
    freq_mhz: u32,
    /// Reported noise floor. [`NOISE_FLOOR_UNKNOWN`] unless a caller has
    /// a better idea; see that constant for why nothing here does.
    pub noise_floor_dbm: i8,
    sequence: u32,
    imp: NodeImpairment,
    re: Vec<f64>,
    im: Vec<f64>,
    iq: Vec<i8>,
}

impl RawNode {
    /// A node with its own receiver drawn from `(seed, node_id, boot)`,
    /// reporting `n` subcarriers at `freq_mhz`.
    pub fn new(
        cfg: ImpairConfig,
        n: usize,
        seed: u64,
        node_id: u8,
        boot: u32,
        freq_mhz: u32,
    ) -> Self {
        Self::with_impairment(
            NodeImpairment::new(cfg, n, seed, node_id, boot),
            node_id,
            freq_mhz,
        )
    }

    /// [`RawNode::new`] with the receiver model built by the caller —
    /// for a non-default [`crate::impair::BinLayout`], say.
    pub fn with_impairment(imp: NodeImpairment, node_id: u8, freq_mhz: u32) -> Self {
        let n = imp.amp_ripple().len();
        Self {
            node_id,
            freq_mhz,
            noise_floor_dbm: NOISE_FLOOR_UNKNOWN,
            sequence: 0,
            imp,
            re: vec![0.0; n],
            im: vec![0.0; n],
            iq: vec![0; 2 * n],
        }
    }

    /// Impair one clean frame and write the datagram into `buf`.
    ///
    /// # Panics
    /// If `frame` does not have the subcarrier count this node was built
    /// for: a node reports what its radio reports, and silently padding
    /// or truncating would put a fabricated subcarrier on the wire.
    pub fn encode(&mut self, frame: &CsiFrame, buf: &mut Vec<u8>) {
        let n = self.re.len();
        assert_eq!(
            frame.amplitudes.len(),
            n,
            "node {} takes {n}-subcarrier frames",
            self.node_id
        );
        assert_eq!(
            frame.phases.len(),
            n,
            "node {} takes {n}-subcarrier frames",
            self.node_id
        );

        // Back to rectangular. The simulator's public frame is polar
        // because that is what the reference and the firmware's capture
        // callback both produce, but a receiver acts on the complex
        // channel — a phase rotation is not an operation on an angle in
        // isolation. The clean path keeps its own (re, im) and is not
        // touched by this; the round trip through hypot/atan2 and back
        // costs a couple of ulp, which is three orders of magnitude
        // below one int8 count.
        for k in 0..n {
            let (sin, cos) = frame.phases[k].sin_cos();
            self.re[k] = frame.amplitudes[k] * cos;
            self.im[k] = frame.amplitudes[k] * sin;
        }
        self.imp
            .apply_to_iq(&mut self.re, &mut self.im, &mut self.iq);

        // RSSI comes from the CLEAN channel, deliberately. A node reports
        // it from the radio's power detector, which is a separate
        // measurement from CSI and keeps its absolute meaning after the
        // AGC has taken the absolute meaning out of the CSI amplitude.
        // That is most of why the field is worth having at all. The
        // clamp is a type conversion, not a model: the simulator's dBm
        // is an i32 and the field is an i8.
        let rssi = frame.rssi_dbm.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
        serialize_raw(
            &self.iq,
            self.node_id,
            self.sequence,
            rssi,
            self.noise_floor_dbm,
            self.freq_mhz,
            buf,
        );
        self.sequence = self.sequence.wrapping_add(1);
    }

    /// [`RawNode::encode`] into a fresh `Vec`.
    pub fn datagram(&mut self, frame: &CsiFrame) -> Vec<u8> {
        let mut buf = Vec::new();
        self.encode(frame, &mut buf);
        buf
    }

    /// The last datagram's I/Q in wire order: `iq[2k]` imaginary,
    /// `iq[2k + 1]` real. All zeros before the first [`RawNode::encode`].
    pub fn iq(&self) -> &[i8] {
        &self.iq
    }

    /// The receiver model, for its AGC gain and static ripple.
    pub fn impairment(&self) -> &NodeImpairment {
        &self.imp
    }

    pub fn node_id(&self) -> u8 {
        self.node_id
    }

    pub fn freq_mhz(&self) -> u32 {
        self.freq_mhz
    }

    /// Sequence number the next datagram will carry.
    ///
    /// This is the firmware's `s_sequence`, which counts frames the radio
    /// delivered to the capture callback — not frames the transmitter
    /// sent. A packet lost in the air never reaches the callback, so it
    /// never consumes a sequence number and nothing downstream can see
    /// that it was lost. (On a node a gap does appear when the capture
    /// queue overflows, because the number is stamped at capture and the
    /// frame is dropped afterwards; the simulator has no queue, so this
    /// counter is simply dense.) The scenario's `loss_prob` is therefore
    /// invisible on this path, exactly as air loss is invisible to a real
    /// node — unlike `simulate.rs`'s FeatureFrame bridge, which passes
    /// the simulator's transmit-side counter and so does show it.
    pub fn sequence(&self) -> u32 {
        self.sequence
    }
}

/// The receiving nodes of one scenario, each with its own receiver.
///
/// Built from the scenario's links, so a node that receives two links
/// gets one [`RawNode`] and not two.
#[derive(Debug, Clone)]
pub struct RawNodes {
    nodes: BTreeMap<u8, RawNode>,
    buf: Vec<u8>,
}

impl RawNodes {
    /// One node per distinct receiver in `sc.links`, all with impairment
    /// config `cfg` and power cycle `boot`, seeded from the scenario.
    pub fn new(sc: &Scenario, opts: &SimOptions, cfg: ImpairConfig, boot: u32) -> Self {
        let f = freq_mhz(opts.carrier_hz);
        let mut nodes = BTreeMap::new();
        for &(_tx, rx) in &sc.links {
            nodes
                .entry(rx)
                .or_insert_with(|| RawNode::new(cfg, sc.n_subcarriers, sc.seed, rx, boot, f));
        }
        Self {
            nodes,
            buf: Vec::new(),
        }
    }

    /// Datagrams for one tick, in the tick's own frame order, each with
    /// the node that sent it. Frames whose receiver is not one of the
    /// scenario's are skipped rather than given a node on the fly: a node
    /// that appears mid-run would have a receiver with no history.
    pub fn datagrams(&mut self, tick: &Tick) -> Vec<(u8, Vec<u8>)> {
        let mut out = Vec::with_capacity(tick.frames.len());
        for lf in &tick.frames {
            if let Some(node) = self.nodes.get_mut(&lf.rx) {
                node.encode(&lf.frame, &mut self.buf);
                out.push((lf.rx, self.buf.clone()));
            }
        }
        out
    }

    pub fn node(&self, rx: u8) -> Option<&RawNode> {
        self.nodes.get(&rx)
    }

    pub fn node_mut(&mut self, rx: u8) -> Option<&mut RawNode> {
        self.nodes.get_mut(&rx)
    }

    /// Receiving node ids, ascending.
    pub fn ids(&self) -> impl Iterator<Item = u8> + '_ {
        self.nodes.keys().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impair::quantize;

    fn frame(n: usize, rssi_dbm: i32) -> CsiFrame {
        let amplitudes: Vec<f64> = (0..n)
            .map(|k| 30.0 + 10.0 * (k as f64 * 0.37).sin())
            .collect();
        let phases: Vec<f64> = (0..n).map(|k| (k as f64 * 0.11).sin() * 3.0).collect();
        CsiFrame {
            amplitudes,
            phases,
            sample_index: 0,
            sample_rate_hz: 20.0,
            rssi_dbm,
            node_id: 3,
        }
    }

    /// The header, field by field at the offsets `spec/wire.md` and
    /// `csi_capture.cpp` give, read back by hand rather than by anything
    /// in this module.
    #[test]
    fn header_is_at_the_documented_offsets() {
        let iq: Vec<i8> = (0..112).map(|i| (i - 56) as i8).collect();
        let mut d = Vec::new();
        serialize_raw(&iq, 7, 0xDEAD_BEEF, -45, -92, 2437, &mut d);

        assert_eq!(d.len(), 20 + 112);
        assert_eq!(&d[0..4], &[0x01, 0x00, 0x11, 0xC5], "magic, little-endian");
        assert_eq!(d[4], 7, "node id");
        assert_eq!(d[5], 1, "antenna count");
        assert_eq!(
            u16::from_le_bytes([d[6], d[7]]),
            56,
            "subcarrier count is pairs, not bytes"
        );
        assert_eq!(
            u32::from_le_bytes(d[8..12].try_into().unwrap()),
            2437,
            "centre frequency MHz"
        );
        assert_eq!(
            u32::from_le_bytes(d[12..16].try_into().unwrap()),
            0xDEAD_BEEF,
            "sequence"
        );
        assert_eq!(d[16] as i8, -45, "RSSI");
        assert_eq!(d[17] as i8, -92, "noise floor");
        assert_eq!((d[18], d[19]), (0, 0), "PPDU type and flags");
        assert!(
            d[20..].iter().zip(&iq).all(|(&b, &v)| b as i8 == v),
            "payload copied verbatim"
        );
    }

    #[test]
    fn payload_is_imaginary_first() {
        let n = 56;
        let mut node = RawNode::new(ImpairConfig::esp32_s3(), n, 2024, 4, 0, 2437);
        let d = node.datagram(&frame(n, -50));
        // What the node reported, independently of the bytes.
        let iq = node.iq().to_vec();
        for k in 0..n {
            assert_eq!(
                d[RAW_HEADER_SIZE + 2 * k] as i8,
                iq[2 * k],
                "byte 2k is the imaginary part of bin {k}"
            );
            assert_eq!(
                d[RAW_HEADER_SIZE + 2 * k + 1] as i8,
                iq[2 * k + 1],
                "byte 2k+1 is the real part of bin {k}"
            );
        }
        // Not a symmetric accident: this frame has bins where the two
        // differ, so a reader that swapped them would see a different
        // channel and not the same one twice.
        assert!(
            (0..n).any(|k| iq[2 * k] != iq[2 * k + 1]),
            "the test frame cannot distinguish the two orders"
        );
    }

    #[test]
    fn identity_puts_the_clean_channel_on_the_wire() {
        // With no receiver model the only thing between the clean channel
        // and the bytes is the quantizer, which the wire format requires.
        let n = 56;
        let f = frame(n, -50);
        let mut node = RawNode::new(ImpairConfig::identity(), n, 1, 1, 0, 2437);
        node.datagram(&f);
        for k in 0..n {
            let (sin, cos) = f.phases[k].sin_cos();
            assert_eq!(
                node.iq()[2 * k],
                quantize(f.amplitudes[k] * sin),
                "bin {k} imaginary"
            );
            assert_eq!(
                node.iq()[2 * k + 1],
                quantize(f.amplitudes[k] * cos),
                "bin {k} real"
            );
        }
    }

    #[test]
    fn sequence_counts_datagrams_and_rssi_comes_from_the_clean_frame() {
        let n = 8;
        let mut node = RawNode::new(ImpairConfig::esp32_s3(), n, 9, 2, 0, 5180);
        assert_eq!(node.sequence(), 0);
        for seq in 0..5u32 {
            let d = node.datagram(&frame(n, -61));
            assert_eq!(u32::from_le_bytes(d[12..16].try_into().unwrap()), seq);
            assert_eq!(d[16] as i8, -61, "RSSI is the radio's, not the AGC's");
            assert_eq!(
                d[17] as i8, NOISE_FLOOR_UNKNOWN,
                "no noise floor is modelled"
            );
            assert_eq!(u32::from_le_bytes(d[8..12].try_into().unwrap()), 5180);
        }
        assert_eq!(node.sequence(), 5);
    }

    #[test]
    fn a_node_is_shared_by_the_links_that_arrive_at_it() {
        let sc = crate::scenarios::scenario(
            "shared",
            crate::scenarios::room_clear_los(),
            crate::scenarios::three_links(),
            crate::scenarios::empty(0.01),
            5,
        );
        let opts = SimOptions::default();
        let nodes = RawNodes::new(&sc, &opts, ImpairConfig::esp32_s3(), 0);
        // Links (0,1), (0,2), (1,2): two receivers, not three links.
        assert_eq!(nodes.ids().collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(nodes.node(1).unwrap().freq_mhz(), 2437);
        assert_ne!(
            nodes.node(1).unwrap().impairment().scale(),
            nodes.node(2).unwrap().impairment().scale()
        );
    }

    #[test]
    fn a_tick_becomes_one_datagram_per_frame() {
        let sc = crate::scenarios::scenario(
            "tick",
            crate::scenarios::room_clear_los(),
            crate::scenarios::two_links(),
            crate::scenarios::empty(0.01),
            11,
        );
        let mut sim = crate::Simulator::new(&sc);
        let mut nodes = RawNodes::new(&sc, sim.options(), ImpairConfig::esp32_s3(), 0);
        let tick = sim.step().unwrap();
        let out = nodes.datagrams(&tick);
        assert_eq!(out.len(), tick.frames.len());
        for ((rx, d), lf) in out.iter().zip(&tick.frames) {
            assert_eq!(*rx, lf.rx);
            assert_eq!(&d[..4], &RAW_MAGIC.to_le_bytes());
            assert_eq!(d[4], lf.rx, "the datagram names the receiving node");
            assert_eq!(u16::from_le_bytes([d[6], d[7]]) as usize, sc.n_subcarriers);
            assert_eq!(d.len(), RAW_HEADER_SIZE + 2 * sc.n_subcarriers);
        }
    }

    #[test]
    fn freq_mhz_of_the_default_carrier() {
        assert_eq!(
            freq_mhz(SimOptions::default().carrier_hz),
            2437,
            "channel 6"
        );
        assert_eq!(freq_mhz(5.18e9), 5180);
    }
}
