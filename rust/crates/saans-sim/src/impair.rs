//! What an ESP32-S3 receiver does to a clean channel before anyone sees
//! it: automatic gain control, a random per-packet phase, a timing
//! offset that tilts phase across the band, a static per-device
//! amplitude and phase ripple, additive noise, and signed int8 I/Q.
//!
//! THE REALISM OF THIS LAYER IS UNVERIFIED. Nobody on this project has
//! recorded a real ESP32. Every constant below is a plausible guess read
//! off published descriptions of the part, not a measurement taken here,
//! and each one says so where it is defined. What would replace them
//! with measurements: one node, one quiet empty room, raw mode on, a
//! single capture of a few thousand packets with nothing moving. From
//! that recording the AGC step and how often it moves, the per-bin
//! amplitude and phase ripple, the packet-to-packet phase statistics and
//! the noise floor in counts are all directly estimable, and every
//! `uncalibrated` comment in this file becomes a number with an error
//! bar. Until then, treat a result that depends on these values as a
//! statement about the model, not about the hardware.
//!
//! # Why this exists
//!
//! The rest of the crate produces a clean complex channel. A real node
//! does not, and the sim-to-real literature says this layer — not body
//! realism — is what decides whether a simulated result survives contact
//! with hardware. The headline failure is the one to keep in mind: a
//! model trained on captures from a software radio with no AGC, then
//! evaluated on ESP32 captures of the *same cabled signal*, loses most
//! of its accuracy. The channel was identical; the receiver was not.
//!
//! # Where it sits
//!
//! This is a separate stage applied AFTER the clean channel, off by
//! default ([`ImpairConfig::identity`] is [`Default`], and it is the
//! exact identity, not an approximation of it). The clean path is the
//! contract between the C++, Python and Rust implementations and is
//! checked against the reference's golden vectors to 1e-9 relative
//! (`tests/parity.rs`); nothing here may touch it.
//!
//! # What this deliberately does not model
//!
//! - Cross-subcarrier smoothing (`channel_filter_en`) and the
//!   L-LTF/HT-LTF merge (`ltf_merge_en`). Our capture path sets both
//!   false, and says why, in `edge/firmware/main/csi_capture.cpp`.
//!   Modelling them would make the simulator less like our nodes, not
//!   more. The static ripple `A_k` here is a per-device frequency
//!   response, not an average of neighbours: no bin is ever mixed into
//!   another.
//! - The driver's `first_word_invalid` case. **Nothing in this crate
//!   models it**, and that is a real gap rather than a deferral: for an
//!   HT20 buffer the firmware does not shorten or blank anything, it
//!   advances the pointer by two complex pairs, so every entry's
//!   subcarrier index shifts by two on an affected frame. A consumer of
//!   simulated raw CSI therefore never sees the one thing the flag does
//!   to the data. How often a real node sets it is a bench question.
//! - Slow CFO drift as a separate term. At 20 Hz packets are 50 ms
//!   apart, and phase errors that far apart are reported as
//!   uncorrelated, so a drift model would buy nothing over the uniform
//!   per-packet draw already here.
//! - Any learned component (ADR-007).
//!
//! # Use
//!
//! ```
//! use saans_sim::impair::{ImpairConfig, NodeImpairment};
//! // node 3, first boot, 56 subcarriers, scenario seed 42
//! let mut node = NodeImpairment::new(ImpairConfig::esp32_s3(), 56, 42, 3, 0);
//! let mut re = vec![40.0; 56];
//! let mut im = vec![0.0; 56];
//! node.apply(&mut re, &mut im);
//! assert!(re.iter().zip(&im).all(|(r, i)| r.abs() <= 128.0 && i.abs() <= 128.0));
//! ```

use crate::rng::Rng;
use std::f64::consts::TAU;

/// Stream id for the per-node impairment draws. The scenario's other
/// streams are the room (1), the plan (2) and the links (1000+); a
/// distant base keeps adding a node or a boot from disturbing any of
/// them, the same reason the link streams are spaced out.
const STREAM_IMPAIR: u64 = 5000;

/// Largest and smallest value of one signed int8 I/Q component. The
/// range is asymmetric, as two's complement is: clipping starts half an
/// LSB sooner on the positive rail.
pub const IQ_MAX: f64 = 127.0;
pub const IQ_MIN: f64 = -128.0;

/// How the entries of a channel array map onto physical subcarrier
/// indices, which is what the timing slope and the static ripple are
/// functions of.
///
/// Source for the HT20 layout: 802.11n's 64-point FFT over a 20 MHz
/// channel. Bins -28..=-1 and 1..=28 carry data or pilots (56 of them),
/// DC is null, and -32..=-29 and 29..=31 are guard. 56 + 8 = 64.
///
/// Our capture is compiled for 56 subcarriers
/// (`CONFIG_SAANS_N_SUBCARRIERS`, "HT20 reports 56 usable subcarriers"),
/// and the scenarios here use 56 as well. The two numbers agree: 56 IS
/// the HT20 occupied count. There are then no null bins inside a 56-wide
/// array to zero — [`BinLayout::Ht20Occupied`] has none, and that is not a
/// gap in the model, it is what the layout is.
///
/// Where the two views could disagree is one level up, and it is worth
/// writing down because this file cannot settle it. By ESP-IDF's own
/// description of the CSI buffer — read from documentation, not from a
/// capture taken here — the callback gets whole FFT blocks of 64 pairs,
/// nulls included, indexed 0..=31 then -32..=-1, one block per LTF
/// field. The firmware slices the FIRST 56 pairs of that buffer
/// (`csi_capture.cpp`). Under that ordering those 56 are bins 0..=31
/// then -32..=-9: they include all 8 null bins, omit the 8 occupied
/// bins -8..=-1, and all fall inside the FIRST block, which with
/// `lltf_en` set is the legacy L-LTF (52 occupied, +/-26) rather than
/// the HT-LTF the DSP is sized for. If that description is right, the
/// firmware's 56 are not these 56 and a subcarrier index does not mean
/// the same thing on both sides. THAT IS UNRESOLVED AND UNVERIFIED
/// HERE: nobody has looked at a real buffer, which is exactly why the
/// firmware records `first_len` and counts `layout_changed` instead of
/// assuming. [`BinLayout::Ht20Fft`] exists so that, once a bench
/// session says which it is, the simulator can produce either without
/// rewriting this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinLayout {
    /// 56 entries, the occupied HT20 bins in ascending order:
    /// -28..=-1 then 1..=28. No null bins — so `zero_null_bins` has
    /// nothing to do under this layout, and the absence of zero bins in
    /// a simulated frame is NOT evidence that a node does not emit any.
    /// This is the idealised occupied set, which is the one thing the
    /// paragraph above says may not be the 56 our firmware slices.
    Ht20Occupied,
    /// 64 entries, FFT bins in the order ESP-IDF's buffer is documented
    /// to use: 0..=31 then -32..=-1. DC and the 7 guard bins are null.
    Ht20Fft,
    /// Anything else: `n` contiguous bins centred on DC, no nulls. A
    /// fallback so an odd `n_subcarriers` still gets a sane slope and
    /// ripple rather than a panic.
    Contiguous,
}

impl BinLayout {
    /// The layout that matches an array of `n` entries.
    pub fn for_n(n: usize) -> Self {
        match n {
            56 => Self::Ht20Occupied,
            64 => Self::Ht20Fft,
            _ => Self::Contiguous,
        }
    }

    /// Entries this layout requires, if it requires a particular count.
    pub fn required_len(self) -> Option<usize> {
        match self {
            Self::Ht20Occupied => Some(56),
            Self::Ht20Fft => Some(64),
            Self::Contiguous => None,
        }
    }

    /// FFT size the timing slope is referred to: a timing offset of `D`
    /// samples rotates bin `k` by `2 pi k D / N`, and `N` is the
    /// transform length, not the number of bins anyone kept.
    pub fn fft_size(self, n: usize) -> usize {
        match self {
            Self::Ht20Occupied | Self::Ht20Fft => 64,
            Self::Contiguous => n.max(1),
        }
    }

    /// Physical subcarrier index of entry `i` (signed, DC is 0).
    pub fn bin(self, i: usize, n: usize) -> i32 {
        let i = i as i32;
        match self {
            // 0..=27 -> -28..=-1, 28..=55 -> 1..=28.
            Self::Ht20Occupied => {
                if i < 28 {
                    i - 28
                } else {
                    i - 27
                }
            }
            Self::Ht20Fft => {
                if i < 32 {
                    i
                } else {
                    i - 64
                }
            }
            Self::Contiguous => i - (n / 2) as i32,
        }
    }

    /// Whether entry `i` is a null or guard bin: nothing is transmitted
    /// there, so nothing is measured there.
    pub fn is_null(self, i: usize, n: usize) -> bool {
        match self {
            Self::Ht20Occupied | Self::Contiguous => false,
            Self::Ht20Fft => {
                let k = self.bin(i, n);
                k == 0 || k.abs() >= 29
            }
        }
    }
}

/// Parameters of the receiver model. [`ImpairConfig::identity`] is the
/// default and does nothing at all; [`ImpairConfig::esp32_s3`] is a
/// plausible, UNCALIBRATED ESP32-S3.
///
/// Every field below is a guess. None of them was measured on this
/// project's hardware, because this project has no hardware yet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImpairConfig {
    /// Quantum of the AGC gain table, dB. The gain is always an integer
    /// multiple of this, so when it moves it moves by at least this
    /// much. 0 disables the AGC entirely (gain fixed at 1).
    ///
    /// UNCALIBRATED. Published descriptions of ESP32 CSI treat the gain
    /// as a coarse discrete control rather than a continuous one, and
    /// the model this implements specifies steps of at least 0.2 dB;
    /// 0.5 dB is a plausible value in that region. The number matters
    /// less than the fact that it dwarfs the signal: a chest moves |h|
    /// by well under 1%, and one 0.5 dB step is a ~6% jump.
    pub agc_step_db: f64,
    /// Error the loop tolerates before it moves, dB. Hysteresis is why
    /// the gain is piecewise constant instead of chasing every packet.
    /// UNCALIBRATED guess; any AGC has a dead zone, nobody here has
    /// measured this one.
    pub agc_deadzone_db: f64,
    /// Most quanta the gain may move in one packet, i.e. the slew rate.
    /// UNCALIBRATED guess: fast enough to recover from a boot or a big
    /// level change in a few packets, slow enough that it cannot teleport.
    pub agc_max_steps: i32,
    /// Chance per packet that the gain moves one quantum although the
    /// level is inside the dead zone. This is the "near-random walk"
    /// term: a real AGC also moves for reasons invisible in the CSI
    /// (a different transmitter, a temperature step, detector noise at
    /// a threshold). UNCALIBRATED; at 20 Hz, 0.02 is a nudge every few
    /// seconds.
    pub agc_dither_prob: f64,
    /// Bound on the gain either side of nominal, dB. UNCALIBRATED: a
    /// radio whose usable input spans roughly -95 dBm to 0 dBm needs
    /// gain range of that order, and 60 dB either side of nominal is a
    /// guess in that region. It is a real limit, not a formality: a
    /// link 60 dB below nominal comes out below the target and reads
    /// quiet, which is what a receiver at the end of its range does.
    pub agc_range_db: f64,
    /// Mean |H| across occupied bins the loop holds, in int8 counts.
    /// This is the number that makes quantization mean something — see
    /// the note on [`NodeImpairment::apply`]. UNCALIBRATED.
    pub agc_target_counts: f64,
    /// Half-width of the per-boot reporting scale, dB: `s` is drawn
    /// uniform in +/- this and persists for the life of the node.
    /// UNCALIBRATED. ESP32 CSI amplitude has no absolute calibration,
    /// and reports of the same link differing between boots are common
    /// in published descriptions; the magnitude is a guess.
    pub boot_scale_db: f64,
    /// Standard deviation of the static per-bin amplitude ripple in the
    /// middle of the band, dB. UNCALIBRATED guess of the scale of a
    /// cheap front end's frequency response.
    pub amp_ripple_db: f64,
    /// Deterministic roll-off at the edge of the band, dB, as
    /// `edge_rolloff_db * (k / k_max)^2`. UNCALIBRATED: analog and
    /// digital filters roll off before the band edge; how much is a
    /// property of this part nobody here has measured.
    pub edge_rolloff_db: f64,
    /// Factor the ripple spread is multiplied by at the band edge and
    /// within `dc_bins` of DC (1 = flat). Both ends of the model's
    /// "larger at the band edges and near DC". UNCALIBRATED guess.
    pub edge_excess: f64,
    /// How many bins either side of DC count as "near DC" for the
    /// excess ripple and the notch. UNCALIBRATED guess.
    pub dc_bins: i32,
    /// Deterministic dip at DC, dB, decaying over `dc_bins`. Stands for
    /// LO leakage and the DC-offset correction that fights it.
    /// UNCALIBRATED guess.
    pub dc_notch_db: f64,
    /// Standard deviation of the static per-bin phase ripple in the
    /// middle of the band, radians, weighted like the amplitude ripple.
    /// UNCALIBRATED guess.
    pub phase_ripple_rad: f64,
    /// Half-width of the per-packet timing offset `D`, in FFT samples,
    /// drawn uniform in +/- this. UNCALIBRATED. The scale is the point:
    /// one whole sample tilts bin 28 of a 64-point FFT by
    /// `2 pi * 28 / 64` = 2.7 rad, so the slope alone ruins any
    /// cross-subcarrier phase relation.
    pub timing_offset_samples: f64,
    /// Draw a uniform `[-pi, pi)` phase per packet. Not a magnitude but
    /// a switch, because the honest model of a free-running PLL sampled
    /// at 20 Hz has no parameter: successive packets are simply
    /// unrelated.
    pub random_phase: bool,
    /// Standard deviation per I and per Q component of the additive
    /// noise, in counts, added AFTER the gain — so it is receiver noise
    /// referred to the converter input, not channel noise (that is the
    /// scenario's own `noise_std`, which the clean stage already
    /// applied). UNCALIBRATED: 1 count against a target of ~32 counts
    /// is roughly 30 dB of per-packet SNR, a decent indoor link.
    pub noise_counts: f64,
    /// Round to signed int8 and clip at the rails.
    pub quantize: bool,
    /// Write exact zeros into the null and guard bins of the layout. A
    /// real node reports nothing there. Off in the identity config so
    /// that configuration is the untouched clean channel whatever the
    /// layout says.
    /// No effect under [`BinLayout::Ht20Occupied`], which is what a
    /// 56-wide frame selects and what every shipping scenario uses: that
    /// layout declares every entry occupied. It bites only on
    /// [`BinLayout::Ht20Fft`].
    pub zero_null_bins: bool,
}

impl Default for ImpairConfig {
    /// The identity. Impairments are opt-in, so that forgetting to
    /// configure them cannot silently change a clean-channel result.
    fn default() -> Self {
        Self::identity()
    }
}

impl ImpairConfig {
    /// Does nothing, exactly: `apply` leaves every bit of the channel
    /// as it found it. Not "almost nothing" — the arithmetic is unit
    /// gains, a zero phase and no draws from the generator, so the
    /// result is bit-identical. `tests` below asserts that.
    pub const fn identity() -> Self {
        Self {
            agc_step_db: 0.0,
            agc_deadzone_db: 0.0,
            agc_max_steps: 0,
            agc_dither_prob: 0.0,
            agc_range_db: 0.0,
            agc_target_counts: 0.0,
            boot_scale_db: 0.0,
            amp_ripple_db: 0.0,
            edge_rolloff_db: 0.0,
            edge_excess: 1.0,
            dc_bins: 0,
            dc_notch_db: 0.0,
            phase_ripple_rad: 0.0,
            timing_offset_samples: 0.0,
            random_phase: false,
            noise_counts: 0.0,
            quantize: false,
            zero_null_bins: false,
        }
    }

    /// A plausible ESP32-S3. EVERY NUMBER HERE IS A GUESS from published
    /// descriptions of the part and of int8 CSI in general; see each
    /// field for what it is a guess about. Nothing in this repository
    /// has measured any of them, and no accuracy figure anyone has
    /// published is reproduced by this configuration.
    pub const fn esp32_s3() -> Self {
        Self {
            agc_step_db: 0.5,
            agc_deadzone_db: 1.0,
            agc_max_steps: 8,
            agc_dither_prob: 0.02,
            agc_range_db: 60.0,
            // ~1/4 of full scale, leaving ~12 dB of crest headroom: the
            // peak bin of a 56-bin frame runs a few times the mean, and
            // a receiver that clips its own strongest subcarrier every
            // packet would be a broken receiver, not a realistic one.
            agc_target_counts: 32.0,
            boot_scale_db: 1.5,
            amp_ripple_db: 1.0,
            edge_rolloff_db: 3.0,
            edge_excess: 2.0,
            dc_bins: 2,
            dc_notch_db: 2.0,
            phase_ripple_rad: 0.15,
            timing_offset_samples: 1.0,
            random_phase: true,
            noise_counts: 1.0,
            quantize: true,
            zero_null_bins: true,
        }
    }
}

/// One simulated node's receiver: the static part of it drawn once and
/// kept, the per-packet part evolving as packets go through.
///
/// Static state (the per-boot scale, `A_k`, `theta_k`) is a property of
/// the device, so it is drawn once in [`NodeImpairment::new`] and never
/// redrawn. Two nodes in one run differ because their stream ids
/// differ; the same node does not drift across a run. Two boots of the
/// same node differ because `boot` is part of the stream id, which is
/// the point of the `s` term.
#[derive(Debug, Clone)]
pub struct NodeImpairment {
    cfg: ImpairConfig,
    layout: BinLayout,
    n: usize,
    /// `s`: per-boot reporting scale, linear.
    scale: f64,
    /// `A_k`, linear, per entry. Null bins hold 0.
    amp: Vec<f64>,
    /// `theta_k`, radians, per entry.
    phase: Vec<f64>,
    /// `2 pi k / N` per entry: multiply by `D[n]` for the slope.
    slope: Vec<f64>,
    /// Entries that carry nothing.
    null: Vec<bool>,
    /// AGC gain in quanta of `agc_step_db`. Starts at nominal, so the
    /// first packets after a boot are mis-gained until the loop catches
    /// up — which is what a boot looks like.
    gain_index: i32,
    rng: Rng,
    packets: u64,
}

impl NodeImpairment {
    /// Draw a node's static impairments. `n` is the channel array
    /// length, `seed` the scenario seed, `node_id` the node, and `boot`
    /// which power cycle this is (any two values give two different
    /// devices-worth of `s`).
    ///
    /// The layout is [`BinLayout::for_n`]; use
    /// [`NodeImpairment::with_layout`] to say otherwise.
    pub fn new(cfg: ImpairConfig, n: usize, seed: u64, node_id: u8, boot: u32) -> Self {
        Self::with_layout(cfg, BinLayout::for_n(n), n, seed, node_id, boot)
    }

    /// [`NodeImpairment::new`] with the subcarrier layout stated rather
    /// than inferred.
    ///
    /// # Panics
    /// If the layout fixes a length and `n` is not it, because a wrong
    /// bin map is worse than a stopped test: the ripple and the slope
    /// would be functions of the wrong subcarrier.
    pub fn with_layout(cfg: ImpairConfig, layout: BinLayout, n: usize, seed: u64, node_id: u8, boot: u32) -> Self {
        if let Some(need) = layout.required_len() {
            assert_eq!(n, need, "layout {layout:?} describes {need} entries, got {n}");
        }
        // One stream per (node, boot). Folding both into the id rather
        // than reseeding keeps every node independent of how many other
        // nodes or boots exist, as the link streams do.
        let mut rng = Rng::stream(seed, STREAM_IMPAIR + ((node_id as u64) << 32) + boot as u64);

        let scale = db_to_linear(rng.range(-cfg.boot_scale_db, cfg.boot_scale_db));

        let k_max = (0..n).filter(|&i| !layout.is_null(i, n)).map(|i| layout.bin(i, n).abs()).max().unwrap_or(1).max(1) as f64;
        let fft = layout.fft_size(n) as f64;

        let (mut amp, mut phase, mut slope, mut null) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let k = layout.bin(i, n);
            let is_null = layout.is_null(i, n);
            null.push(is_null);
            slope.push(TAU * k as f64 / fft);
            if is_null {
                // Nothing is measured here, so there is no response to
                // have. Draw nothing either: a guard bin must not shift
                // the stream the occupied bins are drawn from.
                amp.push(0.0);
                phase.push(0.0);
                continue;
            }
            let x = k.abs() as f64 / k_max;
            // "Larger at the band edges and near DC": the spread grows
            // quadratically towards the edge, and the few bins either
            // side of DC are treated as an edge too, because that is
            // where LO leakage and DC-offset correction land.
            let near_dc = cfg.dc_bins > 0 && k.abs() <= cfg.dc_bins;
            let w = (1.0 + (cfg.edge_excess - 1.0) * x * x).max(if near_dc { cfg.edge_excess } else { 0.0 });
            let notch = if near_dc { cfg.dc_notch_db * (1.0 - k.abs() as f64 / (cfg.dc_bins + 1) as f64) } else { 0.0 };
            let a_db = -cfg.edge_rolloff_db * x * x - notch + cfg.amp_ripple_db * w * rng.gauss();
            amp.push(db_to_linear(a_db));
            phase.push(cfg.phase_ripple_rad * w * rng.gauss());
        }

        Self { cfg, layout, n, scale, amp, phase, slope, null, gain_index: 0, rng, packets: 0 }
    }

    pub fn config(&self) -> &ImpairConfig {
        &self.cfg
    }

    pub fn layout(&self) -> BinLayout {
        self.layout
    }

    /// `s`, the per-boot reporting scale (linear). Constant for the life
    /// of this node.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// `A_k`, linear, one per entry. Constant for the life of this node.
    pub fn amp_ripple(&self) -> &[f64] {
        &self.amp
    }

    /// `theta_k`, radians, one per entry. Constant for the life of this
    /// node.
    pub fn phase_ripple(&self) -> &[f64] {
        &self.phase
    }

    /// The AGC gain in force after the last packet, dB. Always an exact
    /// multiple of `agc_step_db`.
    pub fn gain_db(&self) -> f64 {
        self.gain_index as f64 * self.cfg.agc_step_db
    }

    /// Packets applied since the boot.
    pub fn packets(&self) -> u64 {
        self.packets
    }

    fn gain_linear(&self) -> f64 {
        if self.cfg.agc_step_db > 0.0 {
            db_to_linear(self.gain_db())
        } else {
            1.0
        }
    }

    /// Impair one packet in place: clean complex channel in, impaired
    /// complex channel out, in int8 counts.
    ///
    /// Per packet `n` and entry `i` with subcarrier index `k`:
    ///
    /// ```text
    /// H_meas[i] = Q( s * G[n] * A_k * H[i] * exp(j(phi0[n] + 2 pi k D[n] / N + theta_k)) + w )
    /// ```
    ///
    /// # What the AGC is for, and what that means for quantization
    ///
    /// An AGC exists to keep the signal inside the converter's range.
    /// So the gain is not a free parameter: it is chosen, per packet,
    /// to put the mean |H| of the occupied bins near
    /// `agc_target_counts`, and int8 counts are what comes out. Two
    /// consequences, and they are the reason this stage is worth having:
    ///
    /// - The absolute scale of the clean channel cancels. Whatever
    ///   `STATIC_MAGNITUDE` the physics used, the reported level is
    ///   about the target — which is exactly why ESP32 CSI amplitude
    ///   carries no absolute meaning and why a detector must not learn
    ///   one.
    /// - The quantizer bites where it matters. With the mean bin near
    ///   32 counts, a chest moving |h| by a fraction of a percent is a
    ///   small fraction of one LSB per packet: it survives only by
    ///   averaging over a window, which is what the DSP does. Scaling
    ///   each packet to fill the range instead would have made
    ///   quantization a rounding detail and the whole model pointless.
    ///
    /// The per-boot scale `s` is applied where the loop cannot see it.
    /// A loop that also corrected `s` would cancel it and the term would
    /// be a no-op; physically the loop is closed around an analog
    /// detector, while `s` stands for the uncalibrated mapping from
    /// converter counts to the integers the driver reports. The gain is
    /// also settled within this same packet, because 802.11 settles its
    /// AGC during the preamble, before the training fields the CSI is
    /// measured from.
    ///
    /// # Panics
    /// If `re` and `im` are not both the length this node was built for.
    pub fn apply(&mut self, re: &mut [f64], im: &mut [f64]) {
        assert_eq!(re.len(), self.n, "channel length");
        assert_eq!(im.len(), self.n, "channel length");
        self.packets = self.packets.wrapping_add(1);

        // The level the gain control sees: post-ripple, pre-scale, over
        // the bins that carry something.
        let mut sum = 0.0;
        let mut count = 0usize;
        for i in 0..self.n {
            if self.null[i] {
                continue;
            }
            sum += self.amp[i] * re[i].hypot(im[i]);
            count += 1;
        }
        let level = if count > 0 { sum / count as f64 } else { 0.0 };
        self.update_gain(level);
        let g = self.scale * self.gain_linear();

        // Per-packet phase terms. Drawn unconditionally when enabled so
        // the stream does not depend on the channel.
        let phi0 = if self.cfg.random_phase { self.rng.range(-std::f64::consts::PI, std::f64::consts::PI) } else { 0.0 };
        let d = if self.cfg.timing_offset_samples > 0.0 { self.rng.range(-self.cfg.timing_offset_samples, self.cfg.timing_offset_samples) } else { 0.0 };

        for i in 0..self.n {
            if self.null[i] {
                if self.cfg.zero_null_bins {
                    re[i] = 0.0;
                    im[i] = 0.0;
                }
                continue;
            }
            let (sin, cos) = (phi0 + d * self.slope[i] + self.phase[i]).sin_cos();
            let a = g * self.amp[i];
            let (hr, hi) = (re[i], im[i]);
            let mut r = a * (hr * cos - hi * sin);
            let mut j = a * (hr * sin + hi * cos);
            if self.cfg.noise_counts > 0.0 {
                r += self.cfg.noise_counts * self.rng.gauss();
                j += self.cfg.noise_counts * self.rng.gauss();
            }
            if self.cfg.quantize {
                r = quantize(r) as f64;
                j = quantize(j) as f64;
            }
            re[i] = r;
            im[i] = j;
        }
    }

    /// [`NodeImpairment::apply`], then the signed int8 pair for each
    /// entry written in WIRE ORDER: `iq[2i]` is the imaginary part and
    /// `iq[2i + 1]` the real part.
    ///
    /// That order is not a choice made here. It is what the firmware's
    /// `serialize_raw` writes (`edge/firmware/main/csi_capture.cpp`) and
    /// what the gateway's raw parser reads (`saans-gateway/src/raw.rs`),
    /// which is a byte-exact port of the Python reference. Getting it
    /// backwards leaves amplitude right and mirrors every phase, so it
    /// survives any amplitude-only test and then breaks anything
    /// phase-based — which is why it is written once, here, rather than
    /// at each call site.
    ///
    /// `re` and `im` are left holding the quantized values, whatever
    /// `cfg.quantize` says, so the floating-point view and the bytes
    /// cannot disagree about what this node reported.
    ///
    /// # Panics
    /// If `iq` is not `2 * n` long, or `re`/`im` are the wrong length.
    pub fn apply_to_iq(&mut self, re: &mut [f64], im: &mut [f64], iq: &mut [i8]) {
        assert_eq!(iq.len(), 2 * self.n, "iq length");
        self.apply(re, im);
        for i in 0..self.n {
            // Idempotent where `apply` already quantized.
            let (q_im, q_re) = (quantize(im[i]), quantize(re[i]));
            iq[2 * i] = q_im;
            iq[2 * i + 1] = q_re;
            im[i] = q_im as f64;
            re[i] = q_re as f64;
        }
    }

    /// One packet of the AGC walk. Piecewise constant by construction:
    /// the gain lives on an integer grid of `agc_step_db`, so it either
    /// holds or moves by at least one whole quantum.
    fn update_gain(&mut self, level: f64) {
        if self.cfg.agc_step_db <= 0.0 {
            return;
        }
        // Drawn before anything is tested, for the reason `phi0` and `d`
        // are: a stream whose *length* depends on what the channel was
        // doing de-pairs every seed-matched comparison. Two runs that
        // differ only in body motion would otherwise take different
        // numbers of draws from the packet where one of them first
        // leaves the dead zone, and every noise sample after that would
        // differ for a reason that has nothing to do with the body.
        let dither = self.rng.uniform();
        let direction = self.rng.uniform();
        if level > 0.0 && self.cfg.agc_target_counts > 0.0 {
            let err_db = 20.0 * (self.cfg.agc_target_counts / (level * self.gain_linear())).log10();
            if err_db.abs() > self.cfg.agc_deadzone_db {
                let steps = (err_db / self.cfg.agc_step_db).round() as i32;
                self.gain_index += steps.clamp(-self.cfg.agc_max_steps, self.cfg.agc_max_steps);
            } else if dither < self.cfg.agc_dither_prob {
                // Inside the dead zone the loop has no reason to move,
                // and sometimes moves anyway.
                self.gain_index += if direction < 0.5 { 1 } else { -1 };
            }
        }
        let limit = (self.cfg.agc_range_db / self.cfg.agc_step_db).floor().max(0.0) as i32;
        self.gain_index = self.gain_index.clamp(-limit, limit);
    }
}

/// dB to a linear amplitude ratio. Exactly 1.0 at 0 dB, which is what
/// makes the identity configuration the exact identity.
fn db_to_linear(db: f64) -> f64 {
    if db == 0.0 {
        1.0
    } else {
        10f64.powf(db / 20.0)
    }
}

/// One I or Q component to signed int8: round half away from zero, clip
/// at the rails. Clipping is kept rather than designed around — an AGC
/// that is a step too slow really does clip the next packet, and a model
/// that rescales to avoid it would hide the failure it exists to show.
pub fn quantize(x: f64) -> i8 {
    let r = if x.is_nan() { 0.0 } else { x.round() };
    r.clamp(IQ_MIN, IQ_MAX) as i8
}

#[cfg(test)]
mod tests {

    /// The RNG stream must advance by the same number of draws per
    /// packet whatever the channel is doing. It did not: the AGC dither
    /// drew conditionally, so a loud packet shifted every later draw by
    /// one and two runs differing only in body motion got different
    /// noise. That silently de-pairs every seed-matched A/B measurement
    /// made with this layer, which is most of what it is for.
    #[test]
    fn the_stream_does_not_depend_on_the_channel() {
        // Noise off, so the phase a packet comes out with is the static
        // ripple plus `phi0` plus the timing slope and nothing else. The
        // gain is an amplitude, so with noise off it cannot reach the
        // phase — which makes any phase difference proof that the draw
        // stream moved, rather than that a different gain was applied.
        let cfg = ImpairConfig { noise_counts: 0.0, quantize: false, ..ImpairConfig::esp32_s3() };
        let n = 56;
        let run = |loud: bool| {
            let mut node = NodeImpairment::new(cfg, n, 7, 1, 0);
            let mut seen = Vec::new();
            for packet in 0..8 {
                // The ordinary level sits ON the AGC target, so the
                // loop is inside its dead zone and the dither branch —
                // the one that used to draw conditionally — is the one
                // that runs. One packet is driven far outside it, so
                // that packet takes the other branch instead.
                let level = if loud && packet == 2 { 4000.0 } else { cfg.agc_target_counts };
                let mut re = vec![level; n];
                let mut im = vec![0.0; n];
                node.apply(&mut re, &mut im);
                // The phase offset is drawn after the gain update, so it
                // is where a lost or extra draw shows up first.
                seen.push(im[0].atan2(re[0]));
            }
            seen
        };
        let quiet = run(false);
        let loud = run(true);
        // Packet 2 itself legitimately differs: a different gain really
        // was applied. Every packet after it must not.
        for (packet, (q, l)) in quiet.iter().zip(&loud).enumerate().skip(3) {
            assert!(
                (q - l).abs() < 1e-12,
                "packet {packet}: the stream diverged after a loud packet ({q} vs {l})"
            );
        }
    }
    use super::*;

    /// A clean channel with structure across the band, so a bug that
    /// touches one bin shows up.
    fn clean(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut rng = Rng::new(12345);
        let (mut re, mut im) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let a = 30.0 + 10.0 * (i as f64 * 0.37).sin();
            let p = (i as f64) * 0.11 + rng.uniform();
            re.push(a * p.cos());
            im.push(a * p.sin());
        }
        (re, im)
    }

    #[test]
    fn identity_is_exactly_the_identity() {
        for &n in &[56usize, 64] {
            let (re0, im0) = clean(n);
            let (mut re, mut im) = (re0.clone(), im0.clone());
            let mut node = NodeImpairment::new(ImpairConfig::identity(), n, 7, 2, 0);
            for _ in 0..25 {
                node.apply(&mut re, &mut im);
                // Bitwise: "within an epsilon" would not prove the
                // clean path is untouched, and that is the claim.
                for i in 0..n {
                    assert_eq!(re[i].to_bits(), re0[i].to_bits(), "n={n} re[{i}]");
                    assert_eq!(im[i].to_bits(), im0[i].to_bits(), "n={n} im[{i}]");
                }
            }
            assert_eq!(node.gain_db(), 0.0);
        }
    }

    #[test]
    fn default_config_is_the_identity() {
        assert_eq!(ImpairConfig::default(), ImpairConfig::identity());
    }

    #[test]
    fn ht20_layout_has_56_occupied_bins() {
        // The reconciliation with the firmware's 56: 64 FFT bins minus
        // DC minus 7 guard bins IS 56.
        let n = 64;
        let occupied: Vec<i32> = (0..n).filter(|&i| !BinLayout::Ht20Fft.is_null(i, n)).map(|i| BinLayout::Ht20Fft.bin(i, n)).collect();
        assert_eq!(occupied.len(), 56);
        let mut sorted = occupied.clone();
        sorted.sort_unstable();
        let want: Vec<i32> = (-28..=-1).chain(1..=28).collect();
        assert_eq!(sorted, want);
        // And the 56-entry layout is those same bins, in order.
        let direct: Vec<i32> = (0..56).map(|i| BinLayout::Ht20Occupied.bin(i, 56)).collect();
        assert_eq!(direct, want);
        assert!((0..56).all(|i| !BinLayout::Ht20Occupied.is_null(i, 56)));
    }

    #[test]
    fn null_bins_are_zero() {
        let n = 64;
        let (mut re, mut im) = clean(n);
        let mut node = NodeImpairment::new(ImpairConfig::esp32_s3(), n, 99, 1, 0);
        for _ in 0..10 {
            node.apply(&mut re, &mut im);
            for i in 0..n {
                let k = BinLayout::Ht20Fft.bin(i, n);
                if k == 0 || k.abs() >= 29 {
                    assert_eq!(re[i], 0.0, "bin {k} real");
                    assert_eq!(im[i], 0.0, "bin {k} imag");
                    assert_eq!(node.amp_ripple()[i], 0.0, "bin {k} ripple");
                }
            }
            // A zeroed bin must not make the next packet's input zero:
            // refill from a clean channel each time.
            let (r2, i2) = clean(n);
            re.copy_from_slice(&r2);
            im.copy_from_slice(&i2);
        }
    }

    #[test]
    fn static_ripple_is_stable_across_packets_and_differs_between_nodes() {
        let n = 56;
        let cfg = ImpairConfig::esp32_s3();
        let mut a = NodeImpairment::new(cfg, n, 4242, 1, 0);
        let b = NodeImpairment::new(cfg, n, 4242, 2, 0);
        let (amp0, ph0, s0) = (a.amp_ripple().to_vec(), a.phase_ripple().to_vec(), a.scale());

        let (mut re, mut im) = clean(n);
        for _ in 0..200 {
            let (r, i) = clean(n);
            re.copy_from_slice(&r);
            im.copy_from_slice(&i);
            a.apply(&mut re, &mut im);
        }
        // A property of the device, not of the packet.
        assert_eq!(a.amp_ripple(), &amp0[..]);
        assert_eq!(a.phase_ripple(), &ph0[..]);
        assert_eq!(a.scale(), s0);

        // Two nodes in one run are two different devices.
        assert_ne!(a.amp_ripple(), b.amp_ripple());
        assert_ne!(a.phase_ripple(), b.phase_ripple());
        assert_ne!(a.scale(), b.scale());

        // And two boots of one node differ too: that is what `s` is for.
        let boot1 = NodeImpairment::new(cfg, n, 4242, 1, 1);
        assert_ne!(boot1.scale(), s0);

        // The ripple is a believable size rather than an accidental 0
        // or an accidental 10x.
        let max_db = amp0.iter().map(|a| 20.0 * a.log10()).fold(0.0f64, |m, v: f64| m.max(v.abs()));
        assert!((0.5..12.0).contains(&max_db), "amplitude ripple spread {max_db} dB");
    }

    #[test]
    fn agc_is_piecewise_constant_with_steps_no_smaller_than_the_minimum() {
        let n = 56;
        let cfg = ImpairConfig::esp32_s3();
        let mut node = NodeImpairment::new(cfg, n, 5, 1, 0);
        let (base_re, base_im) = clean(n);

        let mut gains = Vec::new();
        let mut moves = 0;
        for p in 0..400 {
            // Walk the channel level over ~40 dB so the loop has to
            // work, with a jump partway: a slow ramp alone could be
            // tracked by anything.
            let level = if p < 200 { 10f64.powf(p as f64 * 0.1 / 20.0) } else { 0.01 };
            let mut re: Vec<f64> = base_re.iter().map(|v| v * level).collect();
            let mut im: Vec<f64> = base_im.iter().map(|v| v * level).collect();
            node.apply(&mut re, &mut im);
            gains.push(node.gain_db());
        }
        for w in gains.windows(2) {
            let d = w[1] - w[0];
            if d != 0.0 {
                moves += 1;
                assert!(d.abs() >= cfg.agc_step_db - 1e-12, "step {d} dB is smaller than the {} dB quantum", cfg.agc_step_db);
            }
            // On the grid, not between its points.
            let q = w[1] / cfg.agc_step_db;
            assert!((q - q.round()).abs() < 1e-9, "gain {} dB is off the quantum grid", w[1]);
        }
        assert!(moves > 5, "the AGC never moved ({moves} changes): the test proved nothing");
        // Piecewise: it holds far more often than it moves.
        assert!(moves < gains.len() / 4, "the AGC moved on {moves} of {} packets, which is not piecewise constant", gains.len());
    }

    #[test]
    fn agc_holds_the_level_near_the_target() {
        let n = 56;
        let cfg = ImpairConfig::esp32_s3();
        let mut node = NodeImpairment::new(cfg, n, 11, 4, 0);
        let (base_re, base_im) = clean(n);
        // 80 dB of channel level, inside the modelled gain range. Push
        // further (level 1e-3 needs ~60 dB of gain) and the clamp binds
        // and the level comes out low -- deliberately, see
        // `agc_range_db`; that is the end of the receiver, not a bug.
        for level in [0.01, 1.0, 100.0] {
            let (mut re, mut im) = (vec![0.0; n], vec![0.0; n]);
            // A few packets to let the loop slew, then look.
            for _ in 0..20 {
                for i in 0..n {
                    re[i] = base_re[i] * level;
                    im[i] = base_im[i] * level;
                }
                node.apply(&mut re, &mut im);
            }
            let mean: f64 = (0..n).map(|i| re[i].hypot(im[i])).sum::<f64>() / n as f64;
            let err_db = 20.0 * (mean / cfg.agc_target_counts).log10();
            // Dead zone + boot scale + half a quantum + the ripple's
            // effect on the mean. Nowhere near the rails either way.
            assert!(err_db.abs() < 6.0, "level {level}: mean {mean} counts is {err_db} dB from target");
        }
    }

    #[test]
    fn quantization_round_trips_within_half_a_count() {
        let n = 56;
        let quantizing = ImpairConfig::esp32_s3();
        assert!(quantizing.quantize);
        let mut plain = quantizing;
        plain.quantize = false;

        // Same seed, same draws: `quantize` consumes no randomness, so
        // the two runs differ only by the rounding.
        let mut a = NodeImpairment::new(quantizing, n, 2024, 6, 0);
        let mut b = NodeImpairment::new(plain, n, 2024, 6, 0);
        for _ in 0..50 {
            let (mut re_a, mut im_a) = clean(n);
            let (mut re_b, mut im_b) = clean(n);
            a.apply(&mut re_a, &mut im_a);
            b.apply(&mut re_b, &mut im_b);
            for i in 0..n {
                assert_eq!(re_a[i], re_a[i].round(), "not an integer count");
                assert!((IQ_MIN..=IQ_MAX).contains(&re_a[i]) && (IQ_MIN..=IQ_MAX).contains(&im_a[i]), "outside int8");
                // Only meaningful where the unquantized value was in
                // range; at the rails the error is the clip, not the
                // rounding, and that is deliberate.
                if (IQ_MIN..=IQ_MAX).contains(&re_b[i]) {
                    assert!((re_a[i] - re_b[i]).abs() <= 0.5 + 1e-12, "re error {} counts", re_a[i] - re_b[i]);
                }
                if (IQ_MIN..=IQ_MAX).contains(&im_b[i]) {
                    assert!((im_a[i] - im_b[i]).abs() <= 0.5 + 1e-12, "im error {} counts", im_a[i] - im_b[i]);
                }
            }
        }

        // And the wire pairs are the same numbers, imaginary first.
        let (mut re, mut im) = clean(n);
        let mut iq = vec![0i8; 2 * n];
        let mut c = NodeImpairment::new(quantizing, n, 2024, 6, 0);
        c.apply_to_iq(&mut re, &mut im, &mut iq);
        for i in 0..n {
            assert_eq!(iq[2 * i] as f64, im[i], "iq[2k] must be the imaginary part");
            assert_eq!(iq[2 * i + 1] as f64, re[i], "iq[2k+1] must be the real part");
        }
    }

    #[test]
    fn same_seed_gives_the_same_output_twice() {
        let n = 56;
        let run = || {
            let mut node = NodeImpairment::new(ImpairConfig::esp32_s3(), n, 31337, 5, 2);
            let mut out = Vec::new();
            for _ in 0..40 {
                let (mut re, mut im) = clean(n);
                let mut iq = vec![0i8; 2 * n];
                node.apply_to_iq(&mut re, &mut im, &mut iq);
                out.extend_from_slice(&iq);
            }
            out
        };
        assert_eq!(run(), run());
        // A different seed is a different receiver.
        let mut other = NodeImpairment::new(ImpairConfig::esp32_s3(), n, 31338, 5, 2);
        let (mut re, mut im) = clean(n);
        let mut iq = vec![0i8; 2 * n];
        other.apply_to_iq(&mut re, &mut im, &mut iq);
        assert_ne!(iq, run()[..2 * n].to_vec());
    }

    #[test]
    fn phase_is_destroyed_packet_to_packet_and_amplitude_is_not() {
        // Why the DSP works on amplitude, as a test. Hold the channel
        // perfectly static and look at what varies between packets: the
        // phase profile is unrecognisable from one packet to the next
        // (phi0 and the timing slope are redrawn every packet), while
        // the amplitude profile repeats, because s and A_k do not move
        // and the AGC gain is a scalar every bin shares.
        let n = 56;
        let mut cfg = ImpairConfig::esp32_s3();
        cfg.noise_counts = 0.0; // isolate the phase terms from the noise
        let mut node = NodeImpairment::new(cfg, n, 808, 9, 0);
        let (c_re, c_im) = clean(n);
        let c_amp: Vec<f64> = (0..n).map(|i| c_re[i].hypot(c_im[i])).collect();

        let mut first: Option<(Vec<f64>, Vec<f64>)> = None;
        let (mut max_phase_jump, mut worst_amp_repeat, mut worst_vs_clean) = (0.0f64, 1.0f64, 1.0f64);
        for _ in 0..30 {
            let (mut re, mut im) = (c_re.clone(), c_im.clone());
            node.apply(&mut re, &mut im);
            let ph: Vec<f64> = (0..n).map(|i| im[i].atan2(re[i])).collect();
            let amp: Vec<f64> = (0..n).map(|i| re[i].hypot(im[i])).collect();
            worst_vs_clean = worst_vs_clean.min(corr(&amp, &c_amp));
            match &first {
                Some((f_ph, f_amp)) => {
                    max_phase_jump = max_phase_jump.max((0..n).map(|i| wrap(ph[i] - f_ph[i]).abs()).fold(0.0, f64::max));
                    worst_amp_repeat = worst_amp_repeat.min(corr(&amp, f_amp));
                }
                None => first = Some((ph, amp)),
            }
        }
        assert!(max_phase_jump > 2.0, "phase barely moved ({max_phase_jump} rad): the per-packet terms are not doing their job");
        assert!(worst_amp_repeat > 0.99, "a static channel's amplitude profile did not repeat (corr {worst_amp_repeat})");

        // The other half of the story, asserted loosely because the
        // number is only as good as the uncalibrated ripple that
        // produces it: the reported amplitude profile is NOT the clean
        // one. With these constants the correlation sits around 0.7 --
        // a +/-2 dB static ripple is a large fraction of a band's
        // structure. It is fixed per device, so a detector that works
        // on changes is fine and one that compares nodes is not.
        assert!((0.4..0.95).contains(&worst_vs_clean), "clean-profile correlation {worst_vs_clean} is outside the range this model should give");
    }

    fn wrap(x: f64) -> f64 {
        let t = std::f64::consts::TAU;
        let y = (x + std::f64::consts::PI).rem_euclid(t);
        y - std::f64::consts::PI
    }

    fn corr(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
        let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
        for i in 0..a.len() {
            let (x, y) = (a[i] - ma, b[i] - mb);
            num += x * y;
            da += x * x;
            db += y * y;
        }
        if da <= 0.0 || db <= 0.0 {
            return 0.0;
        }
        num / (da * db).sqrt()
    }
}
