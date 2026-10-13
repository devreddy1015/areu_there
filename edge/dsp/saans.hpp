// SPDX-License-Identifier: MIT
#pragma once
//
// Saans edge DSP — the whole library in one include.
//
// For firmware and host tools that want everything. A translation unit that
// needs one stage should include that stage's header directly: these are
// header-only templates, and on an ESP-IDF build the difference between
// including `biquad.hpp` and including all of this is real compile time.
//
// ---------------------------------------------------------------------------
// WHAT IS IN HERE, ROUGHLY IN THE ORDER A CSI FRAME MEETS IT
// ---------------------------------------------------------------------------
//
//   core.hpp              CsiFrame, RingBuffer, compensated summation; the
//                         no-allocation and float32 contracts (ADR-003)
//   gapfill.hpp           SequenceGapFiller — restore the time axis first
//   preprocess.hpp        StaticSuppressor, corner_to_alpha, fuse
//   presence.hpp          PresenceDetector, RunningVariance
//   biquad.hpp            BiquadBandpass
//   spectral.hpp          goertzel_power, hann_window, band_peak, sideband_peak
//   fft.hpp               next_pow2, fft_inplace, power_at
//   bnr.hpp               subcarrier_bnr — per-subcarrier link quality
//   vitals.hpp            BreathingExtractor, HeartRateExtractor
//   pipeline.hpp          SensingPipeline — the stages wired together
//   crc32.hpp             CRC-32 (zlib), for the wire frame
//   feature_frame.hpp     FeatureFrame v1 encode/decode and the consent gate
//   feature_extractor.hpp FeatureExtractor — the object the firmware drives
//
// And one header that is NOT on the CSI path at all:
//
//   tdma.hpp              TdmaSchedule — the S6 broadcast-TDMA model. Pure
//                         arithmetic over a mesh configuration: per-link rate,
//                         slot utilisation, clock skew, which links a node
//                         owns, and whether the schedule is runnable at all.
//                         It answers "does this mesh fit?" at boot or in a
//                         planning tool. Its numbers are SYNTHETIC — no radio
//                         has been involved — and its own header says so at
//                         length.
//
// ---------------------------------------------------------------------------
// THE THREE RULES THIS LIBRARY IS BUILT UNDER
// ---------------------------------------------------------------------------
//   1. HEADER-ONLY C++17, compiled with -fno-exceptions -fno-rtti. There is
//      no .cpp anywhere in it. A configuration that cannot be honoured leaves
//      the object INERT and reports `valid() == false`; nothing is ever
//      silently approximated, because a mis-configured DSP stage still emits a
//      plausible-looking waveform.
//   2. NO HEAP AFTER INIT. Every buffer is a member array sized by a template
//      parameter, or scratch the caller owns. No vector, deque, string,
//      function, new or malloc.
//   3. FLOAT, NOT DOUBLE (ADR-003). The ESP32-S3 FPU is single precision, so a
//      double is a software-emulation trap. The documented exceptions are time
//      accumulators and init-time sample counts; each site says so and says
//      why.
//
// Parity with `wifisense/` is the contract (SAANS_PLAN §4): 1e-4 relative on
// filter outputs, 0.05 BPM on rates, byte equality on the codec. The host
// tests in `test/` check it against `spec/golden/`, and a missing vector is a
// hard failure there, never a skip.
//
#include "biquad.hpp"
#include "bnr.hpp"
#include "core.hpp"
#include "crc32.hpp"
#include "feature_extractor.hpp"
#include "feature_frame.hpp"
#include "fft.hpp"
#include "gapfill.hpp"
#include "pipeline.hpp"
#include "preprocess.hpp"
#include "presence.hpp"
#include "spectral.hpp"
#include "tdma.hpp"
#include "vitals.hpp"
