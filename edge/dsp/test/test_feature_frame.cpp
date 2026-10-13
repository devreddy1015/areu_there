// FeatureFrame v1 codec parity against the Python reference.
//
// Vector set: feature_codec (spec/golden/). This one is BYTE EQUALITY, not a
// tolerance: the codec is pure serialisation, so there is no float32-vs-f64
// excuse available to it. The golden field values are f64; the reference
// rounds them to f32 at pack time and so does `static_cast<float>` here, with
// the same round-to-nearest-even, which is why the 68 bytes can match exactly.
//
//   g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wdouble-promotion
//       -fno-exceptions -fno-rtti -I.. test_feature_frame.cpp -o test_feature_frame
//   ./test_feature_frame [path/to/spec/golden]
//
// LOADER: the shared one, `edge/dsp/test/golden.hpp`. This file originally
// carried a local stand-in, written before the shared loader was in the tree,
// with a note to delete it once that landed; it has, and it is. Everything
// this file needs from a loader is `scalar()` and `array()`, which the shared
// one provides with the same signatures and the same contract -- including the
// one that matters, that a missing vector is a loud non-zero exit and never a
// skip.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "crc32.hpp"
#include "feature_frame.hpp"

#include "golden.hpp"

// ---------------------------------------------------------------------------
// Tiny harness
// ---------------------------------------------------------------------------

namespace {

int g_checks = 0;
int g_failures = 0;
int g_cases = 0;
int g_case_failures = 0;
const char* g_case = "";

void begin_case(const char* name) {
    g_case = name;
    g_cases += 1;
    g_case_failures = g_failures;
}

void end_case() {
    std::printf("  %-44s %s\n", g_case, g_failures == g_case_failures ? "ok" : "FAILED");
}

bool check(bool ok, const char* what) {
    g_checks += 1;
    if (!ok) {
        g_failures += 1;
        std::printf("    FAIL [%s] %s\n", g_case, what);
    }
    return ok;
}

bool check_u32(std::uint32_t got, std::uint32_t want, const char* what) {
    const bool ok = check(got == want, what);
    if (!ok) {
        std::printf("      got 0x%08X want 0x%08X\n", got, want);
    }
    return ok;
}

bool check_i64(long long got, long long want, const char* what) {
    const bool ok = check(got == want, what);
    if (!ok) {
        std::printf("      got %lld want %lld\n", got, want);
    }
    return ok;
}

bool check_f32(float got, float want, const char* what) {
    // Exact: the codec is lossless once the value is a float32. Anything else
    // here would be a bug being tolerated.
    const bool ok = check(saans::detail::f32_bits(got) == saans::detail::f32_bits(want), what);
    if (!ok) {
        std::printf("      got %.9g (0x%08X) want %.9g (0x%08X)\n", static_cast<double>(got),
                    saans::detail::f32_bits(got), static_cast<double>(want),
                    saans::detail::f32_bits(want));
    }
    return ok;
}

// ---------------------------------------------------------------------------
// 1. CRC-32 against zlib
// ---------------------------------------------------------------------------

void test_crc32_known_answers() {
    begin_case("crc32 matches zlib.crc32");
    const std::uint8_t* empty = reinterpret_cast<const std::uint8_t*>("");
    check_u32(saans::crc32(empty, 0), 0x00000000u, "crc32(b\"\") == 0");

    const char* a = "a";
    check_u32(saans::crc32(reinterpret_cast<const std::uint8_t*>(a), 1), 0xE8B7BE43u,
              "crc32(b\"a\")");

    const char* check_vec = "123456789";  // the standard CRC-32 check value
    check_u32(saans::crc32(reinterpret_cast<const std::uint8_t*>(check_vec), 9), 0xCBF43926u,
              "crc32(b\"123456789\") == 0xCBF43926");

    const char* sans = "SANS";
    check_u32(saans::crc32(reinterpret_cast<const std::uint8_t*>(sans), 4), 0x153CFA62u,
              "crc32(b\"SANS\")");

    std::uint8_t all[256];
    for (int i = 0; i < 256; ++i) {
        all[i] = static_cast<std::uint8_t>(i);
    }
    check_u32(saans::crc32(all, sizeof all), 0x29058C73u, "crc32(bytes(range(256)))");

    // zlib's chaining convention: crc32(b, crc32(a)) == crc32(a + b).
    const std::uint32_t part = saans::crc32(reinterpret_cast<const std::uint8_t*>("12345"), 5);
    check_u32(saans::crc32_update(part, reinterpret_cast<const std::uint8_t*>("6789"), 4),
              0xCBF43926u, "crc32_update chains like zlib.crc32(data, value)");
    end_case();
}

// ---------------------------------------------------------------------------
// 2. feature_codec: byte equality
// ---------------------------------------------------------------------------

saans::FeatureFrame frame_from_golden(const saans::golden::Vectors& g, std::size_t i) {
    saans::FeatureFrame f;
    f.node_id = static_cast<std::uint8_t>(g.array("node_id")[i]);
    f.tx_id = static_cast<std::uint8_t>(g.array("tx_id")[i]);
    f.flags = static_cast<std::uint8_t>(g.array("flags")[i]);
    f.seq = static_cast<std::uint32_t>(g.array("seq")[i]);
    f.uptime_ms = static_cast<std::uint32_t>(g.array("uptime_ms")[i]);
    f.csi_rate_chz = static_cast<std::int32_t>(g.array("csi_rate_chz")[i]);
    f.loss_permille = static_cast<std::int32_t>(g.array("loss_permille")[i]);
    f.motion_energy = static_cast<float>(g.array("motion_energy")[i]);
    f.presence_threshold = static_cast<float>(g.array("presence_threshold")[i]);
    f.gross_motion = static_cast<float>(g.array("gross_motion")[i]);
    f.breath_rms = static_cast<float>(g.array("breath_rms")[i]);
    f.breathing_bpm = static_cast<float>(g.array("breathing_bpm")[i]);
    f.breathing_conf = static_cast<float>(g.array("breathing_conf")[i]);
    f.heart_bpm = static_cast<float>(g.array("heart_bpm")[i]);
    f.heart_conf = static_cast<float>(g.array("heart_conf")[i]);
    f.best_bnr = static_cast<float>(g.array("best_bnr")[i]);
    f.calib_age_s = static_cast<std::uint32_t>(g.array("calib_age_s")[i]);
    f.rssi_dbm = static_cast<std::int32_t>(g.array("rssi_dbm")[i]);
    f.noise_floor_dbm = static_cast<std::int32_t>(g.array("noise_floor_dbm")[i]);
    f.n_subcarriers = static_cast<std::int32_t>(g.array("n_subcarriers")[i]);
    return f;
}

void test_golden_bytes(const saans::golden::Vectors& g) {
    const std::size_t frames = static_cast<std::size_t>(g.scalar("frames"));
    const std::vector<double>& bytes = g.array("bytes");

    begin_case("feature_codec vector is what we expect");
    check_i64(static_cast<long long>(g.scalar("frame_size")),
              static_cast<long long>(saans::kFeatureFrameSize), "frame_size == 68");
    check_u32(static_cast<std::uint32_t>(g.scalar("magic")), saans::kFeatureMagic,
              "magic == 0x534E4153");
    check_i64(static_cast<long long>(bytes.size()),
              static_cast<long long>(frames * saans::kFeatureFrameSize),
              "bytes array is frames * 68 long");
    end_case();

    for (std::size_t i = 0; i < frames; ++i) {
        char label[80];
        std::snprintf(label, sizeof label, "encode(frame %zu) is byte-identical", i);
        begin_case(label);

        const saans::FeatureFrame src = frame_from_golden(g, i);
        std::uint8_t out[saans::kFeatureFrameSize];
        std::memset(out, 0xAA, sizeof out);  // every byte must be written
        saans::encode_feature_frame(src, out);

        bool all_equal = true;
        for (std::size_t b = 0; b < saans::kFeatureFrameSize; ++b) {
            const std::uint8_t want = static_cast<std::uint8_t>(bytes[i * saans::kFeatureFrameSize + b]);
            if (out[b] != want) {
                all_equal = false;
                std::printf("      byte %2zu: got 0x%02X want 0x%02X\n", b, out[b], want);
            }
        }
        check(all_equal, "all 68 bytes match the golden vector");

        // The CRC in the vector is the reference's zlib.crc32 over bytes 0..63.
        const std::uint32_t want_crc =
            static_cast<std::uint32_t>(bytes[i * saans::kFeatureFrameSize + 64]) |
            (static_cast<std::uint32_t>(bytes[i * saans::kFeatureFrameSize + 65]) << 8) |
            (static_cast<std::uint32_t>(bytes[i * saans::kFeatureFrameSize + 66]) << 16) |
            (static_cast<std::uint32_t>(bytes[i * saans::kFeatureFrameSize + 67]) << 24);
        check_u32(saans::crc32(out, saans::kFeatureBodySize), want_crc,
                  "CRC over bytes 0..63 matches the vector");
        end_case();

        std::snprintf(label, sizeof label, "decode(frame %zu) returns the exact fields", i);
        begin_case(label);
        saans::FeatureFrame back;
        const saans::DecodeError err = saans::decode_feature_frame(out, sizeof out, back);
        if (check(err == saans::DecodeError::kOk, "decode accepts the golden bytes")) {
            check_i64(back.version, saans::kFeatureVersion, "version");
            check_i64(back.node_id, src.node_id, "node_id");
            check_i64(back.tx_id, src.tx_id, "tx_id");
            check_i64(back.flags, src.flags, "flags");
            check_u32(back.seq, src.seq, "seq");
            check_u32(back.uptime_ms, src.uptime_ms, "uptime_ms");
            check_i64(back.csi_rate_chz, src.csi_rate_chz, "csi_rate_chz");
            check_i64(back.loss_permille, src.loss_permille, "loss_permille");
            check_f32(back.motion_energy, src.motion_energy, "motion_energy");
            check_f32(back.presence_threshold, src.presence_threshold, "presence_threshold");
            check_f32(back.gross_motion, src.gross_motion, "gross_motion");
            check_f32(back.breath_rms, src.breath_rms, "breath_rms");
            check_f32(back.breathing_bpm, src.breathing_bpm, "breathing_bpm");
            check_f32(back.breathing_conf, src.breathing_conf, "breathing_conf");
            check_f32(back.heart_bpm, src.heart_bpm, "heart_bpm");
            check_f32(back.heart_conf, src.heart_conf, "heart_conf");
            check_f32(back.best_bnr, src.best_bnr, "best_bnr");
            check_u32(back.calib_age_s, src.calib_age_s, "calib_age_s");
            check_i64(back.rssi_dbm, src.rssi_dbm, "rssi_dbm");
            check_i64(back.noise_floor_dbm, src.noise_floor_dbm, "noise_floor_dbm");
            check_i64(back.n_subcarriers, src.n_subcarriers, "n_subcarriers");
        }

        // encode(decode(x)) == x, over the golden bytes themselves.
        std::uint8_t again[saans::kFeatureFrameSize];
        saans::encode_feature_frame(back, again);
        check(std::memcmp(again, out, sizeof out) == 0, "round trip encode(decode(bytes)) == bytes");
        end_case();
    }
}

// ---------------------------------------------------------------------------
// 3. Decode validates
// ---------------------------------------------------------------------------

void test_decode_rejects() {
    begin_case("decode rejects malformed datagrams");

    saans::FeatureFrame src;
    src.node_id = 3;
    src.tx_id = 7;
    src.flags = saans::kFlagCalibrated | saans::kFlagOccupied;
    src.seq = 42;
    src.motion_energy = 0.25f;
    std::uint8_t good[saans::kFeatureFrameSize];
    saans::encode_feature_frame(src, good);

    saans::FeatureFrame out;
    check(saans::decode_feature_frame(good, sizeof good, out) == saans::DecodeError::kOk,
          "a well-formed frame decodes");

    // Length: 68 and only 68. A short read or a coalesced datagram is not a
    // frame with a missing tail, it is not a frame.
    check(saans::decode_feature_frame(good, 67, out) == saans::DecodeError::kBadLength,
          "67 bytes -> kBadLength");
    check(saans::decode_feature_frame(good, 69, out) == saans::DecodeError::kBadLength,
          "69 bytes -> kBadLength");
    check(saans::decode_feature_frame(good, 0, out) == saans::DecodeError::kBadLength,
          "0 bytes -> kBadLength");

    // A flipped payload bit, with the CRC left alone.
    std::uint8_t corrupt_body[saans::kFeatureFrameSize];
    std::memcpy(corrupt_body, good, sizeof good);
    corrupt_body[20] ^= 0x01u;
    check(saans::decode_feature_frame(corrupt_body, sizeof corrupt_body, out) ==
              saans::DecodeError::kBadCrc,
          "flipped payload bit -> kBadCrc");

    // A flipped CRC bit, with the payload left alone.
    std::uint8_t corrupt_crc[saans::kFeatureFrameSize];
    std::memcpy(corrupt_crc, good, sizeof good);
    corrupt_crc[64] ^= 0x80u;
    check(saans::decode_feature_frame(corrupt_crc, sizeof corrupt_crc, out) ==
              saans::DecodeError::kBadCrc,
          "flipped CRC bit -> kBadCrc");

    // Bad magic, CRC recomputed so it is the magic that is rejected and not
    // the checksum. This is the case that matters on a shared port: an
    // raw CSI debug datagram must not be mistaken for a FeatureFrame.
    std::uint8_t bad_magic[saans::kFeatureFrameSize];
    std::memcpy(bad_magic, good, sizeof good);
    bad_magic[0] = 0x01u;
    bad_magic[1] = 0x00u;
    bad_magic[2] = 0x11u;
    bad_magic[3] = 0xC5u;  // 0xC5110001, the raw CSI magic
    saans::detail::put_u32(bad_magic + 64, saans::crc32(bad_magic, saans::kFeatureBodySize));
    check(saans::decode_feature_frame(bad_magic, sizeof bad_magic, out) ==
              saans::DecodeError::kBadMagic,
          "raw-CSI magic -> kBadMagic");

    // Wrong version, CRC recomputed.
    std::uint8_t bad_version[saans::kFeatureFrameSize];
    std::memcpy(bad_version, good, sizeof good);
    bad_version[4] = 2;
    saans::detail::put_u32(bad_version + 64, saans::crc32(bad_version, saans::kFeatureBodySize));
    check(saans::decode_feature_frame(bad_version, sizeof bad_version, out) ==
              saans::DecodeError::kBadVersion,
          "version 2 -> kBadVersion");
    bad_version[4] = 0;
    saans::detail::put_u32(bad_version + 64, saans::crc32(bad_version, saans::kFeatureBodySize));
    check(saans::decode_feature_frame(bad_version, sizeof bad_version, out) ==
              saans::DecodeError::kBadVersion,
          "version 0 -> kBadVersion");

    // The std::optional form says the same thing.
    check(saans::decode_feature_frame(good, sizeof good).has_value(), "optional form accepts");
    check(!saans::decode_feature_frame(corrupt_crc, sizeof corrupt_crc).has_value(),
          "optional form rejects a bad CRC");

    // A rejected frame leaves the caller's frame untouched.
    saans::FeatureFrame untouched;
    untouched.node_id = 99;
    (void)saans::decode_feature_frame(corrupt_crc, sizeof corrupt_crc, untouched);
    check_i64(untouched.node_id, 99, "a rejected decode does not write to out");
    end_case();
}

// ---------------------------------------------------------------------------
// 4. The consent gate (ADR-001, ADR-005)
// ---------------------------------------------------------------------------

/// A frame with every gated field loudly non-zero and every "valid" flag set,
/// so that a gate that fails to clear one is visible on the wire.
saans::FeatureFrame loud_frame() {
    saans::FeatureFrame f;
    f.node_id = 3;
    f.tx_id = 7;
    f.flags = saans::kFlagCalibrated | saans::kFlagOccupied | saans::kFlagVitalsEnabled |
              saans::kFlagBreathingValid | saans::kFlagHeartValid;
    f.motion_energy = 0.125f;
    f.presence_threshold = 0.0625f;
    f.gross_motion = 0.5f;
    f.breath_rms = 0.2f;
    f.breathing_bpm = 15.25f;
    f.breathing_conf = 0.93f;
    f.heart_bpm = 68.5f;
    f.heart_conf = 0.41f;
    f.best_bnr = 42.0f;
    f.n_subcarriers = 56;
    return f;
}

/// What a gateway would see: decode the bytes, do not trust the struct.
saans::FeatureFrame on_the_wire(const saans::FeatureFrame& f, saans::VitalsConsent consent) {
    std::uint8_t buf[saans::kFeatureFrameSize];
    saans::encode_feature_frame_gated(f, consent, buf);
    saans::FeatureFrame out;
    if (saans::decode_feature_frame(buf, sizeof buf, out) != saans::DecodeError::kOk) {
        check(false, "gated frame failed to decode");
    }
    return out;
}

void test_gate_vitals_disabled() {
    begin_case("gate: vitals disabled zeroes all five fields");
    const saans::FeatureFrame wire =
        on_the_wire(loud_frame(), saans::VitalsConsent{false, false});
    check_f32(wire.breath_rms, 0.0f, "breath_rms is 0.0 on the wire");
    check_f32(wire.breathing_bpm, 0.0f, "breathing_bpm is 0.0 on the wire");
    check_f32(wire.breathing_conf, 0.0f, "breathing_conf is 0.0 on the wire");
    check_f32(wire.heart_bpm, 0.0f, "heart_bpm is 0.0 on the wire");
    check_f32(wire.heart_conf, 0.0f, "heart_conf is 0.0 on the wire");
    check(!wire.flag(saans::kFlagVitalsEnabled), "vitals_enabled flag cleared");
    check(!wire.flag(saans::kFlagBreathingValid), "breathing_valid flag cleared");
    check(!wire.flag(saans::kFlagHeartValid), "heart_valid flag cleared");
    // Presence and link quality are not health data and must survive.
    check_f32(wire.motion_energy, 0.125f, "motion_energy survives the gate");
    check_f32(wire.presence_threshold, 0.0625f, "presence_threshold survives the gate");
    check_f32(wire.gross_motion, 0.5f, "gross_motion survives the gate");
    check_f32(wire.best_bnr, 42.0f, "best_bnr survives the gate");
    check(wire.flag(saans::kFlagOccupied), "occupied flag survives the gate");
    check(wire.flag(saans::kFlagCalibrated), "calibrated flag survives the gate");
    end_case();
}

void test_gate_heart_separately() {
    // Heart gating is tested on its own because it has its own switch: a space
    // can consent to breathing and not to heart, and heart has no flag of its
    // own on the wire to infer the setting from.
    begin_case("gate: vitals on, heart off, zeroes only heart");
    const saans::FeatureFrame wire = on_the_wire(loud_frame(), saans::VitalsConsent{true, false});
    check_f32(wire.heart_bpm, 0.0f, "heart_bpm is 0.0 on the wire");
    check_f32(wire.heart_conf, 0.0f, "heart_conf is 0.0 on the wire");
    check(!wire.flag(saans::kFlagHeartValid), "heart_valid flag cleared");
    check_f32(wire.breath_rms, 0.2f, "breath_rms survives");
    check_f32(wire.breathing_bpm, 15.25f, "breathing_bpm survives");
    check_f32(wire.breathing_conf, 0.93f, "breathing_conf survives");
    check(wire.flag(saans::kFlagVitalsEnabled), "vitals_enabled flag survives");
    check(wire.flag(saans::kFlagBreathingValid), "breathing_valid flag survives");
    end_case();

    begin_case("gate: heart on without vitals is still off");
    // Consent to heart alone is not consent: the reference computes
    // `heart = vitals_enabled and heart_enabled`, and so does the gate.
    const saans::FeatureFrame half = on_the_wire(loud_frame(), saans::VitalsConsent{false, true});
    check_f32(half.heart_bpm, 0.0f, "heart_bpm is 0.0 without vitals consent");
    check_f32(half.heart_conf, 0.0f, "heart_conf is 0.0 without vitals consent");
    check_f32(half.breath_rms, 0.0f, "breath_rms is 0.0 without vitals consent");
    check(!half.flag(saans::kFlagHeartValid), "heart_valid flag cleared");
    end_case();

    begin_case("gate: full consent changes nothing");
    const saans::FeatureFrame full = on_the_wire(loud_frame(), saans::VitalsConsent{true, true});
    const saans::FeatureFrame src = loud_frame();
    check_f32(full.breath_rms, src.breath_rms, "breath_rms unchanged");
    check_f32(full.breathing_bpm, src.breathing_bpm, "breathing_bpm unchanged");
    check_f32(full.breathing_conf, src.breathing_conf, "breathing_conf unchanged");
    check_f32(full.heart_bpm, src.heart_bpm, "heart_bpm unchanged");
    check_f32(full.heart_conf, src.heart_conf, "heart_conf unchanged");
    check_i64(full.flags, src.flags, "flags unchanged");
    end_case();
}

void test_gate_properties() {
    begin_case("gate: only ever clears, and is idempotent");
    saans::FeatureFrame once = loud_frame();
    saans::apply_consent_gate(once, saans::VitalsConsent{false, false});
    saans::FeatureFrame twice = once;
    saans::apply_consent_gate(twice, saans::VitalsConsent{false, false});
    std::uint8_t a[saans::kFeatureFrameSize];
    std::uint8_t b[saans::kFeatureFrameSize];
    saans::encode_feature_frame(once, a);
    saans::encode_feature_frame(twice, b);
    check(std::memcmp(a, b, sizeof a) == 0, "gating twice is the same as gating once");

    // Consent does not conjure a flag the extractor did not set.
    saans::FeatureFrame quiet;  // all defaults: flags 0, vitals fields 0
    saans::apply_consent_gate(quiet, saans::VitalsConsent{true, true});
    check_i64(quiet.flags, 0, "full consent sets no flag by itself");
    check_f32(quiet.breathing_bpm, 0.0f, "full consent invents no breathing rate");
    check_f32(quiet.heart_bpm, 0.0f, "full consent invents no heart rate");

    // A frame already carrying vitals, re-gated with consent, is unchanged:
    // the gate is a filter, not a policy engine.
    saans::FeatureFrame keep = loud_frame();
    saans::apply_consent_gate(keep, saans::VitalsConsent{true, true});
    check_f32(keep.heart_conf, 0.41f, "consented heart_conf is left alone");
    end_case();
}

// ---------------------------------------------------------------------------
// 5. Clamping, exactly where Python clamps
// ---------------------------------------------------------------------------

void test_clamping() {
    begin_case("encode clamps the integer fields like the reference");
    saans::FeatureFrame f;
    f.csi_rate_chz = 70000;       // > 0xFFFF
    f.loss_permille = -5;         // < 0
    f.rssi_dbm = 200;             // > 127
    f.noise_floor_dbm = -5000;    // < -128
    f.n_subcarriers = 300;        // > 255
    std::uint8_t buf[saans::kFeatureFrameSize];
    saans::encode_feature_frame(f, buf);
    saans::FeatureFrame wire;
    check(saans::decode_feature_frame(buf, sizeof buf, wire) == saans::DecodeError::kOk,
          "clamped frame decodes");
    check_i64(wire.csi_rate_chz, 0xFFFF, "csi_rate_chz clamps to 0xFFFF");
    check_i64(wire.loss_permille, 0, "loss_permille clamps to 0");
    check_i64(wire.rssi_dbm, 127, "rssi_dbm clamps to 127");
    check_i64(wire.noise_floor_dbm, -128, "noise_floor_dbm clamps to -128");
    check_i64(wire.n_subcarriers, 255, "n_subcarriers clamps to 255");

    f.csi_rate_chz = -1;
    f.loss_permille = 70000;
    f.rssi_dbm = -200;
    f.noise_floor_dbm = 500;
    f.n_subcarriers = -1;
    saans::encode_feature_frame(f, buf);
    check(saans::decode_feature_frame(buf, sizeof buf, wire) == saans::DecodeError::kOk,
          "clamped frame decodes (other end)");
    check_i64(wire.csi_rate_chz, 0, "csi_rate_chz clamps to 0");
    check_i64(wire.loss_permille, 0xFFFF, "loss_permille clamps to 0xFFFF");
    check_i64(wire.rssi_dbm, -128, "rssi_dbm clamps to -128");
    check_i64(wire.noise_floor_dbm, 127, "noise_floor_dbm clamps to 127");
    check_i64(wire.n_subcarriers, 0, "n_subcarriers clamps to 0");

    // In range, nothing moves.
    f.csi_rate_chz = 1987;
    f.loss_permille = 31;
    f.rssi_dbm = -61;
    f.noise_floor_dbm = -95;
    f.n_subcarriers = 56;
    saans::encode_feature_frame(f, buf);
    check(saans::decode_feature_frame(buf, sizeof buf, wire) == saans::DecodeError::kOk,
          "in-range frame decodes");
    check_i64(wire.csi_rate_chz, 1987, "csi_rate_chz passes through");
    check_i64(wire.rssi_dbm, -61, "negative rssi_dbm round-trips through the i8");
    check_i64(wire.noise_floor_dbm, -95, "negative noise_floor_dbm round-trips through the i8");
    end_case();
}

// ---------------------------------------------------------------------------
// 6. Layout the spec fixes by hand
// ---------------------------------------------------------------------------

void test_wire_layout() {
    begin_case("wire layout: magic bytes, version, reserved, wraparound");
    saans::FeatureFrame f;
    f.seq = 0xFFFFFFFFu;
    f.uptime_ms = 0xFFFFFFFFu;
    f.calib_age_s = saans::kNotCalibrated;
    std::uint8_t buf[saans::kFeatureFrameSize];
    std::memset(buf, 0x5A, sizeof buf);
    saans::encode_feature_frame(f, buf);
    check(buf[0] == 'S' && buf[1] == 'A' && buf[2] == 'N' && buf[3] == 'S',
          "bytes 0..3 spell \"SANS\" on the wire");
    check_i64(buf[4], saans::kFeatureVersion, "byte 4 is the version");
    check_i64(buf[63], 0, "byte 63 (reserved) is written as 0");
    check_i64(buf[6], 0xFF, "default tx_id is 0xFF (AP / unknown)");

    saans::FeatureFrame wire;
    check(saans::decode_feature_frame(buf, sizeof buf, wire) == saans::DecodeError::kOk,
          "decodes");
    check_u32(wire.seq, 0xFFFFFFFFu, "seq survives at the wrap point");
    check_u32(wire.uptime_ms, 0xFFFFFFFFu, "uptime_ms survives at the wrap point");
    check_u32(wire.calib_age_s, saans::kNotCalibrated, "calib_age_s 0xFFFFFFFF = not calibrated");
    check_i64(wire.noise_floor_dbm, saans::kNoiseUnknown, "default noise floor is -128 (unknown)");

    // The fixed-size overload exists and agrees with the pointer/length one.
    saans::FeatureFrame via_array;
    check(saans::decode_feature_frame(buf, via_array) == saans::DecodeError::kOk,
          "fixed-size overload decodes");
    end_case();
}

}  // namespace

int main(int argc, char** argv) {
    // argv[1], then $SAANS_GOLDEN_DIR, then the build's -D. Both Makefiles in
    // this tree drive every test the same way; golden.hpp owns that order.
    saans::golden::dir_from_argv(argc, argv);
    std::printf("test_feature_frame  (vectors: %s)\n", saans::golden::dir().c_str());

    const saans::golden::Vectors codec("feature_codec");

    test_crc32_known_answers();
    test_golden_bytes(codec);
    test_decode_rejects();
    test_gate_vitals_disabled();
    test_gate_heart_separately();
    test_gate_properties();
    test_clamping();
    test_wire_layout();

    std::printf("\n%d cases, %d checks, %d failed\n", g_cases, g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
