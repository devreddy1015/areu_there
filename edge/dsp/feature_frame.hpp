// FeatureFrame v1: what a Saans sensor node sends instead of raw CSI.
//
// C++17 port of `wifisense/features.py` (the encode/decode half of it; the
// extractor that fills these fields lives elsewhere). Authoritative format:
// spec/wire.md §2. 68 bytes, little-endian, magic "SANS".
//
// Why features and not raw CSI (ADR-001): raw CSI is enough to re-derive
// breathing, heartbeat and, with more nodes, movement through walls. Sending
// it to a server makes the server a store of health data whether anyone meant
// it to be or not. A 68-byte frame twice a second carries exactly what the
// health detectors need, with breathing- and heart-band fields zeroed on the
// node unless the space has opted in -- see `apply_consent_gate` below.
//
// Nothing here allocates, throws, or touches a stream: the caller owns the
// 68 bytes. Fields are written one at a time at their documented offsets
// rather than by memcpy-ing a packed struct, so the wire format is enforced
// by this code and not by whatever layout the compiler happened to choose.
// Every multi-byte store is an explicit little-endian byte sequence, so the
// host test and the Xtensa target produce the same bytes without either of
// them relying on its own endianness. (Floats go out through their integer
// bit pattern, which assumes float and integer byte order agree -- true on
// x86, ARM and the ESP32-S3 alike.)
//
// Also FeatureFrame v2 (ADR-012, spec/wire.md §2b): the same fields, signed
// with the node's key so a gateway can tell its frames from anyone else's,
// with a boot counter so it can tell them from old ones. See the end of the
// file; the HMAC is `sha256.hpp`.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>

#include "crc32.hpp"
#include "sha256.hpp"

namespace saans {

constexpr std::uint32_t kFeatureMagic = 0x534E4153u;  // bytes on the wire: "SANS"
constexpr std::uint8_t kFeatureVersion = 1;
constexpr std::size_t kFeatureFrameSize = 68;
constexpr std::size_t kFeatureBodySize = 64;  // CRC covers bytes 0..63

// The sizes are spec, not implementation detail: a frame that is not 68 bytes
// is not a FeatureFrame v1, whatever else it may be.
static_assert(kFeatureFrameSize == 68, "spec/wire.md §2: FeatureFrame v1 is 68 bytes");
static_assert(kFeatureBodySize + sizeof(std::uint32_t) == kFeatureFrameSize,
              "the CRC is the last four bytes of the frame");

// FeatureFrame v2 (spec/wire.md §2b): v1's bytes 0..62 with version 2,
// key_id in v1's reserved byte, a u32 boot counter, then a 16-byte tag.
constexpr std::uint8_t kFeatureVersionSigned = 2;
constexpr std::size_t kSignedFrameSize = 84;
constexpr std::size_t kSignedBodySize = 68;  // the tag covers bytes 0..67
constexpr std::size_t kTagSize = 16;
constexpr std::size_t kFrameKeySize = 32;
static_assert(kSignedBodySize + kTagSize == kSignedFrameSize, "spec/wire.md §2b: 68 + 16 = 84");
static_assert(kTagSize <= kSha256Size, "the tag is a truncated HMAC-SHA256");

/// `calib_age_s` when presence calibration has never completed.
constexpr std::uint32_t kNotCalibrated = 0xFFFFFFFFu;
/// `noise_floor_dbm` when the radio does not report one.
constexpr std::int32_t kNoiseUnknown = -128;

// Flags, byte 7 (spec/wire.md §2). Bit 7 is reserved and always 0.
constexpr std::uint8_t kFlagCalibrated = 0x01;      // bit 0: empty-room threshold exists
constexpr std::uint8_t kFlagOccupied = 0x02;        // bit 1: presence detector says occupied
constexpr std::uint8_t kFlagVitalsEnabled = 0x04;   // bit 2: this space opted in (ADR-005)
constexpr std::uint8_t kFlagRawMode = 0x08;         // bit 3: node is also streaming raw CSI
constexpr std::uint8_t kFlagBreathingValid = 0x10;  // bit 4: breathing VALID or DEGRADED
constexpr std::uint8_t kFlagHeartValid = 0x20;      // bit 5: same for heart, heart enabled
constexpr std::uint8_t kFlagDiscontinuity = 0x40;   // bit 6: gap, reboot or reset since last frame

/// One link's feature frame, in the order the bytes go out.
///
/// The widths are the Python semantics, not the wire widths: the fields the
/// reference clamps rather than masks (`csi_rate_chz`, `loss_permille`,
/// `rssi_dbm`, `noise_floor_dbm`, `n_subcarriers`) are held wide and signed so
/// that an out-of-range value survives to `encode_feature_frame`, which clamps
/// it exactly where Python's `max(lo, min(hi, x))` does. The fields Python
/// masks (`& 0xFF`, `& 0xFFFFFFFF`) are held at their wire width, so the mask
/// is the type.
struct FeatureFrame {
    std::uint8_t version = kFeatureVersion;
    std::uint8_t node_id = 0;
    std::uint8_t tx_id = 0xFF;  // 0xFF = an access point / unknown
    std::uint8_t flags = 0;
    std::uint32_t seq = 0;
    std::uint32_t uptime_ms = 0;
    std::int32_t csi_rate_chz = 0;   // Hz x 100, clamped to [0, 0xFFFF] on encode
    std::int32_t loss_permille = 0;  // clamped to [0, 0xFFFF] on encode
    float motion_energy = 0.0f;
    float presence_threshold = 0.0f;
    float gross_motion = 0.0f;
    float breath_rms = 0.0f;       // vitals-gated
    float breathing_bpm = 0.0f;    // vitals-gated
    float breathing_conf = 0.0f;   // vitals-gated
    float heart_bpm = 0.0f;        // vitals- and heart-gated
    float heart_conf = 0.0f;       // vitals- and heart-gated
    float best_bnr = 0.0f;
    std::uint32_t calib_age_s = kNotCalibrated;
    std::int32_t rssi_dbm = 0;                    // clamped to [-128, 127] on encode
    std::int32_t noise_floor_dbm = kNoiseUnknown; // clamped to [-128, 127] on encode
    std::int32_t n_subcarriers = 0;               // clamped to [0, 255] on encode

    constexpr bool flag(std::uint8_t bit) const noexcept { return (flags & bit) != 0; }
};

// ---------------------------------------------------------------------------
// The consent gate (ADR-001, ADR-005, spec/wire.md "Gating rule")
// ---------------------------------------------------------------------------

/// What this link's space has opted in to. Configuration on the node, not a
/// wire field: `heart_enabled` has no flag bit of its own, which is why the
/// gate cannot live inside `encode_feature_frame` and is a separate step.
struct VitalsConsent {
    bool vitals_enabled = false;
    bool heart_enabled = false;
};

/// Zero every field derived from the breathing or heart band unless the space
/// consented to it. THIS IS THE PRIVACY BOUNDARY: it runs on the node, before
/// the bytes exist, so a gateway never receives what it is not allowed to
/// hold. Nothing downstream can undo a mistake made here.
///
/// Gated by `vitals_enabled`: breath_rms, breathing_bpm, breathing_conf,
/// heart_bpm, heart_conf, and the flags that assert those readings are usable.
/// `heart_bpm` and `heart_conf` additionally need `heart_enabled`.
///
/// Invariant a reviewer can check in one read: this function only ever
/// *clears*. It sets no flag and writes no non-zero value, so calling it twice
/// changes nothing, calling it on an already-gated frame changes nothing, and
/// calling it when it was not needed cannot leak anything that was not already
/// there. `motion_energy`, `presence_threshold`, `gross_motion` and
/// `best_bnr` are deliberately not gated: they are presence and link quality,
/// which every space needs for the safety detectors (SAANS_PLAN §2.1), and
/// none of them is derived from the breathing or heart band.
inline void apply_consent_gate(FeatureFrame& frame, VitalsConsent consent) noexcept {
    const bool heart_ok = consent.vitals_enabled && consent.heart_enabled;

    if (!consent.vitals_enabled) {
        frame.breath_rms = 0.0f;
        frame.breathing_bpm = 0.0f;
        frame.breathing_conf = 0.0f;
        // The flags are health data too: "breathing detected" is a statement
        // about a body even without a rate attached.
        frame.flags = static_cast<std::uint8_t>(
            frame.flags & static_cast<std::uint8_t>(~kFlagVitalsEnabled));
        frame.flags = static_cast<std::uint8_t>(
            frame.flags & static_cast<std::uint8_t>(~kFlagBreathingValid));
    }
    if (!heart_ok) {
        frame.heart_bpm = 0.0f;
        frame.heart_conf = 0.0f;
        frame.flags = static_cast<std::uint8_t>(
            frame.flags & static_cast<std::uint8_t>(~kFlagHeartValid));
    }
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

namespace detail {

/// Python's `max(lo, min(hi, x))`, with the same saturating result.
constexpr std::int32_t clamp_i32(std::int32_t x, std::int32_t lo, std::int32_t hi) noexcept {
    return x < lo ? lo : (x > hi ? hi : x);
}

inline void put_u8(std::uint8_t* p, std::uint8_t v) noexcept { p[0] = v; }

inline void put_u16(std::uint8_t* p, std::uint16_t v) noexcept {
    p[0] = static_cast<std::uint8_t>(v & 0xFFu);
    p[1] = static_cast<std::uint8_t>((v >> 8) & 0xFFu);
}

inline void put_u32(std::uint8_t* p, std::uint32_t v) noexcept {
    p[0] = static_cast<std::uint8_t>(v & 0xFFu);
    p[1] = static_cast<std::uint8_t>((v >> 8) & 0xFFu);
    p[2] = static_cast<std::uint8_t>((v >> 16) & 0xFFu);
    p[3] = static_cast<std::uint8_t>((v >> 24) & 0xFFu);
}

inline std::uint16_t get_u16(const std::uint8_t* p) noexcept {
    return static_cast<std::uint16_t>(static_cast<std::uint32_t>(p[0]) |
                                      (static_cast<std::uint32_t>(p[1]) << 8));
}

inline std::uint32_t get_u32(const std::uint8_t* p) noexcept {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

static_assert(sizeof(float) == 4, "FeatureFrame assumes IEEE-754 binary32 floats");

/// The bit pattern of a float, without type-punning through a union or a
/// reinterpret_cast (both of which are UB here; memcpy of 4 bytes compiles to
/// a register move). NaN payloads and signed zero survive unchanged, which is
/// what byte-exactness against the reference requires.
inline std::uint32_t f32_bits(float v) noexcept {
    std::uint32_t bits;
    std::memcpy(&bits, &v, sizeof bits);
    return bits;
}

inline float bits_f32(std::uint32_t bits) noexcept {
    float v;
    std::memcpy(&v, &bits, sizeof v);
    return v;
}

inline void put_f32(std::uint8_t* p, float v) noexcept { put_u32(p, f32_bits(v)); }

inline float get_f32(const std::uint8_t* p) noexcept { return bits_f32(get_u32(p)); }

}  // namespace detail

namespace detail {
inline void encode_fields(const FeatureFrame& f, std::uint8_t version, std::uint8_t last,
                          std::uint8_t* out) noexcept;
}  // namespace detail

/// Write one frame into a caller-provided 68-byte buffer.
///
/// Byte-for-byte what `FeatureFrame.encode()` in the reference produces,
/// including the clamps; the `feature_codec` golden vector is the proof.
/// This does NOT apply the consent gate -- `heart_enabled` is not on the wire,
/// so the gate cannot be inferred from the frame. Producers call
/// `apply_consent_gate` first, or use `encode_feature_frame_gated`.
inline void encode_feature_frame(const FeatureFrame& f, std::uint8_t* out) noexcept {
    detail::encode_fields(f, f.version, 0, out);  // byte 63 is reserved, spec says 0
    detail::put_u32(out + 64, crc32(out, kFeatureBodySize));
}

namespace detail {

/// Bytes 0..63, which v1 and v2 share. The last is v1's `reserved` (0) or
/// v2's `key_id`.
inline void encode_fields(const FeatureFrame& f, std::uint8_t version, std::uint8_t last,
                          std::uint8_t* out) noexcept {
    detail::put_u32(out + 0, kFeatureMagic);
    detail::put_u8(out + 4, version);
    detail::put_u8(out + 5, f.node_id);  // the type is Python's `& 0xFF`
    detail::put_u8(out + 6, f.tx_id);
    detail::put_u8(out + 7, f.flags);
    detail::put_u32(out + 8, f.seq);
    detail::put_u32(out + 12, f.uptime_ms);
    detail::put_u16(out + 16,
                    static_cast<std::uint16_t>(detail::clamp_i32(f.csi_rate_chz, 0, 0xFFFF)));
    detail::put_u16(out + 18,
                    static_cast<std::uint16_t>(detail::clamp_i32(f.loss_permille, 0, 0xFFFF)));
    detail::put_f32(out + 20, f.motion_energy);
    detail::put_f32(out + 24, f.presence_threshold);
    detail::put_f32(out + 28, f.gross_motion);
    detail::put_f32(out + 32, f.breath_rms);
    detail::put_f32(out + 36, f.breathing_bpm);
    detail::put_f32(out + 40, f.breathing_conf);
    detail::put_f32(out + 44, f.heart_bpm);
    detail::put_f32(out + 48, f.heart_conf);
    detail::put_f32(out + 52, f.best_bnr);
    detail::put_u32(out + 56, f.calib_age_s);
    detail::put_u8(out + 60, static_cast<std::uint8_t>(detail::clamp_i32(f.rssi_dbm, -128, 127)));
    detail::put_u8(out + 61,
                   static_cast<std::uint8_t>(detail::clamp_i32(f.noise_floor_dbm, -128, 127)));
    detail::put_u8(out + 62, static_cast<std::uint8_t>(detail::clamp_i32(f.n_subcarriers, 0, 255)));
    detail::put_u8(out + 63, last);
}

}  // namespace detail

/// Gate, then encode. The one call a sender should reach for: it is not
/// possible to forget the gate and still get bytes out of it.
inline void encode_feature_frame_gated(const FeatureFrame& f, VitalsConsent consent,
                                       std::uint8_t* out) noexcept {
    FeatureFrame gated = f;  // 64 bytes of stack; no allocation
    apply_consent_gate(gated, consent);
    encode_feature_frame(gated, out);
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

/// Why a datagram was rejected. The reference raises ValueError with these
/// same four reasons, checked in this same order.
enum class DecodeError : std::uint8_t {
    kOk = 0,
    kBadLength,   // not exactly 68 bytes
    kBadCrc,      // CRC over bytes 0..63 does not match bytes 64..67
    kBadMagic,    // not 0x534E4153
    kBadVersion,  // not FeatureFrame v1
};

namespace detail {
inline FeatureFrame decode_fields(const std::uint8_t* data) noexcept;
}  // namespace detail

/// Parse and validate one datagram.
///
/// Order matters and matches the reference: length, then CRC, then magic, then
/// version. The CRC comes before the magic deliberately -- a corrupted frame
/// should be reported as corrupt, not as "some other protocol".
/// `out` is left untouched unless the result is `kOk`.
inline DecodeError decode_feature_frame(const std::uint8_t* data, std::size_t len,
                                        FeatureFrame& out) noexcept {
    if (len != kFeatureFrameSize) {
        return DecodeError::kBadLength;
    }
    if (crc32(data, kFeatureBodySize) != detail::get_u32(data + 64)) {
        return DecodeError::kBadCrc;
    }
    if (detail::get_u32(data + 0) != kFeatureMagic) {
        return DecodeError::kBadMagic;
    }
    const std::uint8_t version = data[4];
    if (version != kFeatureVersion) {
        return DecodeError::kBadVersion;
    }

    out = detail::decode_fields(data);
    // byte 63 is reserved; the spec says write 0, and says nothing about
    // rejecting a frame that does not, so a v1.x sender that uses it stays
    // readable here.
    return DecodeError::kOk;
}

namespace detail {

/// Every field from bytes 0..62, which v1 and v2 lay out alike.
inline FeatureFrame decode_fields(const std::uint8_t* data) noexcept {
    FeatureFrame f;
    f.version = data[4];
    f.node_id = data[5];
    f.tx_id = data[6];
    f.flags = data[7];
    f.seq = detail::get_u32(data + 8);
    f.uptime_ms = detail::get_u32(data + 12);
    f.csi_rate_chz = static_cast<std::int32_t>(detail::get_u16(data + 16));
    f.loss_permille = static_cast<std::int32_t>(detail::get_u16(data + 18));
    f.motion_energy = detail::get_f32(data + 20);
    f.presence_threshold = detail::get_f32(data + 24);
    f.gross_motion = detail::get_f32(data + 28);
    f.breath_rms = detail::get_f32(data + 32);
    f.breathing_bpm = detail::get_f32(data + 36);
    f.breathing_conf = detail::get_f32(data + 40);
    f.heart_bpm = detail::get_f32(data + 44);
    f.heart_conf = detail::get_f32(data + 48);
    f.best_bnr = detail::get_f32(data + 52);
    f.calib_age_s = detail::get_u32(data + 56);
    f.rssi_dbm = static_cast<std::int32_t>(static_cast<std::int8_t>(data[60]));
    f.noise_floor_dbm = static_cast<std::int32_t>(static_cast<std::int8_t>(data[61]));
    f.n_subcarriers = static_cast<std::int32_t>(data[62]);
    return f;
}

}  // namespace detail

/// Convenience form for code that would rather branch on a value.
inline std::optional<FeatureFrame> decode_feature_frame(const std::uint8_t* data,
                                                        std::size_t len) noexcept {
    FeatureFrame f;
    if (decode_feature_frame(data, len, f) != DecodeError::kOk) {
        return std::nullopt;
    }
    return f;
}

/// A fixed 68-byte buffer: the length check is then the compiler's job.
inline DecodeError decode_feature_frame(const std::uint8_t (&data)[kFeatureFrameSize],
                                        FeatureFrame& out) noexcept {
    return decode_feature_frame(data, kFeatureFrameSize, out);
}

inline const char* decode_error_str(DecodeError e) noexcept {
    switch (e) {
        case DecodeError::kOk: return "ok";
        case DecodeError::kBadLength: return "wrong length (FeatureFrame is 68 bytes)";
        case DecodeError::kBadCrc: return "CRC mismatch";
        case DecodeError::kBadMagic: return "bad magic (not \"SANS\")";
        case DecodeError::kBadVersion: return "unsupported FeatureFrame version";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// FeatureFrame v2: signed (ADR-012, spec/wire.md §2b)
// ---------------------------------------------------------------------------

/// The key node `node_id` is flashed with, from a site key: HMAC-SHA256(site
/// key, "saans frame key v1" || node_id). A node never holds a site key; this
/// is here so the derivation can be held to the reference's golden keys.
inline void derive_node_key(const std::uint8_t* site_key, std::uint8_t node_id,
                            std::uint8_t* out) noexcept {
    static constexpr char kLabel[] = "saans frame key v1";
    HmacSha256 m(site_key, kFrameKeySize);
    m.update(reinterpret_cast<const std::uint8_t*>(kLabel), sizeof(kLabel) - 1);
    m.update(&node_id, 1);
    m.finish(out);
}

namespace detail {

inline void signed_tag(const std::uint8_t* node_key, const std::uint8_t* body,
                       std::uint8_t* tag) noexcept {
    std::uint8_t full[kSha256Size];
    hmac_sha256(node_key, kFrameKeySize, body, kSignedBodySize, full);
    std::memcpy(tag, full, kTagSize);
}

}  // namespace detail

/// Write one signed frame (84 bytes) into `out`: the fields, `key_id`,
/// `boot`, and the first 16 bytes of HMAC-SHA256 under `node_key` over
/// bytes 0..67. `f.version` is ignored; the result is always version 2.
/// Byte-for-byte `encode_signed` in the reference; the
/// `feature_codec_signed` golden vector is the proof. Like
/// `encode_feature_frame`, this does not apply the consent gate.
inline void encode_signed_frame(const FeatureFrame& f, const std::uint8_t* node_key,
                                std::uint8_t key_id, std::uint32_t boot,
                                std::uint8_t* out) noexcept {
    detail::encode_fields(f, kFeatureVersionSigned, key_id, out);
    detail::put_u32(out + 64, boot);
    detail::signed_tag(node_key, out, out + kSignedBodySize);
}

/// Gate, then sign: the one call a sender should reach for.
inline void encode_signed_frame_gated(const FeatureFrame& f, VitalsConsent consent,
                                      const std::uint8_t* node_key, std::uint8_t key_id,
                                      std::uint32_t boot, std::uint8_t* out) noexcept {
    FeatureFrame gated = f;
    apply_consent_gate(gated, consent);
    encode_signed_frame(gated, node_key, key_id, boot, out);
}

enum class SignedDecodeError : std::uint8_t {
    kOk = 0,
    kBadLength,   // not exactly 84 bytes
    kBadMagic,    // not 0x534E4153
    kBadVersion,  // not FeatureFrame v2
    kBadTag,      // forged, corrupted, or signed with another key
};

/// Check, then parse, one signed frame under `node_key`: length, magic,
/// version, then the tag in constant time, and only then the fields.
/// Choosing the key (by the frame's `key_id` and `node_id`) and the replay
/// floor are the receiver's (the gateway's), not this function's. `out`,
/// `key_id` and `boot` are left untouched unless the result is `kOk`.
inline SignedDecodeError decode_signed_frame(const std::uint8_t* data, std::size_t len,
                                             const std::uint8_t* node_key, FeatureFrame& out,
                                             std::uint8_t& key_id, std::uint32_t& boot) noexcept {
    if (len != kSignedFrameSize) {
        return SignedDecodeError::kBadLength;
    }
    if (detail::get_u32(data + 0) != kFeatureMagic) {
        return SignedDecodeError::kBadMagic;
    }
    if (data[4] != kFeatureVersionSigned) {
        return SignedDecodeError::kBadVersion;
    }
    std::uint8_t want[kTagSize];
    detail::signed_tag(node_key, data, want);
    if (!equal_ct(want, data + kSignedBodySize, kTagSize)) {
        return SignedDecodeError::kBadTag;
    }
    out = detail::decode_fields(data);
    key_id = data[63];
    boot = detail::get_u32(data + 64);
    return SignedDecodeError::kOk;
}

}  // namespace saans
