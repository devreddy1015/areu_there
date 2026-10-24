// FeatureFrame v2: SHA-256, HMAC-SHA256 and signed frames (ADR-012).
//
// Three layers, each against something this code did not write:
//   1. SHA-256 against the FIPS 180-4 / NIST example messages, including one
//      million 'a's (many blocks) and every way of splitting a message into
//      `update` calls (the buffering);
//   2. HMAC-SHA256 against RFC 4231 test cases 1-4, 6 and 7 (6 and 7 have a
//      131-byte key, the hash-the-key-first path);
//   3. the signed codec against the Python reference's `feature_codec_signed`
//      golden vectors, BYTE EQUALITY as for v1: node keys derived, frames
//      signed, decoded back, and every one of the 84 bytes flipped refused.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "feature_frame.hpp"
#include "sha256.hpp"

#include "golden.hpp"

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
    std::printf("  %-48s %s\n", g_case, g_failures == g_case_failures ? "ok" : "FAILED");
}

bool check(bool ok, const std::string& what) {
    g_checks += 1;
    if (!ok) {
        g_failures += 1;
        std::printf("    FAIL [%s] %s\n", g_case, what.c_str());
    }
    return ok;
}

std::string hex(const std::uint8_t* p, std::size_t n) {
    static const char kDigits[] = "0123456789abcdef";
    std::string s;
    for (std::size_t i = 0; i < n; ++i) {
        s += kDigits[p[i] >> 4];
        s += kDigits[p[i] & 0x0F];
    }
    return s;
}

const std::uint8_t* bytes(const char* s) { return reinterpret_cast<const std::uint8_t*>(s); }

bool check_hex(const std::uint8_t* got, std::size_t n, const char* want, const std::string& what) {
    const std::string g = hex(got, n);
    const bool ok = check(g == want, what);
    if (!ok) {
        std::printf("      got  %s\n      want %s\n", g.c_str(), want);
    }
    return ok;
}

// ---------------------------------------------------------------------------
// 1. SHA-256
// ---------------------------------------------------------------------------

void test_sha256_nist() {
    begin_case("SHA-256: NIST example messages");
    struct Case {
        const char* name;
        std::string msg;
        const char* digest;
    };
    const Case cases[] = {
        {"empty", "", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abc", "abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"448-bit", "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
        {"896-bit",
         "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopq"
         "rlmnopqrsmnopqrstnopqrstu",
         "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"},
        {"a x 1,000,000", std::string(1000000, 'a'),
         "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"},
    };
    for (const Case& c : cases) {
        std::uint8_t d[saans::kSha256Size];
        saans::sha256(bytes(c.msg.c_str()), c.msg.size(), d);
        check_hex(d, sizeof d, c.digest, c.name);
    }
    end_case();
}

void test_sha256_any_split() {
    begin_case("SHA-256: every split point gives one digest");
    // 200 bytes crosses three block boundaries; feeding it in two pieces at
    // every possible cut exercises each buffer state the class can be in.
    std::uint8_t msg[200];
    for (std::size_t i = 0; i < sizeof msg; ++i) {
        msg[i] = static_cast<std::uint8_t>(i * 7u + 3u);
    }
    std::uint8_t whole[saans::kSha256Size];
    saans::sha256(msg, sizeof msg, whole);
    bool all = true;
    for (std::size_t cut = 0; cut <= sizeof msg; ++cut) {
        saans::Sha256 s;
        s.update(msg, cut);
        s.update(msg + cut, sizeof msg - cut);
        std::uint8_t d[saans::kSha256Size];
        s.finish(d);
        all = all && std::memcmp(d, whole, sizeof d) == 0;
    }
    check(all, "split anywhere == one call");
    // Lengths around the padding boundary (55, 56, 63, 64 bytes) are where a
    // padding bug shows; the reference digests are Python's hashlib.
    const char* want[] = {
        "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318",  // 55 x 'a'
        "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a",  // 56 x 'a'
        "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34",  // 63 x 'a'
        "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb",  // 64 x 'a'
    };
    const std::size_t lens[] = {55, 56, 63, 64};
    for (int i = 0; i < 4; ++i) {
        const std::string m(lens[i], 'a');
        std::uint8_t d[saans::kSha256Size];
        saans::sha256(bytes(m.c_str()), m.size(), d);
        check_hex(d, sizeof d, want[i], std::to_string(lens[i]) + " x 'a'");
    }
    end_case();
}

// ---------------------------------------------------------------------------
// 2. HMAC-SHA256, RFC 4231
// ---------------------------------------------------------------------------

void test_hmac_rfc4231() {
    begin_case("HMAC-SHA256: RFC 4231 cases 1-4, 6, 7");
    struct Case {
        int n;
        std::vector<std::uint8_t> key;
        std::string data;
        const char* mac;
    };
    std::vector<std::uint8_t> k4(25);
    for (std::size_t i = 0; i < k4.size(); ++i) {
        k4[i] = static_cast<std::uint8_t>(i + 1);
    }
    const Case cases[] = {
        {1, std::vector<std::uint8_t>(20, 0x0b), "Hi There",
         "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"},
        {2, {'J', 'e', 'f', 'e'}, "what do ya want for nothing?",
         "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"},
        {3, std::vector<std::uint8_t>(20, 0xaa), std::string(50, '\xdd'),
         "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe"},
        {4, k4, std::string(50, '\xcd'),
         "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b"},
        {6, std::vector<std::uint8_t>(131, 0xaa),
         "Test Using Larger Than Block-Size Key - Hash Key First",
         "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"},
        {7, std::vector<std::uint8_t>(131, 0xaa),
         "This is a test using a larger than block-size key and a larger than block-size data. "
         "The key needs to be hashed before being used by the HMAC algorithm.",
         "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2"},
    };
    for (const Case& c : cases) {
        std::uint8_t mac[saans::kSha256Size];
        saans::hmac_sha256(c.key.data(), c.key.size(), bytes(c.data.c_str()), c.data.size(), mac);
        check_hex(mac, sizeof mac, c.mac, "RFC 4231 case " + std::to_string(c.n));
    }
    end_case();
}

void test_equal_ct() {
    begin_case("equal_ct");
    const std::uint8_t a[4] = {1, 2, 3, 4};
    const std::uint8_t b[4] = {1, 2, 3, 5};
    check(saans::equal_ct(a, a, 4), "equal");
    check(!saans::equal_ct(a, b, 4), "last byte differs");
    check(saans::equal_ct(a, b, 3), "prefix only");
    end_case();
}

// ---------------------------------------------------------------------------
// 3. Signed frames against the reference
// ---------------------------------------------------------------------------

saans::FeatureFrame frame_from(const saans::golden::Vectors& v, std::size_t i) {
    auto g = [&](const char* k) { return v.array(k)[i]; };
    saans::FeatureFrame f;
    f.node_id = static_cast<std::uint8_t>(g("node_id"));
    f.tx_id = static_cast<std::uint8_t>(g("tx_id"));
    f.flags = static_cast<std::uint8_t>(g("flags"));
    f.seq = static_cast<std::uint32_t>(g("seq"));
    f.uptime_ms = static_cast<std::uint32_t>(g("uptime_ms"));
    f.csi_rate_chz = static_cast<std::int32_t>(g("csi_rate_chz"));
    f.loss_permille = static_cast<std::int32_t>(g("loss_permille"));
    f.motion_energy = static_cast<float>(g("motion_energy"));
    f.presence_threshold = static_cast<float>(g("presence_threshold"));
    f.gross_motion = static_cast<float>(g("gross_motion"));
    f.breath_rms = static_cast<float>(g("breath_rms"));
    f.breathing_bpm = static_cast<float>(g("breathing_bpm"));
    f.breathing_conf = static_cast<float>(g("breathing_conf"));
    f.heart_bpm = static_cast<float>(g("heart_bpm"));
    f.heart_conf = static_cast<float>(g("heart_conf"));
    f.best_bnr = static_cast<float>(g("best_bnr"));
    f.calib_age_s = static_cast<std::uint32_t>(g("calib_age_s"));
    f.rssi_dbm = static_cast<std::int32_t>(g("rssi_dbm"));
    f.noise_floor_dbm = static_cast<std::int32_t>(g("noise_floor_dbm"));
    f.n_subcarriers = static_cast<std::int32_t>(g("n_subcarriers"));
    return f;
}

std::vector<std::uint8_t> byte_array(const saans::golden::Vectors& v, const char* k) {
    const std::vector<double>& a = v.array(k);
    std::vector<std::uint8_t> out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = static_cast<std::uint8_t>(a[i]);
    }
    return out;
}

void test_golden_signed(const saans::golden::Vectors& v) {
    begin_case("signed frames: byte-exact vs reference");
    const std::size_t n = static_cast<std::size_t>(v.integer("frames"));
    check(static_cast<std::size_t>(v.integer("frame_size")) == saans::kSignedFrameSize,
          "frame_size is 84");
    const std::vector<std::uint8_t> site = byte_array(v, "site_key");
    const std::vector<std::uint8_t> node_keys = byte_array(v, "node_keys");
    const std::vector<std::uint8_t> want = byte_array(v, "bytes");
    const std::vector<double>& key_ids = v.array("key_id");
    const std::vector<double>& boots = v.array("boot");
    check(site.size() == 32 && node_keys.size() == 32 * n && want.size() == 84 * n, "sizes");
    for (std::size_t i = 0; i < n; ++i) {
        const std::string tag = "frame " + std::to_string(i);
        const saans::FeatureFrame f = frame_from(v, i);
        std::uint8_t key[saans::kFrameKeySize];
        saans::derive_node_key(site.data(), f.node_id, key);
        check(std::memcmp(key, node_keys.data() + 32 * i, 32) == 0, tag + ": node key");
        const auto key_id = static_cast<std::uint8_t>(key_ids[i]);
        const auto boot = static_cast<std::uint32_t>(boots[i]);
        std::uint8_t out[saans::kSignedFrameSize];
        saans::encode_signed_frame(f, key, key_id, boot, out);
        const std::uint8_t* ref = want.data() + 84 * i;
        if (!check(std::memcmp(out, ref, 84) == 0, tag + ": 84 bytes")) {
            std::printf("      got  %s\n      want %s\n", hex(out, 84).c_str(), hex(ref, 84).c_str());
        }

        saans::FeatureFrame back;
        std::uint8_t got_key_id = 0;
        std::uint32_t got_boot = 0;
        const auto r = saans::decode_signed_frame(ref, 84, key, back, got_key_id, got_boot);
        check(r == saans::SignedDecodeError::kOk, tag + ": the reference's frame verifies");
        check(got_key_id == key_id && got_boot == boot, tag + ": key_id and boot");
        check(back.version == saans::kFeatureVersionSigned, tag + ": version 2");
        std::uint8_t again[saans::kSignedFrameSize];
        saans::encode_signed_frame(back, key, got_key_id, got_boot, again);
        check(std::memcmp(again, ref, 84) == 0, tag + ": decoded fields re-sign byte-exact");

        // Every byte, flipped, is refused, with the reason the reference gives.
        bool all = true;
        for (std::size_t b = 0; b < saans::kSignedFrameSize; ++b) {
            std::uint8_t bad[saans::kSignedFrameSize];
            std::memcpy(bad, ref, sizeof bad);
            bad[b] = static_cast<std::uint8_t>(bad[b] ^ 0x80u);
            saans::FeatureFrame unused;
            std::uint8_t k = 0;
            std::uint32_t bt = 0;
            const auto e = saans::decode_signed_frame(bad, sizeof bad, key, unused, k, bt);
            const auto expected = b < 4    ? saans::SignedDecodeError::kBadMagic
                                  : b == 4 ? saans::SignedDecodeError::kBadVersion
                                           : saans::SignedDecodeError::kBadTag;
            if (e != expected) {
                all = false;
                std::printf("      byte %zu flipped: got %d\n", b, static_cast<int>(e));
            }
        }
        check(all, tag + ": every flipped byte refused");
    }
    end_case();
}

void test_signed_layout_and_gate() {
    begin_case("signed frames: v1 offsets kept, gate applies");
    saans::FeatureFrame f;
    f.node_id = 7;
    f.tx_id = 2;
    f.seq = 99;
    f.flags = saans::kFlagOccupied | saans::kFlagVitalsEnabled | saans::kFlagBreathingValid;
    f.breathing_bpm = 14.0f;
    f.breathing_conf = 0.8f;
    f.breath_rms = 0.1f;
    f.motion_energy = 0.2f;
    std::uint8_t key[saans::kFrameKeySize];
    std::memset(key, 0x42, sizeof key);

    std::uint8_t v1[saans::kFeatureFrameSize];
    saans::encode_feature_frame(f, v1);
    std::uint8_t v2[saans::kSignedFrameSize];
    saans::encode_signed_frame(f, key, 9, 123456u, v2);
    check(std::memcmp(v1, v2, 4) == 0 && std::memcmp(v1 + 5, v2 + 5, 58) == 0,
          "bytes 0..62 are v1's, but for the version");
    check(v2[4] == 2 && v2[63] == 9, "version 2, key_id in byte 63");
    check(saans::detail::get_u32(v2 + 64) == 123456u, "boot at 64");

    std::uint8_t gated[saans::kSignedFrameSize];
    saans::encode_signed_frame_gated(f, saans::VitalsConsent{}, key, 9, 123456u, gated);
    saans::FeatureFrame back;
    std::uint8_t k = 0;
    std::uint32_t b = 0;
    check(saans::decode_signed_frame(gated, sizeof gated, key, back, k, b) ==
              saans::SignedDecodeError::kOk,
          "gated frame verifies");
    check(back.breathing_bpm == 0.0f && back.breath_rms == 0.0f &&
              !back.flag(saans::kFlagBreathingValid) && back.motion_energy == 0.2f,
          "no consent: vitals zeroed before signing, presence kept");
    std::uint8_t other[saans::kFrameKeySize];
    std::memset(other, 0x43, sizeof other);
    check(saans::decode_signed_frame(v2, sizeof v2, other, back, k, b) ==
              saans::SignedDecodeError::kBadTag,
          "another key is refused");
    check(saans::decode_signed_frame(v1, sizeof v1, key, back, k, b) ==
              saans::SignedDecodeError::kBadLength,
          "a v1 frame is not a v2 frame");
    end_case();
}

}  // namespace

int main(int argc, char** argv) {
    saans::golden::dir_from_argv(argc, argv);
    std::printf("test_frame_auth  (vectors: %s)\n", saans::golden::dir().c_str());

    const saans::golden::Vectors signed_codec("feature_codec_signed");

    test_sha256_nist();
    test_sha256_any_split();
    test_hmac_rfc4231();
    test_equal_ct();
    test_golden_signed(signed_codec);
    test_signed_layout_and_gate();

    std::printf("\n%d cases, %d checks, %d failed\n", g_cases, g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
