// SHA-256 (FIPS 180-4) and HMAC-SHA256 (RFC 2104), for signing
// FeatureFrame v2 (ADR-012, spec/wire.md §2b).
//
// Why not mbedTLS: the node has it (ESP-IDF ships it), the host test build
// does not, and the bytes have to be the same on both. One implementation,
// tested on the host against the NIST and RFC 4231 vectors and against the
// Python reference's golden frames (test_frame_auth.cpp), is the one the
// node runs. At 2 Hz per link the cost is nothing: one tag is four
// compression blocks.
//
// No data-dependent branches or table lookups: SHA-256's control flow does
// not depend on the key or the message, so signing leaks nothing through
// timing. The one comparison that must be constant-time, a received tag
// against the expected one, is `equal_ct`.
//
// Like the rest of edge/dsp: no allocation, no exceptions, no streams, and
// every multi-byte value is assembled byte by byte in big-endian order, as
// the standard defines it, so host and target agree regardless of their own
// byte order.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace saans {

constexpr std::size_t kSha256Size = 32;
constexpr std::size_t kSha256BlockSize = 64;

namespace detail {

constexpr std::uint32_t kSha256K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
    0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
    0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
    0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
    0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
    0xc67178f2u,
};

constexpr std::uint32_t rotr(std::uint32_t x, unsigned n) noexcept {
    return (x >> n) | (x << (32u - n));
}

inline std::uint32_t load_be32(const std::uint8_t* p) noexcept {
    return (static_cast<std::uint32_t>(p[0]) << 24) | (static_cast<std::uint32_t>(p[1]) << 16) |
           (static_cast<std::uint32_t>(p[2]) << 8) | static_cast<std::uint32_t>(p[3]);
}

inline void store_be32(std::uint8_t* p, std::uint32_t v) noexcept {
    p[0] = static_cast<std::uint8_t>(v >> 24);
    p[1] = static_cast<std::uint8_t>(v >> 16);
    p[2] = static_cast<std::uint8_t>(v >> 8);
    p[3] = static_cast<std::uint8_t>(v);
}

}  // namespace detail

/// Streaming SHA-256: `update` any number of times, then `finish` once.
class Sha256 {
  public:
    Sha256() noexcept { reset(); }

    void reset() noexcept {
        static constexpr std::uint32_t kInit[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                                                   0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                                                   0x1f83d9abu, 0x5be0cd19u};
        std::memcpy(h_, kInit, sizeof(h_));
        buf_len_ = 0;
        total_ = 0;
    }

    void update(const std::uint8_t* data, std::size_t len) noexcept {
        total_ += len;
        while (len > 0) {
            const std::size_t take =
                len < kSha256BlockSize - buf_len_ ? len : kSha256BlockSize - buf_len_;
            std::memcpy(buf_ + buf_len_, data, take);
            buf_len_ += take;
            data += take;
            len -= take;
            if (buf_len_ == kSha256BlockSize) {
                compress(buf_);
                buf_len_ = 0;
            }
        }
    }

    /// Pad, finish and write the digest. Call `reset` before reusing.
    void finish(std::uint8_t out[kSha256Size]) noexcept {
        const std::uint64_t bits = total_ * 8u;
        const std::uint8_t one = 0x80;
        update(&one, 1);
        const std::uint8_t zero = 0;
        while (buf_len_ != kSha256BlockSize - 8) {
            update(&zero, 1);
        }
        std::uint8_t len_be[8];
        for (int i = 0; i < 8; ++i) {
            len_be[i] = static_cast<std::uint8_t>(bits >> (56 - 8 * i));
        }
        update(len_be, 8);
        for (int i = 0; i < 8; ++i) {
            detail::store_be32(out + 4 * i, h_[i]);
        }
    }

  private:
    void compress(const std::uint8_t* block) noexcept {
        using detail::rotr;
        std::uint32_t w[64];
        for (int t = 0; t < 16; ++t) {
            w[t] = detail::load_be32(block + 4 * t);
        }
        for (int t = 16; t < 64; ++t) {
            const std::uint32_t s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >> 3);
            const std::uint32_t s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16] + s0 + w[t - 7] + s1;
        }
        std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
        std::uint32_t e = h_[4], f = h_[5], g = h_[6], h = h_[7];
        for (int t = 0; t < 64; ++t) {
            const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ (~e & g);
            const std::uint32_t t1 = h + s1 + ch + detail::kSha256K[t] + w[t];
            const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = s0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h_[0] += a;
        h_[1] += b;
        h_[2] += c;
        h_[3] += d;
        h_[4] += e;
        h_[5] += f;
        h_[6] += g;
        h_[7] += h;
    }

    std::uint32_t h_[8];
    std::uint8_t buf_[kSha256BlockSize];
    std::size_t buf_len_;
    std::uint64_t total_;  // bytes
};

inline void sha256(const std::uint8_t* data, std::size_t len, std::uint8_t out[kSha256Size]) noexcept {
    Sha256 s;
    s.update(data, len);
    s.finish(out);
}

/// HMAC-SHA256 (RFC 2104): a key longer than a block is hashed first, then
/// the inner hash over (key ^ ipad) || message, the outer over
/// (key ^ opad) || inner.
class HmacSha256 {
  public:
    HmacSha256(const std::uint8_t* key, std::size_t key_len) noexcept {
        std::uint8_t k0[kSha256BlockSize] = {};
        if (key_len > kSha256BlockSize) {
            sha256(key, key_len, k0);
        } else {
            std::memcpy(k0, key, key_len);
        }
        std::uint8_t ipad[kSha256BlockSize];
        for (std::size_t i = 0; i < kSha256BlockSize; ++i) {
            ipad[i] = static_cast<std::uint8_t>(k0[i] ^ 0x36u);
            opad_key_[i] = static_cast<std::uint8_t>(k0[i] ^ 0x5cu);
        }
        inner_.update(ipad, kSha256BlockSize);
        // The padded key is secret: do not leave copies on the stack. Through
        // a volatile pointer, so the stores are not optimised away as dead.
        volatile std::uint8_t* wipe_k0 = k0;
        volatile std::uint8_t* wipe_ipad = ipad;
        for (std::size_t i = 0; i < kSha256BlockSize; ++i) {
            wipe_k0[i] = 0;
            wipe_ipad[i] = 0;
        }
    }

    void update(const std::uint8_t* data, std::size_t len) noexcept { inner_.update(data, len); }

    void finish(std::uint8_t out[kSha256Size]) noexcept {
        std::uint8_t inner_digest[kSha256Size];
        inner_.finish(inner_digest);
        Sha256 outer;
        outer.update(opad_key_, kSha256BlockSize);
        outer.update(inner_digest, kSha256Size);
        outer.finish(out);
    }

  private:
    Sha256 inner_;
    std::uint8_t opad_key_[kSha256BlockSize];
};

inline void hmac_sha256(const std::uint8_t* key, std::size_t key_len, const std::uint8_t* data,
                        std::size_t len, std::uint8_t out[kSha256Size]) noexcept {
    HmacSha256 m(key, key_len);
    m.update(data, len);
    m.finish(out);
}

/// Compare without an early exit, so the time taken does not say how many
/// leading bytes of a forged tag were right.
inline bool equal_ct(const std::uint8_t* a, const std::uint8_t* b, std::size_t n) noexcept {
    std::uint8_t diff = 0;
    for (std::size_t i = 0; i < n; ++i) {
        diff = static_cast<std::uint8_t>(diff | (a[i] ^ b[i]));
    }
    return diff == 0;
}

}  // namespace saans
