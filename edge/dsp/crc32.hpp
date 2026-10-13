// IEEE 802.3 CRC-32, byte-for-byte what Python's zlib.crc32 produces.
//
// FeatureFrame v1 (spec/wire.md §2) ends in a CRC over its first 64 bytes.
// A gateway drops a frame whose CRC does not match, so this has to agree
// with the reference implementation exactly, not approximately: a one-bit
// disagreement is a link that silently carries nothing.
//
// Convention (zlib / PKZIP / Ethernet FCS, the "CRC-32" everyone means):
//   polynomial 0x04C11DB7, used here in its reflected form 0xEDB88320
//   input bits reflected, output reflected, init 0xFFFFFFFF, final XOR
//   0xFFFFFFFF.
// The reflection is folded into the table, which is why the code below
// shifts right and the polynomial constant looks bit-reversed.
//
// The table is `constexpr`, so it lands in .rodata (flash on an ESP32-S3)
// and costs no RAM. 1 KiB of flash buys ~8x over the bitwise loop; the
// node computes one CRC per 68-byte frame at 2 Hz, so this is already
// generous, but the table is free at runtime.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace saans {

namespace detail {

// Generated at compile time rather than written out as 256 magic numbers:
// the polynomial above is then the only thing a reviewer has to check.
constexpr std::array<std::uint32_t, 256> make_crc32_table() {
    std::array<std::uint32_t, 256> table{};
    for (std::uint32_t i = 0; i < 256; ++i) {
        std::uint32_t c = i;
        for (int bit = 0; bit < 8; ++bit) {
            c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        }
        table[i] = c;
    }
    return table;
}

}  // namespace detail

inline constexpr std::array<std::uint32_t, 256> kCrc32Table = detail::make_crc32_table();

/// Continue a CRC over more bytes. `crc` is a previous return value, or 0
/// to start -- the same calling convention as `zlib.crc32(data, value)`,
/// so `crc32_update(crc32_update(0, a, n), b, m)` equals the CRC of the
/// concatenation. The init/final XOR with 0xFFFFFFFF lives inside this
/// function, which is what makes chaining work.
inline std::uint32_t crc32_update(std::uint32_t crc, const std::uint8_t* data,
                                  std::size_t len) noexcept {
    std::uint32_t c = crc ^ 0xFFFFFFFFu;
    for (std::size_t i = 0; i < len; ++i) {
        c = kCrc32Table[static_cast<std::size_t>((c ^ data[i]) & 0xFFu)] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

/// CRC-32 of one buffer. Equals `zlib.crc32(data) & 0xFFFFFFFF`.
inline std::uint32_t crc32(const std::uint8_t* data, std::size_t len) noexcept {
    return crc32_update(0u, data, len);
}

}  // namespace saans
