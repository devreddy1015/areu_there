#pragma once
// Portions adapted from MIT-licensed code; copyright and permission notice: /NOTICE.
//
// CSI capture: the ESP-IDF callback, and the raw CSI debug wire format.
//
// Kept separate from the sensing loop for one reason: the callback runs in WiFi
// task context, where blocking or heavy arithmetic will drop packets and stall
// the stack. So the callback does the minimum — convert I/Q to amplitude and
// phase, stamp it, hand it off — and all DSP happens on the sensing task.
//
// The raw debug path emits the format in spec/wire.md section 1 (magic
// 0xC5110001), so the Python tooling in wifisense/sources/esp32_udp.py parses
// it unchanged.

#include <cstddef>
#include <cstdint>

#include "esp_err.h"

#include "saans_config.h"

namespace saans {
namespace csi {

// One captured frame, already converted out of int8 I/Q.
//
// Fixed arrays, sized from Kconfig: this struct is the queue element, and the
// no-heap-after-init rule applies to the capture path as much as to the DSP.
struct Frame {
    float         amplitudes[cfg::kNSubcarriers];
    float         phases[cfg::kNSubcarriers];
    std::size_t   n_subcarriers;
    std::uint32_t sequence;      // wire sequence number, for the gap filler
    std::int64_t  timestamp_us;  // esp_timer, for link rate and uptime
    std::int8_t   rssi_dbm;
    std::int8_t   noise_floor_dbm;
#if CONFIG_SAANS_RAW_MODE
    // The ORIGINAL int8 I/Q, kept only in raw builds. Amplitude and phase are
    // lossy derivatives: you cannot reconstruct the pairs from them, and
    // re-synthesising something I/Q-shaped would produce a raw stream that
    // looks plausible and is not the channel. Costs 2*N bytes per queued
    // frame, which is why it is compiled out when raw mode is off.
    std::int8_t   iq[2 * cfg::kNSubcarriers];
    std::size_t   iq_len;
#endif
};

// Starts the radio's CSI callback and the queue behind it.
// `queue_depth` frames of slack absorb a sensing-task hiccup; beyond that the
// oldest is dropped and counted, because a stale CSI frame is worthless and
// blocking the WiFi task to keep it would cost the next one too.
esp_err_t start(std::size_t queue_depth = 32);

// Blocks until a frame is available or the timeout expires.
bool receive(Frame& out, int timeout_ms);

// Frames the capture path dropped because the queue was full. This is a real
// health signal: a climbing count means the DSP is not keeping up with the
// radio, which shows up at the gateway as a falling csi_rate_chz.
std::uint32_t dropped();

// --- raw CSI debug frame (debug mode only) ----------------------------------
// 20-byte header, then int8 I/Q pairs, IMAGINARY FIRST. Reversing that pair
// order leaves amplitude correct and mirrors every phase, which survives
// amplitude-only testing and then quietly breaks anything phase-based.
inline constexpr std::uint32_t kRawMagic = 0xC5110001;
inline constexpr std::size_t   kRawHeaderSize = 20;

// Centre frequency for an 802.11 channel number, for the raw frame header.
// Covers 2.4 GHz 1-13, channel 14, and the 5 GHz band; returns 0 if unknown.
std::uint32_t channel_to_mhz(std::uint8_t channel);

// Serialises the most recent raw I/Q into `buf`. Returns bytes written, or 0 if
// it would not fit.
std::size_t serialize_raw(const std::int8_t* iq, std::size_t iq_len, std::uint8_t node_id,
                          std::uint32_t sequence, std::int8_t rssi, std::int8_t noise,
                          std::uint32_t freq_mhz, std::uint8_t* buf, std::size_t buf_len);

}  // namespace csi
}  // namespace saans
