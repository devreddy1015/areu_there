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
    // The receiver's gain for this frame (0, 0 where unreadable); see
    // csi_capture.cpp for where they come from and how sure that is.
    std::uint8_t  agc_gain;
    std::int8_t   fft_gain;
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

// What the radio is actually delivering, for the first bench session.
//
// The DSP is compiled for a fixed subcarrier count on the assumption that
// the frames being measured are HT20. Nothing in the firmware can enforce
// that -- it depends on what the transmitter sends and what the driver
// reports -- so this says what arrived instead of assuming.
// The PHY mode is read directly (`sig_mode`, `cwb`), and `len` is kept
// beside it: the LTF layout determines the buffer length, so a change in
// length is a change in what a subcarrier index means.
//
// And the receiver's gain (PROTOCOL_S5 §1.1): an AGC step scales every
// subcarrier at once, which in simulation inverts the presence detector
// (NOTES.md §9). How often it moves is the first number a bench session
// owes. Counted over every frame the radio delivered, in order.
struct Stats {
    std::uint32_t layout_changed;     // frames whose len differed from the first seen
    std::uint32_t first_word_invalid; // frames where subcarriers 0-1 were dropped
    std::uint32_t short_frames;       // fewer subcarriers than the DSP expects
    std::uint32_t dropped_short;      // too short to use at all
    std::uint16_t first_len;          // bytes in the first CSI buffer seen
    std::uint16_t last_len;           // bytes in the most recent one
    std::uint8_t  secondary_channel;  // 0 none, 1 above, 2 below -- 40 MHz tells here
    std::uint32_t not_ht;             // frames with sig_mode != HT: L-LTF only
    std::uint32_t wide;               // frames with cwb set: 40 MHz
    bool          gain_readable;      // false on a target without the gain bytes
    std::uint32_t gain_frames;        // frames whose gain was read
    std::uint32_t agc_moves;          // frames whose AGC gain differed from the previous one
    std::uint32_t fft_moves;          // the same for the FFT gain
    std::uint8_t  agc_gain;           // the most recent frame's
    std::uint8_t  agc_min;            // 0xFF until a frame is read
    std::uint8_t  agc_max;
    std::int8_t   fft_gain;           // the most recent frame's
};

Stats capture_stats();

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
                          std::uint32_t freq_mhz, std::uint8_t agc_gain, std::int8_t fft_gain,
                          std::uint8_t* buf, std::size_t buf_len);

}  // namespace csi
}  // namespace saans
