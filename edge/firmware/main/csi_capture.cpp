// Portions adapted from MIT-licensed code; copyright and permission notice: /NOTICE.
#include "csi_capture.h"

#include <cmath>
#include <cstring>

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

namespace saans {
namespace csi {
namespace {

constexpr const char* kTag = "saans.csi";

QueueHandle_t s_queue = nullptr;
std::uint32_t s_sequence = 0;
std::uint32_t s_dropped = 0;
std::uint8_t  s_channel = 0;
// Capture-path observability. Nothing here has run against a radio yet
// (SAANS_PLAN S5), so the first bench session needs to be able to answer
// "what is the driver actually handing us?" without a rebuild.
std::uint32_t s_layout_changed = 0;
std::uint32_t s_first_word_invalid = 0;
std::uint32_t s_short_frames = 0;
std::uint32_t s_dropped_short = 0;
std::uint16_t s_last_len = 0;
std::uint16_t s_first_len = 0;
std::uint8_t  s_secondary_channel = 0xFF;
// The PHY mode of what arrived, which the S3 does report after all (see the
// callback): frames that are not HT, and frames that are 40 MHz wide.
std::uint32_t s_not_ht = 0;
std::uint32_t s_wide = 0;
// The receiver's gain, frame to frame (PROTOCOL_S5 §1.1). Every callback is
// counted, including frames later dropped, because the question is how often
// the radio changes its gain, not how often the DSP saw it change.
std::uint32_t s_gain_frames = 0;
std::uint32_t s_agc_moves = 0;
std::uint32_t s_fft_moves = 0;
std::uint8_t  s_agc_last = 0;
std::uint8_t  s_agc_min = 0xFF;
std::uint8_t  s_agc_max = 0;
std::int8_t   s_fft_last = 0;

// THE AGC AND FFT GAIN ARE NOT DOCUMENTED FIELDS. In IDF 5.4.1's
// `wifi_pkt_rx_ctrl_t` for the S3 they sit in the 24 reserved bits after
// `noise_floor` (byte 20). Espressif's own esp_csi_gain_ctrl component
// (v0.1.5, Apache-2.0, binary-only) reads them there: disassembled, its
// esp_csi_gain_ctrl_get_rx_gain is two byte loads, AGC from byte 23 and the
// signed FFT gain from byte 22, identically in its builds for IDF 4.4
// through 6.0 (checked 2026-10-24). That is the vendor's reading, not a
// measurement: until a bench session attenuates the TX and watches the AGC
// value move (PROTOCOL_S5 §1.1), treat these numbers as unconfirmed.
#if CONFIG_IDF_TARGET_ESP32S3
constexpr bool kGainReadable = true;
static_assert(sizeof(wifi_pkt_rx_ctrl_t) >= 24, "rx_ctrl is shorter than the gain offsets");
void read_gain(const wifi_pkt_rx_ctrl_t& rx, std::uint8_t& agc, std::int8_t& fft) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(&rx);
    agc = bytes[23];
    fft = static_cast<std::int8_t>(bytes[22]);
}
#else
constexpr bool kGainReadable = false;
void read_gain(const wifi_pkt_rx_ctrl_t&, std::uint8_t& agc, std::int8_t& fft) {
    agc = 0;
    fft = 0;
}
#endif

// Runs in WiFi TASK context — not an ISR. Everything here is O(n_subcarriers)
// with no allocation, no logging and no blocking send: a slow callback drops
// packets for the whole stack, not just for us.
//
// Deliberately NOT IRAM_ATTR. IRAM_ATTR is for interrupt handlers and for code
// that must run with the flash cache disabled; this is neither. The ESP32-S3
// has 16 KB of IRAM total and the build sits at 99.99% of it, so putting a
// loop containing sqrtf and atan2f there costs the scarcest memory on the chip
// to buy nothing — a task-context callback already executes from cached flash
// perfectly well.
void csi_callback(void* /*ctx*/, wifi_csi_info_t* info) {
    if (info == nullptr || info->buf == nullptr || s_queue == nullptr) {
        return;
    }

    std::uint8_t agc = 0;
    std::int8_t  fft = 0;
    read_gain(info->rx_ctrl, agc, fft);
    if (kGainReadable) {
        if (s_gain_frames != 0) {
            s_agc_moves += (agc != s_agc_last) ? 1U : 0U;
            s_fft_moves += (fft != s_fft_last) ? 1U : 0U;
        }
        ++s_gain_frames;
        s_agc_last = agc;
        s_fft_last = fft;
        s_agc_min = agc < s_agc_min ? agc : s_agc_min;
        s_agc_max = agc > s_agc_max ? agc : s_agc_max;
    }

    // WHICH LTF BLOCK IS THIS? The buffer layout depends on the PHY mode of
    // the frame that was received, and we do not get to assume it: a non-HT
    // frame carries only L-LTF, and slicing the first kNSubcarriers pairs
    // from byte 0 would then hand the DSP a different set of subcarriers
    // from the one it was sized for.
    //
    // The S3 does report the mode: IDF 5.4.1's `wifi_pkt_rx_ctrl_t` has
    // sig_mode (0 non-HT, 1 HT) and cwb (0 20 MHz, 1 40 MHz) on every target
    // without 11ax, the S3 included. (An earlier comment here said those
    // fields existed on the original ESP32 only; the header says otherwise.)
    // `len` is kept as well: the LTF layout determines it, so a change in
    // `len` is a change in what the subcarrier indices mean, whatever the
    // mode fields say. The first job on real hardware is to find out what the
    // radio does, not to assume it.
    s_not_ht += (info->rx_ctrl.sig_mode != 1) ? 1U : 0U;
    s_wide += (info->rx_ctrl.cwb != 0) ? 1U : 0U;
    s_last_len = static_cast<std::uint16_t>(info->len);
    s_secondary_channel = static_cast<std::uint8_t>(info->rx_ctrl.secondary_channel);
    if (s_first_len == 0) {
        s_first_len = s_last_len;
    } else if (s_last_len != s_first_len) {
        ++s_layout_changed;
    }

    // ESP-IDF hands back signed int8 pairs, IMAGINARY FIRST:
    //   buf[2k] = imag, buf[2k+1] = real
    // Amplitude is order-independent, phase is not. atan2(imag, real).
    const auto* iq = reinterpret_cast<const std::int8_t*>(info->buf);
    std::size_t available = static_cast<std::size_t>(info->len) / 2U;

    // Hardware limitation Espressif documents on `wifi_csi_info_t` (NOT on
    // rx_ctrl): when first_word_invalid is set, the first four bytes -- the
    // first TWO complex pairs -- are invalid. Reading them as signal puts a
    // fixed artefact into subcarriers 0 and 1 of every affected frame,
    // which the static suppressor would then learn as part of the room.
    const std::size_t skip = info->first_word_invalid ? 2U : 0U;
    if (skip != 0U) {
        ++s_first_word_invalid;
    }
    if (available <= skip) {
        ++s_dropped_short;
        return;
    }
    iq += 2U * skip;
    available -= skip;

    const std::size_t n = available < cfg::kNSubcarriers ? available : cfg::kNSubcarriers;
    if (n == 0) {
        return;
    }
    if (n < cfg::kNSubcarriers) {
        // Fewer subcarriers than the DSP was compiled for. The tail of the
        // frame is left zeroed, which is a real signal the BNR stage will
        // see; count it rather than let it look like a quiet channel.
        ++s_short_frames;
    }

    Frame frame{};
    frame.n_subcarriers = n;
    frame.sequence = s_sequence++;
    frame.timestamp_us = esp_timer_get_time();
    frame.rssi_dbm = static_cast<std::int8_t>(info->rx_ctrl.rssi);
    frame.noise_floor_dbm = static_cast<std::int8_t>(info->rx_ctrl.noise_floor);
    frame.agc_gain = agc;
    frame.fft_gain = fft;
    s_channel = static_cast<std::uint8_t>(info->rx_ctrl.channel);

    for (std::size_t k = 0; k < n; ++k) {
        const float imag = static_cast<float>(iq[2U * k]);
        const float real = static_cast<float>(iq[2U * k + 1U]);
        frame.amplitudes[k] = sqrtf(imag * imag + real * real);
        frame.phases[k] = atan2f(imag, real);
    }

#if CONFIG_SAANS_RAW_MODE
    frame.iq_len = 2U * n;
    std::memcpy(frame.iq, iq, frame.iq_len);
#endif

    // Never block the WiFi task. If the sensing task is behind, drop the oldest
    // and count it: a stale frame has no value, and waiting here would cost the
    // next frame too.
    if (xQueueSend(s_queue, &frame, 0) != pdTRUE) {
        Frame discard{};
        if (xQueueReceive(s_queue, &discard, 0) == pdTRUE) {
            ++s_dropped;
        }
        if (xQueueSend(s_queue, &frame, 0) != pdTRUE) {
            ++s_dropped;
        }
    }
}

}  // namespace

std::uint32_t channel_to_mhz(std::uint8_t channel) {
    if (channel >= 1 && channel <= 13) {
        return 2412U + (channel - 1U) * 5U;
    }
    if (channel == 14) {
        return 2484U;
    }
    if (channel >= 36 && channel <= 177) {
        return 5000U + channel * 5U;
    }
    return 0U;
}

esp_err_t start(std::size_t queue_depth) {
    s_queue = xQueueCreate(queue_depth, sizeof(Frame));
    if (s_queue == nullptr) {
        ESP_LOGE(kTag, "queue alloc failed (%u x %u bytes)",
                 static_cast<unsigned>(queue_depth), static_cast<unsigned>(sizeof(Frame)));
        return ESP_ERR_NO_MEM;
    }

    wifi_csi_config_t config{};
    // Both LTF fields enabled. Note that enabling htltf_en does NOT make
    // HT frames appear: the transmitter has to send them. tx_role.cpp asks
    // ESP-NOW for an HT20 MCS rate for exactly that reason, and
    // capture_stats() reports what actually arrived.
    config.lltf_en = true;
    config.htltf_en = true;
    config.stbc_htltf2_en = false;
    // FALSE, deliberately. ESP-IDF: ltf_merge_en "generate htltf data by
    // averaging lltf and ht_ltf data when receiving HT packet. Otherwise,
    // use ht_ltf data directly." L-LTF and HT-LTF do not cover the same
    // subcarriers (+-26 against +-28), so averaging them mixes two
    // different channel measurements under one set of indices -- and it
    // flatly contradicts channel_filter_en = false on the next line, which
    // is set precisely to keep adjacent subcarriers independent.
    //
    // This was true until the TX was made to send HT frames, where it was
    // inert because no HT packet ever arrived. Turning the transmitter on
    // would have turned this on with it.
    config.ltf_merge_en = false;
    config.channel_filter_en = false;
    // Phase is left uncorrected on purpose. The DSP works on amplitude, and a
    // 1T1R ESP32 has a free-running PLL whose phase is not usable for ranging
    // anyway (see RESEARCH.md). Letting the driver "fix" it would only add an
    // undocumented transform between us and the Python reference.
    config.manu_scale = false;
    config.shift = 0;

    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(&csi_callback, nullptr));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    ESP_LOGI(kTag, "CSI capture started (queue %u frames, %u bytes each)",
             static_cast<unsigned>(queue_depth), static_cast<unsigned>(sizeof(Frame)));
    return ESP_OK;
}

bool receive(Frame& out, int timeout_ms) {
    if (s_queue == nullptr) {
        return false;
    }
    return xQueueReceive(s_queue, &out, pdMS_TO_TICKS(timeout_ms)) == pdTRUE;
}

std::uint32_t dropped() { return s_dropped; }

Stats capture_stats() {
    Stats st{};
    st.layout_changed = s_layout_changed;
    st.first_word_invalid = s_first_word_invalid;
    st.short_frames = s_short_frames;
    st.dropped_short = s_dropped_short;
    st.first_len = s_first_len;
    st.last_len = s_last_len;
    st.secondary_channel = s_secondary_channel;
    st.not_ht = s_not_ht;
    st.wide = s_wide;
    st.gain_readable = kGainReadable;
    st.gain_frames = s_gain_frames;
    st.agc_moves = s_agc_moves;
    st.fft_moves = s_fft_moves;
    st.agc_gain = s_agc_last;
    st.agc_min = s_agc_min;
    st.agc_max = s_agc_max;
    st.fft_gain = s_fft_last;
    return st;
}

std::size_t serialize_raw(const std::int8_t* iq, std::size_t iq_len, std::uint8_t node_id,
                          std::uint32_t sequence, std::int8_t rssi, std::int8_t noise,
                          std::uint32_t freq_mhz, std::uint8_t agc_gain, std::int8_t fft_gain,
                          std::uint8_t* buf, std::size_t buf_len) {
    if (iq == nullptr || buf == nullptr) {
        return 0;
    }
    const std::size_t total = kRawHeaderSize + iq_len;
    if (total > buf_len) {
        return 0;
    }

    const std::uint8_t n_antennas = 1;
    const std::uint16_t n_subcarriers = static_cast<std::uint16_t>(iq_len / 2U);

    // Written field by field at documented offsets rather than by casting a
    // packed struct, so the wire format is enforced by this code and cannot be
    // changed by a compiler's padding decision.
    std::memcpy(&buf[0], &kRawMagic, 4);
    buf[4] = node_id;
    buf[5] = n_antennas;
    std::memcpy(&buf[6], &n_subcarriers, 2);
    std::memcpy(&buf[8], &freq_mhz, 4);
    std::memcpy(&buf[12], &sequence, 4);
    buf[16] = static_cast<std::uint8_t>(rssi);
    buf[17] = static_cast<std::uint8_t>(noise);
    // spec/wire.md §1: the receiver's AGC and FFT gain for this frame (0, 0
    // from a build that cannot read them). The parsers still name these
    // bytes `ppdu_type` and `flags`.
    buf[18] = agc_gain;
    buf[19] = static_cast<std::uint8_t>(fft_gain);
    std::memcpy(&buf[kRawHeaderSize], iq, iq_len);
    return total;
}

}  // namespace csi
}  // namespace saans
