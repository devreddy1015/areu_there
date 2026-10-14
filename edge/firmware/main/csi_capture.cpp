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

    // WHICH LTF BLOCK IS THIS? The buffer layout depends on the PHY mode of
    // the frame that was received, and we do not get to assume it: a non-HT
    // frame carries only L-LTF, and slicing the first kNSubcarriers pairs
    // from byte 0 would then hand the DSP a different set of subcarriers
    // from the one it was sized for.
    //
    // On the S3 we cannot ask directly: `wifi_pkt_rx_ctrl_t` carries
    // sig_mode, cwb and rate only on the original ESP32. What is available
    // here is `len`, and that is enough, because the LTF layout determines
    // it -- a change in `len` IS a change in what the subcarrier indices
    // mean. So record the first length seen and count departures from it.
    // The first job on real hardware is to find out what the radio does,
    // not to assume it.
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
    config.ltf_merge_en = true;
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
    return Stats{s_layout_changed, s_first_word_invalid, s_short_frames, s_dropped_short,
                 s_first_len, s_last_len, s_secondary_channel};
}

std::size_t serialize_raw(const std::int8_t* iq, std::size_t iq_len, std::uint8_t node_id,
                          std::uint32_t sequence, std::int8_t rssi, std::int8_t noise,
                          std::uint32_t freq_mhz, std::uint8_t* buf, std::size_t buf_len) {
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
    buf[18] = 0;  // PPDU type: unused here
    buf[19] = 0;  // flags
    std::memcpy(&buf[kRawHeaderSize], iq, iq_len);
    return total;
}

}  // namespace csi
}  // namespace saans
