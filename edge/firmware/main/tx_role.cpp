// TX role: a metronome for the radio channel.
//
// The RX cannot measure a channel nobody is using. Rather than depending on
// whatever ambient traffic happens to exist — which is bursty, and whose rate
// you do not control — a TX node broadcasts a small ESP-NOW frame at a fixed
// rate. ESP-NOW is the right primitive here: connectionless, no association, no
// IP stack, and it works with the two nodes simply agreeing on a channel.
//
// This role deliberately knows nothing about sensing. It holds no state about
// people and emits nothing derived from them.

#include <cstring>

#include "esp_log.h"
#include "esp_now.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "net.h"
#include "roles.h"
#include "saans_config.h"
#include "telemetry.h"

namespace saans {
namespace {

constexpr const char* kTag = "saans.tx";

// ESP-NOW broadcast: no pairing, no ACK, every listener on the channel sees it.
constexpr std::uint8_t kBroadcast[ESP_NOW_ETH_ALEN] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// Payload is deliberately tiny and carries no sensing data — just enough for an
// RX to recognise the source and spot gaps. A longer frame would occupy the
// channel for longer without improving the channel estimate.
struct __attribute__((packed)) BeaconPayload {
    std::uint32_t magic;   // 'SNTX'
    std::uint8_t  node_id;
    std::uint8_t  reserved[3];
    std::uint32_t seq;
};
static_assert(sizeof(BeaconPayload) == 12, "beacon payload must stay small");

constexpr std::uint32_t kBeaconMagic = 0x58544E53;  // "SNTX" little-endian

}  // namespace

void run_tx_role() {
    ESP_LOGI(kTag, "TX role: node %u, channel %d, %d Hz",
             static_cast<unsigned>(cfg::kNodeId), cfg::kWifiChannel, cfg::kTxRateHz);

    ESP_ERROR_CHECK(net::wifi_start_tx_only());
    ESP_ERROR_CHECK(esp_now_init());

    esp_now_peer_info_t peer{};
    std::memcpy(peer.peer_addr, kBroadcast, ESP_NOW_ETH_ALEN);
    peer.channel = static_cast<std::uint8_t>(cfg::kWifiChannel);
    peer.ifidx = WIFI_IF_STA;
    peer.encrypt = false;
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));

    // ASK FOR HT20. Without this an ESP-NOW frame goes out at the non-HT
    // default, which carries only L-LTF -- and the RX's whole premise is
    // that it is measuring the HT-LTF subcarriers. Enabling htltf_en on the
    // receiver does nothing if the transmitter never sends an HT frame, so
    // the two halves have to agree here, not in a comment.
    //
    // MCS0 with a long guard interval is the slowest, most robust HT rate.
    // Throughput is irrelevant: the payload is 12 bytes and the point of
    // the frame is to be a channel probe, not to carry data.
    esp_now_rate_config_t rate{};
    rate.phymode = WIFI_PHY_MODE_HT20;
    rate.rate = WIFI_PHY_RATE_MCS0_LGI;
    rate.ersu = false;
    rate.dcm = false;
    const esp_err_t rate_err = esp_now_set_peer_rate_config(peer.peer_addr, &rate);
    if (rate_err != ESP_OK) {
        // Not fatal: the link still works at the default rate, the RX will
        // just be measuring L-LTF. It MUST be loud, because every claim the
        // DSP makes about subcarrier count depends on it.
        ESP_LOGE(kTag, "HT20 rate config FAILED (%s): frames will be non-HT and the RX "
                       "will see L-LTF, not the HT-LTF subcarriers the DSP is sized for",
                 esp_err_to_name(rate_err));
    } else {
        ESP_LOGI(kTag, "ESP-NOW rate: HT20 MCS0 LGI");
    }

    BeaconPayload payload{};
    payload.magic = kBeaconMagic;
    payload.node_id = cfg::kNodeId;

    // Pace against absolute deadlines, not "sleep for 1/rate". vTaskDelay
    // rounds to the tick and the send itself takes time, so a naive loop drifts
    // slow — and the RX's whole time axis is built on this rate being what it
    // claims to be.
    const TickType_t period = pdMS_TO_TICKS(1000 / cfg::kTxRateHz);
    TickType_t last_wake = xTaskGetTickCount();
    std::int64_t last_report_us = esp_timer_get_time();
    std::uint32_t sent = 0;
    std::uint32_t failed = 0;

    for (;;) {
        esp_err_t err = esp_now_send(kBroadcast, reinterpret_cast<std::uint8_t*>(&payload),
                                     sizeof(payload));
        if (err == ESP_OK) {
            ++sent;
        } else {
            ++failed;
        }
        payload.seq += 1;

        const std::int64_t now_us = esp_timer_get_time();
        if (now_us - last_report_us >= cfg::kTelemetryS * 1000000LL) {
            const float elapsed_s = static_cast<float>(now_us - last_report_us) / 1e6f;
            ESP_LOGI(kTag, "sent=%lu failed=%lu rate=%.1f Hz",
                     static_cast<unsigned long>(sent), static_cast<unsigned long>(failed),
                     static_cast<double>(static_cast<float>(sent) / elapsed_s));
            telemetry::log_health();
            sent = 0;
            failed = 0;
            last_report_us = now_us;
        }

        vTaskDelayUntil(&last_wake, period > 0 ? period : 1);
    }
}

}  // namespace saans
