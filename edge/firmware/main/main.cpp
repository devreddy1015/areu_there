// Saans sensor node entry point.
//
// Role is a compile-time choice, not a runtime one. A TX binary contains no
// sensing code at all, and an RX built without vitals consent contains no code
// that could emit a breathing rate (ADR-001, ADR-005). Making that structural
// rather than conditional means a misconfigured deployment cannot leak data
// that the binary does not know how to produce.

#include "esp_log.h"
#include "esp_system.h"
#include "esp_task_wdt.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "roles.h"
#include "saans_config.h"
#include "telemetry.h"

namespace {

constexpr const char* kTag = "saans";

void log_banner() {
    ESP_LOGI(kTag, "Saans node starting");
    ESP_LOGI(kTag, "  role        : %s",
             saans::cfg::kRole == saans::cfg::Role::Tx ? "TX (beacon)" : "RX (sensing)");
    ESP_LOGI(kTag, "  node id     : %u", static_cast<unsigned>(saans::cfg::kNodeId));
    ESP_LOGI(kTag, "  channel     : %d", saans::cfg::kWifiChannel);
    if (saans::cfg::kRole == saans::cfg::Role::Rx) {
        ESP_LOGI(kTag, "  subcarriers : %u",
                 static_cast<unsigned>(saans::cfg::kNSubcarriers));
        ESP_LOGI(kTag, "  csi rate    : %.0f Hz",
                 static_cast<double>(saans::cfg::kCsiRateHz));
        ESP_LOGI(kTag, "  frame rate  : %.0f Hz",
                 static_cast<double>(saans::cfg::kFrameRateHz));
        // Say the consent state out loud at every boot. Someone reading a
        // serial log should never have to guess whether this node is emitting
        // health data.
        ESP_LOGI(kTag, "  vitals      : %s",
                 saans::cfg::kVitalsEnabled ? "ENABLED (consented)" : "disabled");
        ESP_LOGI(kTag, "  heart       : %s",
                 saans::cfg::kHeartEnabled ? "ENABLED (consented)" : "disabled");
        if (saans::cfg::kRawMode) {
            ESP_LOGW(kTag, "  raw mode    : ON — raw CSI is leaving this node");
        }
    }
}

}  // namespace

extern "C" void app_main(void) {
    log_banner();
    saans::telemetry::init();

    // The watchdog subscribes the sensing task, so a wedged DSP loop reboots
    // the node rather than leaving a room silently unmonitored. Silence is the
    // failure mode that matters here: nobody notices a sensor that stops
    // reporting "nothing happening".
    esp_task_wdt_config_t wdt{};
    wdt.timeout_ms = static_cast<std::uint32_t>(saans::cfg::kWatchdogS) * 1000U;
    wdt.idle_core_mask = 0;
    wdt.trigger_panic = true;
    esp_err_t err = esp_task_wdt_init(&wdt);
    if (err == ESP_ERR_INVALID_STATE) {
        // Already initialised by the SDK; reconfigure to our timeout.
        esp_task_wdt_reconfigure(&wdt);
    } else {
        ESP_ERROR_CHECK(err);
    }

    if constexpr (saans::cfg::kRole == saans::cfg::Role::Tx) {
        saans::run_tx_role();
    } else {
        saans::run_rx_role();
    }

    // Neither role returns. If one does, something is badly wrong and a reboot
    // is more useful than a hung task.
    ESP_LOGE(kTag, "role returned — restarting");
    esp_restart();
}
