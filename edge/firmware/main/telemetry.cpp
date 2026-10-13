#include "telemetry.h"

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "nvs.h"
#include "nvs_flash.h"

namespace saans {
namespace telemetry {
namespace {

constexpr const char* kTag = "saans.health";
constexpr const char* kNamespace = "saans";
constexpr const char* kRebootKey = "reboots";

std::uint32_t s_reboot_count = 0;

}  // namespace

void init() {
    nvs_handle_t handle;
    if (nvs_open(kNamespace, NVS_READWRITE, &handle) != ESP_OK) {
        ESP_LOGW(kTag, "NVS unavailable; reboot count will read 0");
        return;
    }
    std::uint32_t count = 0;
    nvs_get_u32(handle, kRebootKey, &count);  // absent on first boot, leaves 0
    count += 1;
    s_reboot_count = count;
    if (nvs_set_u32(handle, kRebootKey, count) == ESP_OK) {
        nvs_commit(handle);
    }
    nvs_close(handle);
    ESP_LOGI(kTag, "boot #%lu", static_cast<unsigned long>(count));
}

Health sample() {
    Health h{};
    h.uptime_s = static_cast<std::uint32_t>(esp_timer_get_time() / 1000000LL);
    h.free_heap = static_cast<std::uint32_t>(heap_caps_get_free_size(MALLOC_CAP_DEFAULT));
    h.min_free_heap =
        static_cast<std::uint32_t>(heap_caps_get_minimum_free_size(MALLOC_CAP_DEFAULT));
    h.reboot_count = s_reboot_count;

    wifi_ap_record_t ap{};
    h.rssi_dbm = (esp_wifi_sta_get_ap_info(&ap) == ESP_OK)
                     ? static_cast<std::int8_t>(ap.rssi)
                     : static_cast<std::int8_t>(0);
    return h;
}

void log_health() {
    const Health h = sample();
    // min_free_heap is the number that matters. free_heap fluctuates; the
    // low-water mark only ever falls, so a downward trend across reports is a
    // leak, and the DSP allocating nothing after init means any trend is a bug
    // somewhere else.
    ESP_LOGI(kTag, "up=%lus heap=%lu min=%lu reboots=%lu rssi=%d",
             static_cast<unsigned long>(h.uptime_s),
             static_cast<unsigned long>(h.free_heap),
             static_cast<unsigned long>(h.min_free_heap),
             static_cast<unsigned long>(h.reboot_count),
             static_cast<int>(h.rssi_dbm));
}

}  // namespace telemetry
}  // namespace saans
