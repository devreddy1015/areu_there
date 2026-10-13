#include "net.h"

#include <cstring>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "lwip/sockets.h"
#include "nvs_flash.h"

#include "saans_config.h"

namespace saans {
namespace net {
namespace {

constexpr const char* kTag = "saans.net";
constexpr int kGotIpBit = BIT0;

EventGroupHandle_t s_events = nullptr;
int s_sock = -1;
sockaddr_in s_gateway{};
bool s_common_started = false;

void on_wifi_event(void*, esp_event_base_t base, int32_t id, void*) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        // Reconnect forever rather than giving up. A node on a ceiling has no
        // one to press a button, and an AP reboot must not end the deployment.
        ESP_LOGW(kTag, "disconnected, retrying");
        if (s_events != nullptr) {
            xEventGroupClearBits(s_events, kGotIpBit);
        }
        esp_wifi_connect();
    }
}

void on_ip_event(void*, esp_event_base_t, int32_t id, void* data) {
    if (id != IP_EVENT_STA_GOT_IP) {
        return;
    }
    auto* event = static_cast<ip_event_got_ip_t*>(data);
    ESP_LOGI(kTag, "got ip " IPSTR, IP2STR(&event->ip_info.ip));
    if (s_events != nullptr) {
        xEventGroupSetBits(s_events, kGotIpBit);
    }
}

esp_err_t common_init() {
    if (s_common_started) {
        return ESP_OK;
    }
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    if (err != ESP_OK) {
        return err;
    }
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    s_common_started = true;
    return ESP_OK;
}

}  // namespace

esp_err_t wifi_start_tx_only() {
    ESP_ERROR_CHECK(common_init());

    wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());
    return set_channel(cfg::kWifiChannel);
}

esp_err_t wifi_start_station() {
    ESP_ERROR_CHECK(common_init());

    s_events = xEventGroupCreate();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &on_wifi_event, nullptr, nullptr));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &on_ip_event, nullptr, nullptr));

    wifi_config_t wifi{};
    std::strncpy(reinterpret_cast<char*>(wifi.sta.ssid), cfg::kWifiSsid,
                 sizeof(wifi.sta.ssid) - 1);
    std::strncpy(reinterpret_cast<char*>(wifi.sta.password), cfg::kWifiPassword,
                 sizeof(wifi.sta.password) - 1);

    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi));
    ESP_ERROR_CHECK(esp_wifi_start());

    // Power save off. The modem sleeping between beacons is exactly the thing
    // that would make CSI arrive in bursts with gaps, and the DSP's whole
    // premise is a uniform time axis.
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    return ESP_OK;
}

esp_err_t wait_for_ip(int timeout_ms) {
    if (s_events == nullptr) {
        return ESP_ERR_INVALID_STATE;
    }
    EventBits_t bits = xEventGroupWaitBits(s_events, kGotIpBit, pdFALSE, pdTRUE,
                                           pdMS_TO_TICKS(timeout_ms));
    return (bits & kGotIpBit) ? ESP_OK : ESP_ERR_TIMEOUT;
}

esp_err_t set_channel(int channel) {
    return esp_wifi_set_channel(static_cast<uint8_t>(channel), WIFI_SECOND_CHAN_NONE);
}

esp_err_t udp_open() {
    if (s_sock >= 0) {
        return ESP_OK;
    }
    s_sock = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s_sock < 0) {
        ESP_LOGE(kTag, "socket() failed: errno %d", errno);
        return ESP_FAIL;
    }
    std::memset(&s_gateway, 0, sizeof(s_gateway));
    s_gateway.sin_family = AF_INET;
    s_gateway.sin_port = htons(static_cast<uint16_t>(cfg::kGatewayPort));
    if (::inet_pton(AF_INET, cfg::kGatewayIp, &s_gateway.sin_addr) != 1) {
        ESP_LOGE(kTag, "bad gateway address '%s'", cfg::kGatewayIp);
        udp_close();
        return ESP_ERR_INVALID_ARG;
    }
    ESP_LOGI(kTag, "uplink -> %s:%d", cfg::kGatewayIp, cfg::kGatewayPort);
    return ESP_OK;
}

esp_err_t udp_send(const void* data, std::size_t len) {
    if (s_sock < 0) {
        return ESP_ERR_INVALID_STATE;
    }
    ssize_t sent = ::sendto(s_sock, data, len, 0,
                            reinterpret_cast<sockaddr*>(&s_gateway), sizeof(s_gateway));
    return (sent == static_cast<ssize_t>(len)) ? ESP_OK : ESP_FAIL;
}

void udp_close() {
    if (s_sock >= 0) {
        ::close(s_sock);
        s_sock = -1;
    }
}

}  // namespace net
}  // namespace saans
