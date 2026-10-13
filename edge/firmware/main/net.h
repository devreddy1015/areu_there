#pragma once
// WiFi bring-up and the UDP uplink to the gateway.

#include <cstddef>
#include <cstdint>

#include "esp_err.h"

namespace saans {
namespace net {

// Brings up NVS, the event loop and WiFi. The TX role needs the radio but no
// association (ESP-NOW is connectionless); the RX role associates so it can
// reach the gateway over IP.
esp_err_t wifi_start_tx_only();
esp_err_t wifi_start_station();

// Blocks until associated and an IP is assigned, or the timeout expires.
esp_err_t wait_for_ip(int timeout_ms);

// One UDP socket, opened once and reused. Datagrams are fire-and-forget: a
// dropped FeatureFrame is a gap the gateway can see in the seq field, which is
// cheaper than buffering and retrying on a node with 512 KB of RAM.
esp_err_t udp_open();
esp_err_t udp_send(const void* data, std::size_t len);
void udp_close();

// Channel must be pinned on both roles; ESP-NOW has nothing else tying them
// together.
esp_err_t set_channel(int channel);

}  // namespace net
}  // namespace saans
