#pragma once
// Node health reporting.
//
// SAANS_PLAN section 2.2 makes the system itself a monitored subject, not an
// afterthought: a sensor that quietly degrades is worse than one that fails
// loudly, because the silence reads as "nothing happening in that room".

#include <cstdint>

namespace saans {
namespace telemetry {

struct Health {
    std::uint32_t uptime_s;
    std::uint32_t free_heap;
    std::uint32_t min_free_heap;   // low-water mark since boot: catches a slow leak
    std::uint32_t reboot_count;    // persisted in NVS; a climbing count is a fault
    std::int8_t   rssi_dbm;
};

// Reads the reboot counter from NVS and increments it. Call once at boot.
void init();

Health sample();

// Logs a health line. The gateway also infers health from FeatureFrame fields
// (csi_rate_chz, loss_permille, calib_age_s), so this is for the serial console
// and for anyone physically at the node.
void log_health();

}  // namespace telemetry
}  // namespace saans
