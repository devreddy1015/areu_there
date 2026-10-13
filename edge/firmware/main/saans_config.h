#pragma once
// Kconfig values as typed constants, in one place.
//
// Kconfig macros are untyped and vanish when an option is disabled, which makes
// them awkward to use directly in C++ (`#ifdef` scattered through logic, and no
// compile-time checking of ranges). Converting them once here means the rest of
// the firmware reads ordinary constants and the invariants get static_assert'd.

#include "sdkconfig.h"

#include <cstdint>

namespace saans {
namespace cfg {

enum class Role { Tx, Rx };

#if defined(CONFIG_SAANS_ROLE_TX)
inline constexpr Role kRole = Role::Tx;
#else
inline constexpr Role kRole = Role::Rx;
#endif

inline constexpr std::uint8_t kNodeId = CONFIG_SAANS_NODE_ID;
inline constexpr int kWifiChannel = CONFIG_SAANS_WIFI_CHANNEL;

// Only set on a TX build. The fallback is the CSI rate rather than a literal
// because that is what the two have to be: one beacon is one CSI frame at the
// RX and nothing between the radio and the DSP decimates, so a TX beaconing
// faster than the RX's configured rate scales the RX's whole time axis. The
// Kconfig default tracks SAANS_CSI_RATE_HZ for the same reason.
#if defined(CONFIG_SAANS_TX_RATE_HZ)
inline constexpr int kTxRateHz = CONFIG_SAANS_TX_RATE_HZ;
#else
inline constexpr int kTxRateHz = CONFIG_SAANS_CSI_RATE_HZ;
#endif

inline constexpr std::size_t kNSubcarriers = CONFIG_SAANS_N_SUBCARRIERS;
inline constexpr float kCsiRateHz = static_cast<float>(CONFIG_SAANS_CSI_RATE_HZ);

// The same rate as an integer, because the DSP's capacities are derived from
// it at compile time (dsp/node_config.hpp) and C++17 has no float non-type
// template parameters. The static_assert below keeps the two spellings from
// drifting apart if one of them is ever edited by hand.
inline constexpr unsigned kCsiRateHzInt = CONFIG_SAANS_CSI_RATE_HZ;
static_assert(kCsiRateHz == static_cast<float>(kCsiRateHzInt),
              "the float and integer spellings of the CSI rate disagree");

inline constexpr float kFrameRateHz = static_cast<float>(CONFIG_SAANS_FRAME_RATE_HZ);
inline constexpr float kCalibrationS = static_cast<float>(CONFIG_SAANS_CALIBRATION_S);

// --- privacy (ADR-001, ADR-005) ---------------------------------------------
// Compile-time, not runtime. A node built without vitals consent cannot be
// talked into emitting vitals by a malformed packet, because the code that
// would do it is not in the binary.
#if defined(CONFIG_SAANS_VITALS_ENABLED)
inline constexpr bool kVitalsEnabled = true;
#else
inline constexpr bool kVitalsEnabled = false;
#endif

#if defined(CONFIG_SAANS_HEART_ENABLED)
inline constexpr bool kHeartEnabled = true;
#else
inline constexpr bool kHeartEnabled = false;
#endif

#if defined(CONFIG_SAANS_RAW_MODE)
inline constexpr bool kRawMode = true;
#else
inline constexpr bool kRawMode = false;
#endif

// --- link quality -----------------------------------------------------------
// Not a privacy switch: best_bnr is a property of the radio link, not of the
// person in the room (that is why the consent gate keeps it; see
// feature_frame.hpp). It is here because it is a MEMORY switch — the 30 s row
// store is more than half of what one link costs. Off, best_bnr goes out as
// 0.0, which spec/wire.md defines as "unknown", and nothing else in the frame
// moves. Compile-time, so with it off the row store is not in the binary.
#if defined(CONFIG_SAANS_COMPUTE_BNR)
inline constexpr bool kComputeBnr = true;
#else
inline constexpr bool kComputeBnr = false;
#endif

static_assert(!kHeartEnabled || kVitalsEnabled,
              "heart rate requires vitals consent; Kconfig should enforce this, "
              "but assert it here too because it is a consent rule, not a preference");

// --- gateway -----------------------------------------------------------------
inline constexpr const char* kGatewayIp = CONFIG_SAANS_GATEWAY_IP;
inline constexpr int kGatewayPort = CONFIG_SAANS_GATEWAY_PORT;

#if defined(CONFIG_SAANS_WIFI_SSID)
inline constexpr const char* kWifiSsid = CONFIG_SAANS_WIFI_SSID;
inline constexpr const char* kWifiPassword = CONFIG_SAANS_WIFI_PASSWORD;
#else
inline constexpr const char* kWifiSsid = "";
inline constexpr const char* kWifiPassword = "";
#endif

// --- reliability --------------------------------------------------------------
inline constexpr int kWatchdogS = CONFIG_SAANS_WATCHDOG_S;
inline constexpr int kTelemetryS = CONFIG_SAANS_TELEMETRY_S;

// --- derived / invariants -----------------------------------------------------
static_assert(kNodeId != 0xFF, "0xFF is reserved for 'unknown / access point' in tx_id");
static_assert(kNSubcarriers >= 8 && kNSubcarriers <= 64, "DSP is sized for 8..64 subcarriers");

// The gross-motion band reaches 4 Hz, so anything under ~10 Hz cannot represent
// it. features.py refuses to construct below 10 Hz; fail at compile time here
// instead of at boot on a ceiling.
static_assert(CONFIG_SAANS_CSI_RATE_HZ >= 10,
              "CSI rate must be >= 10 Hz: the gross-motion band reaches 4 Hz");
static_assert(CONFIG_SAANS_FRAME_RATE_HZ <= CONFIG_SAANS_CSI_RATE_HZ,
              "cannot emit FeatureFrames faster than CSI arrives");

}  // namespace cfg
}  // namespace saans
