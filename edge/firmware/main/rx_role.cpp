// RX role: capture CSI, run the DSP, emit FeatureFrames.
//
// Structure is deliberate. The CSI callback (csi_capture.cpp) runs in WiFi task
// context and does the minimum; everything expensive happens here, on a task
// with its own stack, fed by a queue. The DSP allocates nothing after init, so
// the only heap on this path is that queue, allocated once at startup.

#include <cstdint>
#include <cstring>
#include <optional>

#include "esp_log.h"
#include "esp_task_wdt.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "csi_capture.h"
#include "frame_auth.h"
#include "net.h"
#include "roles.h"
#include "saans_config.h"
#include "telemetry.h"

#include "feature_extractor.hpp"
#include "feature_frame.hpp"
#include "node_config.hpp"

namespace saans {
namespace {

constexpr const char* kTag = "saans.rx";

// THE NODE'S SHAPE, not the library's defaults.
//
// `BasicFeatureExtractor`'s defaults are sized for any sample rate a test or a
// bench run might name; this node has pinned its rate and its subcarrier
// count, so `node_config.hpp` derives every capacity from those and
// static_asserts each one against the requirement it has to meet. That is
// 69,040 bytes a link down to 46,792 (host figures; the S3's are 67,432 and 45,408),
// with the spectral scratch — which is per-CALL working space, not per-link
// state — moved into one arena the whole node computes in.
//
// THE SHARED ARENA ASSUMES ONE CALLER AT A TIME. There is exactly one sensing
// task in this firmware and it services its links in turn. A second sensing
// task, or a second core, needs its own arena: see pipeline.hpp, which spells
// out what interleaving into one arena does and why nothing in the output
// would say so.
//
// CONFIG_SAANS_COMPUTE_BNR drops the BNR row store from the TYPE when it is
// off — 26.9 KB of the 46 — and best_bnr then goes out as 0.0 = UNKNOWN.
using Extractor =
    node::NodeExtractor<cfg::kNSubcarriers, cfg::kCsiRateHzInt, cfg::kComputeBnr>;

// One extractor, one wire buffer, both static. The extractor is tens of KB of
// member arrays; putting it on the sensing task's stack would require a stack
// larger than the object, for no benefit.
Extractor  g_extractor;
std::uint8_t g_wire[kFeatureFrameSize];
std::uint8_t g_wire_signed[kSignedFrameSize];

void log_startup_expectations() {
    // Five periods of the presence band's low edge, which is how the DSP
    // derives its warmup. Stated in seconds because the person doing the
    // calibration is standing in a doorway with a phone, not reading source.
    const float warmup_s = 5.0f / 0.1f;
    ESP_LOGI(kTag, "keep the space EMPTY for ~%.0f s (%.0f s filter warmup + %.0f s calibration)",
             static_cast<double>(warmup_s + cfg::kCalibrationS),
             static_cast<double>(warmup_s), static_cast<double>(cfg::kCalibrationS));
    ESP_LOGI(kTag, "calibrating with someone present learns them as the baseline, "
                   "and the room then reads as permanently empty");
}

bool configure_extractor() {
    FeatureExtractorConfig fx{};
    fx.n_subcarriers  = cfg::kNSubcarriers;
    fx.sample_rate_hz = cfg::kCsiRateHz;
    fx.node_id        = cfg::kNodeId;
    fx.tx_id          = 0xFF;  // one TX beacon source, not individually identified yet
    fx.frame_rate_hz  = cfg::kFrameRateHz;
    fx.vitals_enabled = cfg::kVitalsEnabled;
    fx.heart_enabled  = cfg::kHeartEnabled;
    fx.calibration_s  = cfg::kCalibrationS;
    fx.raw_mode       = cfg::kRawMode;
    // Must agree with the TYPE above: a build with no row store refuses a
    // configuration that asks for link quality, rather than reporting a 0.0
    // the gateway would read as "the window has not filled yet".
    fx.compute_bnr    = cfg::kComputeBnr;

    // configure() returning false means the configuration does not fit the
    // compiled capacities, and the extractor stays INERT rather than silently
    // processing a shortened window. That must be loud: an inert link looks
    // exactly like an empty room from the gateway.
    if (!g_extractor.configure(fx)) {
        ESP_LOGE(kTag, "extractor configuration does not fit compiled capacities "
                       "(%u subcarriers @ %.0f Hz) — node is INERT",
                 static_cast<unsigned>(cfg::kNSubcarriers),
                 static_cast<double>(cfg::kCsiRateHz));
        return false;
    }
    // The per-link figure for THIS build, on the part, which is the only one
    // that is actually true of the node — every other number in the headers is
    // a host `sizeof`. The arena is reported separately because it is paid
    // once for the node rather than once per link.
    ESP_LOGI(kTag, "extractor ready (%u bytes/link + %u bytes of shared spectral arena, "
                   "best_bnr %s)",
             static_cast<unsigned>(sizeof(Extractor)),
             static_cast<unsigned>(Extractor::Pipeline::kScratchFloats * sizeof(float)),
             g_extractor.computes_bnr() ? "on" : "OFF (reported as 0 = unknown)");
    return true;
}

}  // namespace

void run_rx_role() {
    ESP_LOGI(kTag, "RX role: node %u, %u subcarriers @ %.0f Hz -> FeatureFrames @ %.0f Hz",
             static_cast<unsigned>(cfg::kNodeId), static_cast<unsigned>(cfg::kNSubcarriers),
             static_cast<double>(cfg::kCsiRateHz), static_cast<double>(cfg::kFrameRateHz));

    ESP_ERROR_CHECK(net::wifi_start_station());
    if (net::wait_for_ip(30000) != ESP_OK) {
        // Keep going rather than rebooting. CSI capture needs no IP and the
        // uplink may come back; a node that reboots on a slow AP never reaches
        // a steady state.
        ESP_LOGW(kTag, "no IP yet — continuing, uplink will retry");
    }
    ESP_ERROR_CHECK(net::udp_open());
    ESP_ERROR_CHECK(net::set_channel(cfg::kWifiChannel));

    // A MISCONFIGURED EXTRACTOR IS FATAL, NOT SOMETHING TO RUN WITHOUT.
    //
    // This node's entire output is "is someone there". An extractor that
    // cannot be configured emits nothing, and nothing is indistinguishable at
    // the gateway from a working sensor watching an empty room -- so a silent
    // inert node reports "empty" forever and no one finds out. The failure
    // must be loud and it must keep being loud, which means refusing to run
    // and saying why on a schedule, not one ESP_LOGE at boot that scrolls away.
    //
    // Halting rather than rebooting is deliberate: the cause is a compile-time
    // capacity mismatch, so a reboot loop would just repeat it while making
    // the serial log unreadable.
    if (!configure_extractor()) {
        esp_task_wdt_add(nullptr);
        for (;;) {
            ESP_LOGE(kTag, "HALTED: extractor misconfigured, this node is NOT sensing. "
                           "%u subcarriers @ %.0f Hz does not fit the compiled capacities; "
                           "rebuild with matching Kconfig.",
                     static_cast<unsigned>(cfg::kNSubcarriers),
                     static_cast<double>(cfg::kCsiRateHz));
            telemetry::log_health();
            esp_task_wdt_reset();
            vTaskDelay(pdMS_TO_TICKS(cfg::kTelemetryS * 1000));
        }
    }

    // Before the first frame: this advances and commits the boot counter
    // the frames of this boot are signed with (ADR-012).
    const std::optional<frame_auth::Key> key = frame_auth::load();

    ESP_ERROR_CHECK(csi::start());
    log_startup_expectations();

    esp_task_wdt_add(nullptr);

    csi::Frame frame{};
    std::int64_t last_report_us = esp_timer_get_time();
    std::uint32_t captured = 0;
    std::uint32_t emitted = 0;
    std::uint32_t send_failures = 0;
    std::uint32_t last_overflows = 0;

#if CONFIG_SAANS_RAW_MODE
    std::uint8_t raw[csi::kRawHeaderSize + 2 * cfg::kNSubcarriers];
    const std::uint32_t freq_mhz = csi::channel_to_mhz(cfg::kWifiChannel);
    ESP_LOGW(kTag, "raw mode ON — raw CSI is leaving this node (flagged in every frame)");
    ESP_LOGW(kTag, "raw CSI is never signed: a gateway with [[frame_key]] refuses it");
#endif

    for (;;) {
        esp_task_wdt_reset();

        if (!csi::receive(frame, 1000)) {
            // No CSI for a second. Usually the TX is down or on another
            // channel. Say so: silence from a sensing node reads downstream as
            // "nothing happening in that room", which is the dangerous failure.
            ESP_LOGW(kTag, "no CSI for 1 s (TX down, or channel mismatch?)");
            continue;
        }
        ++captured;

        {
            // A view over the queue element — CsiFrame holds non-owning
            // pointers, so nothing is copied here.
            CsiFrame view{};
            view.amplitudes     = frame.amplitudes;
            view.phases         = frame.phases;
            view.n_subcarriers  = frame.n_subcarriers;
            view.sample_index   = frame.sequence;
            view.sample_rate_hz = cfg::kCsiRateHz;
            view.rssi_dbm       = frame.rssi_dbm;
            view.node_id        = cfg::kNodeId;

            const double t_s = static_cast<double>(frame.timestamp_us) / 1e6;
            const auto out = g_extractor.push(view, frame.sequence, t_s,
                                              frame.noise_floor_dbm);

            for (const FeatureFrame& f : out) {
                // Both encoders apply the consent gate as a backstop before
                // writing bytes, so there is no path from here to the wire that
                // skips it. Signed when a key is provisioned, never both.
                esp_err_t sent;
                if (key) {
                    g_extractor.encode_signed(f, key->node_key, key->key_id, key->boot,
                                              g_wire_signed);
                    sent = net::udp_send(g_wire_signed, sizeof(g_wire_signed));
                } else {
                    g_extractor.encode(f, g_wire);
                    sent = net::udp_send(g_wire, sizeof(g_wire));
                }
                if (sent == ESP_OK) {
                    ++emitted;
                } else {
                    ++send_failures;
                }
            }
        }

#if CONFIG_SAANS_RAW_MODE
        // Debug only, and visible: raw_mode is a flag in every FeatureFrame
        // precisely so this cannot happen quietly. The ORIGINAL I/Q bytes are
        // sent, not a reconstruction from amplitude and phase.
        {
            const std::size_t n = csi::serialize_raw(
                frame.iq, frame.iq_len, cfg::kNodeId, frame.sequence, frame.rssi_dbm,
                frame.noise_floor_dbm, freq_mhz, frame.agc_gain, frame.fft_gain, raw, sizeof(raw));
            if (n > 0) {
                net::udp_send(raw, n);
            }
        }
#endif

        const std::int64_t now_us = esp_timer_get_time();
        if (now_us - last_report_us >= cfg::kTelemetryS * 1000000LL) {
            const float elapsed_s = static_cast<float>(now_us - last_report_us) / 1e6f;
            const float delivered_hz = static_cast<float>(captured) / elapsed_s;
            ESP_LOGI(kTag, "csi=%.1f Hz frames=%lu dropped=%lu send_fail=%lu",
                     static_cast<double>(delivered_hz),
                     static_cast<unsigned long>(emitted),
                     static_cast<unsigned long>(csi::dropped()),
                     static_cast<unsigned long>(send_failures));
            if (!key) {
                // Every report, not once at boot: a gateway with frame keys
                // refuses these frames, and from there this node looks dead.
                ESP_LOGW(kTag, "frames are UNSIGNED (no key provisioned): a gateway with "
                               "[[frame_key]] refuses them (ADR-012)");
            }

            // WHAT THE RADIO IS ACTUALLY DELIVERING. The DSP is compiled for
            // a fixed subcarrier count on the assumption that these frames
            // are HT20; nothing in this firmware can enforce that, because it
            // depends on the transmitter and the driver. Until a bench run
            // has confirmed it, this line is the evidence.
            const csi::Stats cs = csi::capture_stats();
            ESP_LOGI(kTag, "capture: len=%u (first %u) -> %u subcarriers, expecting %u; "
                           "2nd_ch=%u first_word_invalid=%lu short=%lu layout_changed=%lu too_short=%lu "
                           "not_ht=%lu 40mhz=%lu",
                     static_cast<unsigned>(cs.last_len), static_cast<unsigned>(cs.first_len),
                     static_cast<unsigned>(cs.last_len / 2U), static_cast<unsigned>(cfg::kNSubcarriers),
                     static_cast<unsigned>(cs.secondary_channel),
                     static_cast<unsigned long>(cs.first_word_invalid),
                     static_cast<unsigned long>(cs.short_frames),
                     static_cast<unsigned long>(cs.layout_changed),
                     static_cast<unsigned long>(cs.dropped_short),
                     static_cast<unsigned long>(cs.not_ht),
                     static_cast<unsigned long>(cs.wide));
            if (cs.not_ht != 0U) {
                ESP_LOGW(kTag, "%lu frames arrived without HT (L-LTF only): those measure a "
                               "different set of subcarriers. The TX should log "
                               "'ESP-NOW rate: HT20 MCS0 LGI'.",
                         static_cast<unsigned long>(cs.not_ht));
            }

            // THE NUMBER PROTOCOL_S5 §1.1 ASKS FOR: how often the receiver
            // changes its gain. Cumulative since boot; on an idle node in an
            // empty room for ten minutes, `agc moved on N of M frames` is the
            // fraction to report. The source of the values is undocumented and
            // unconfirmed until the bench check in §1.1 (csi_capture.cpp).
            if (cs.gain_readable && cs.gain_frames > 1U) {
                const double pairs = static_cast<double>(cs.gain_frames - 1U);
                ESP_LOGI(kTag, "gain: agc=%u (range %u-%u) fft=%d; agc moved on %lu of %lu frame "
                               "pairs (%.3f%%), fft on %lu",
                         static_cast<unsigned>(cs.agc_gain), static_cast<unsigned>(cs.agc_min),
                         static_cast<unsigned>(cs.agc_max), static_cast<int>(cs.fft_gain),
                         static_cast<unsigned long>(cs.agc_moves),
                         static_cast<unsigned long>(cs.gain_frames - 1U),
                         100.0 * static_cast<double>(cs.agc_moves) / pairs,
                         static_cast<unsigned long>(cs.fft_moves));
            } else if (!cs.gain_readable) {
                ESP_LOGW(kTag, "gain: not readable on this target; PROTOCOL_S5 §1.1 needs the "
                               "RSSI proxy instead");
            }
            if (cs.last_len != 0U && cs.last_len / 2U < cfg::kNSubcarriers) {
                ESP_LOGW(kTag, "THE RADIO IS DELIVERING FEWER SUBCARRIERS THAN THE DSP EXPECTS. "
                               "The tail is zero-padded, which the BNR stage reads as a quiet "
                               "channel. Check that the TX is sending HT20 (tx_role sets an MCS "
                               "rate) and that CONFIG_SAANS_N_SUBCARRIERS matches what arrives.");
            }
            if (cs.layout_changed != 0U) {
                ESP_LOGW(kTag, "CSI buffer length changed mid-run (%lu times): the subcarrier "
                               "indices do not mean the same thing across frames, and the static "
                               "suppressor is averaging two different channels together.",
                         static_cast<unsigned long>(cs.layout_changed));
            }

            // THE DELIVERED RATE IS NOT THE CONFIGURED RATE, AND THAT IS NOT
            // COSMETIC. Nothing on this path decimates — csi_capture queues
            // every callback and the loop above pushes every queued frame — so
            // a radio running faster than the DSP was told feeds it more
            // samples per second than it believes it is getting. Its sample
            // clock, its vitals bands and its gap filler are then all scaled
            // by the ratio, and the vitals are confidently wrong rather than
            // absent. Arriving SLOWER is fine: that is what the gap filler is
            // for.
            //
            // The arrival ring is sized with 25% headroom over the configured
            // rate (dsp/node_config.hpp), so that is the threshold at which
            // the link statistics also start being computed over less than the
            // spec's 10 s. arrival_overflows is reported alongside because it
            // is the DSP's own account of the same condition.
            if (delivered_hz > cfg::kCsiRateHz * 1.25f) {
                ESP_LOGE(kTag,
                         "CSI arriving at %.1f Hz but the DSP is configured for %.0f Hz "
                         "— nothing decimates, so vitals and link statistics are both "
                         "wrong. Set the TX beacon rate to %.0f Hz. "
                         "(link-stats window overflows: %lu)",
                         static_cast<double>(delivered_hz),
                         static_cast<double>(cfg::kCsiRateHz),
                         static_cast<double>(cfg::kCsiRateHz),
                         static_cast<unsigned long>(g_extractor.arrival_overflows));
            } else if (g_extractor.arrival_overflows != last_overflows) {
                // Below the warning threshold but the ring still evicted:
                // burstier than the headroom covers. The density is still
                // right, the span is not.
                //
                // The DELTA, not the running total: the counter never resets,
                // so reporting the total would keep announcing a condition
                // that had already cleared, which is how a real one stops
                // being read.
                ESP_LOGW(kTag,
                         "link-stats window overflowed %lu time(s) this interval: "
                         "csi_rate_chz and loss_permille are over a shorter span than 10 s",
                         static_cast<unsigned long>(g_extractor.arrival_overflows -
                                                    last_overflows));
            }
            last_overflows = g_extractor.arrival_overflows;
            telemetry::log_health();
            captured = 0;
            emitted = 0;
            send_failures = 0;
            last_report_us = now_us;
        }
    }
}

}  // namespace saans
