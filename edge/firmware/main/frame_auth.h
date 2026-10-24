#pragma once
// The node's side of signed frames (ADR-012, spec/wire.md §2b): its key,
// the key_id it was derived under, and the boot counter, all from the
// `saanskey` NVS partition that `edge/build.sh provision` writes.
//
// The partition is separate from the default `nvs` one on purpose: IDF
// erases `nvs` when its format version changes, and that must not take the
// key or the counter with it.

#include <cstdint>
#include <optional>

#include "feature_frame.hpp"

namespace saans {
namespace frame_auth {

struct Key {
    std::uint8_t node_key[kFrameKeySize];
    std::uint8_t key_id;
    /// This boot's counter, already advanced and committed.
    std::uint32_t boot;
};

/// Read the key and advance the boot counter. Call once, before the first
/// frame. Empty when no key is provisioned: the node then sends unsigned
/// (v1) frames, says so at every telemetry report, and a gateway with
/// `[[frame_key]]` refuses them.
std::optional<Key> load();

}  // namespace frame_auth
}  // namespace saans
