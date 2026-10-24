#include "frame_auth.h"

#include <cstring>

#include "esp_log.h"
#include "nvs.h"
#include "nvs_flash.h"

namespace saans {
namespace frame_auth {
namespace {

constexpr const char* kTag = "saans.auth";
// Must match edge/firmware/partitions.csv and edge/build.sh provision.
constexpr const char* kPartition = "saanskey";
constexpr const char* kNamespace = "frameauth";

}  // namespace

std::optional<Key> load() {
    esp_err_t err = nvs_flash_init_partition(kPartition);
    if (err != ESP_OK) {
        ESP_LOGW(kTag, "no '%s' partition (%s): frames go out UNSIGNED", kPartition,
                 esp_err_to_name(err));
        return std::nullopt;
    }
    nvs_handle_t h;
    err = nvs_open_from_partition(kPartition, kNamespace, NVS_READWRITE, &h);
    if (err != ESP_OK) {
        ESP_LOGW(kTag, "no frame key provisioned (%s): frames go out UNSIGNED; "
                       "see edge/build.sh provision",
                 esp_err_to_name(err));
        return std::nullopt;
    }

    Key key{};
    std::size_t len = sizeof(key.node_key);
    err = nvs_get_blob(h, "key", key.node_key, &len);
    if (err != ESP_OK || len != sizeof(key.node_key)) {
        ESP_LOGE(kTag, "frame key unreadable or not %u bytes: frames go out UNSIGNED",
                 static_cast<unsigned>(sizeof(key.node_key)));
        nvs_close(h);
        return std::nullopt;
    }
    if (nvs_get_u8(h, "key_id", &key.key_id) != ESP_OK) {
        ESP_LOGE(kTag, "frame key has no key_id: frames go out UNSIGNED");
        std::memset(key.node_key, 0, sizeof(key.node_key));
        nvs_close(h);
        return std::nullopt;
    }

    // The counter the gateway's replay floor is compared against. Provisioning
    // starts it at the Unix time, so a re-provisioned node is always above
    // anything the gateway saw before; each boot adds one.
    std::uint32_t boot = 0;
    nvs_get_u32(h, "boot", &boot);  // absent: provisioned without one, start at 0
    if (boot == UINT32_MAX) {
        ESP_LOGE(kTag, "boot counter exhausted: re-provision this node; frames go out UNSIGNED");
        std::memset(key.node_key, 0, sizeof(key.node_key));
        nvs_close(h);
        return std::nullopt;
    }
    key.boot = boot + 1;
    // Committed before the first frame is signed with it. If this fails the
    // frames of this boot are still good, but the next boot repeats this
    // counter and the gateway refuses that boot's frames as replays: loud,
    // and safe, which is the trade ADR-012 chose.
    if (nvs_set_u32(h, "boot", key.boot) != ESP_OK || nvs_commit(h) != ESP_OK) {
        ESP_LOGE(kTag, "cannot persist the boot counter: after the next reboot the gateway "
                       "will refuse this node's frames until it is re-provisioned");
    }
    nvs_close(h);
    // Never the key, not even a prefix of it.
    ESP_LOGI(kTag, "frames are SIGNED: key_id %u, boot %lu", static_cast<unsigned>(key.key_id),
             static_cast<unsigned long>(key.boot));
    return key;
}

}  // namespace frame_auth
}  // namespace saans
