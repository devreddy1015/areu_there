//! ADR-008 decision 1: geometry is declared, never inferred. Nothing that
//! carries CSI may write a scene.
//!
//! The ADR's eventual mechanism is a `write` cargo feature on this crate
//! that only `saans-cli` enables. Today something stronger holds and is
//! asserted here instead: no crate except `saans-cli` depends on
//! `saans-scene` at all, so no CSI-carrying crate can even read a scene,
//! let alone write one.
//!
//! The allowlist is the point. A new crate is forbidden by default, and
//! the day Phase 5 makes the gateway read scenes this test fails and says
//! what has to happen first.

use std::path::Path;

/// Crates allowed to depend on saans-scene, and why.
const ALLOWED: &[(&str, &str)] = &[("saans-cli", "imports RoomPlan scans and writes scene files: a human tool, no CSI")];

/// Does this manifest declare a dependency on saans-scene, in any table
/// (`[dependencies]`, `[dev-dependencies]`, `[build-dependencies]`,
/// target-specific, or the `[dependencies.saans-scene]` form)?
fn depends_on_scene(manifest: &str) -> bool {
    manifest.lines().map(str::trim).any(|l| {
        !l.starts_with('#')
            && (l.starts_with("saans-scene ")
                || l.starts_with("saans-scene=")
                || l.starts_with("saans-scene.")
                || (l.starts_with('[') && l.contains("dependencies.saans-scene")))
    })
}

#[test]
fn no_crate_that_carries_csi_depends_on_the_scene_crate() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let mut checked = 0;
    for entry in std::fs::read_dir(crates).unwrap() {
        let dir = entry.unwrap().path();
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let Ok(manifest) = std::fs::read_to_string(dir.join("Cargo.toml")) else { continue };
        checked += 1;
        if name == "saans-scene" || !depends_on_scene(&manifest) {
            continue;
        }
        assert!(
            ALLOWED.iter().any(|(ok, _)| *ok == name),
            "ADR-008 decision 1: {name} now depends on saans-scene.\n\
             Before a crate that carries CSI may read scenes, put the scene-writing APIs \
             (Scene::save, Scene::to_json_pretty, the roomplan module) behind a `write` \
             feature that {name} does not enable, and change this test to assert that \
             instead of forbidding the dependency."
        );
    }
    // Guard against the test passing because it looked in the wrong place.
    assert!(checked >= 6, "found only {checked} crates; is the workspace layout different?");
}

#[test]
fn the_detector_sees_every_way_a_dependency_can_be_written() {
    // Checked against hand-written manifests, not against this crate's
    // own code, so a blind spot in the detector shows up here.
    for yes in [
        "[dependencies]\nsaans-scene = { path = \"../saans-scene\" }\n",
        "[dev-dependencies]\nsaans-scene={ path = \"../saans-scene\" }\n",
        "[dependencies]\nsaans-scene.workspace = true\n",
        "[dependencies.saans-scene]\npath = \"../saans-scene\"\n",
        "[target.'cfg(unix)'.dependencies.saans-scene]\npath = \"x\"\n",
    ] {
        assert!(depends_on_scene(yes), "missed: {yes}");
    }
    for no in [
        "[dependencies]\nsaans-sim = { path = \"../saans-sim\" }\n",
        "# saans-scene = { path = \"../saans-scene\" }\n",
        "description = \"talks about saans-scene in prose\"\n",
    ] {
        assert!(!depends_on_scene(no), "false positive: {no}");
    }
}
