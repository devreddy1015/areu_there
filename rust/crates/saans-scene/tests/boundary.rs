//! ADR-008 decision 1: geometry is declared, never inferred. Nothing that
//! carries CSI may write a scene.
//!
//! The mechanism is the `write` cargo feature: `Serialize` on the scene
//! types, `Scene::save`, `Scene::to_json_pretty` and the RoomPlan importer
//! exist only with it. Human tools enable it; a crate that carries CSI may
//! READ scenes (the simulator must) but may not enable it, so it cannot
//! write a scene by any route.
//!
//! Both lists are allowlists. A new crate that depends on saans-scene is
//! forbidden until someone decides which list it belongs on.

use std::path::Path;

/// May enable `write`: tools a person runs to make or edit scene files.
const WRITERS: &[(&str, &str)] = &[(
    "saans-cli",
    "a human tool: imports RoomPlan scans and writes scene files; its simulated exports \
     (`export`, `roomsim-export`) only read a scene, and no command writes one from CSI",
)];

/// May depend on saans-scene WITHOUT `write`: they read scenes to simulate
/// or draw them, and carry or produce CSI.
const READERS: &[(&str, &str)] = &[
    ("saans-roomsim", "simulates the CSI a declared room would produce"),
    (
        "saans-locate",
        "locates simulated people and a simulated phone; reads scenes only under its `roomsim` feature and in its tests",
    ),
];

/// Does this manifest declare a dependency on saans-scene, in any table?
fn depends_on_scene(manifest: &str) -> bool {
    manifest.lines().map(str::trim).any(|l| {
        !l.starts_with('#')
            && (l.starts_with("saans-scene ")
                || l.starts_with("saans-scene=")
                || l.starts_with("saans-scene.")
                || (l.starts_with('[') && l.contains("dependencies.saans-scene")))
    })
}

/// Does this manifest switch on saans-scene's `write` feature — inline
/// (`saans-scene = { ..., features = ["write"] }`), in a
/// `[dependencies.saans-scene]` table, or by forwarding
/// (`some-feature = ["saans-scene/write"]`)?
fn enables_write(manifest: &str) -> bool {
    let mut in_scene_table = false;
    for line in manifest.lines().map(str::trim) {
        if line.starts_with('#') {
            continue;
        }
        if line.contains("saans-scene/write") {
            return true;
        }
        if line.starts_with('[') {
            in_scene_table = line.contains("dependencies.saans-scene");
            continue;
        }
        let inline = line.starts_with("saans-scene ") || line.starts_with("saans-scene=");
        if (inline || (in_scene_table && line.starts_with("features"))) && line.contains("\"write\"") {
            return true;
        }
    }
    false
}

#[test]
fn only_writers_enable_write_and_only_known_readers_depend_on_the_scene_crate() {
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
        let writer = WRITERS.iter().any(|(ok, _)| *ok == name);
        let reader = READERS.iter().any(|(ok, _)| *ok == name);
        if enables_write(&manifest) {
            assert!(
                writer,
                "ADR-008 decision 1: {name} enables saans-scene's `write` feature but is not a writer. \
                 A crate that carries CSI may read scenes, never write them."
            );
        } else {
            assert!(
                writer || reader,
                "ADR-008 decision 1: {name} now depends on saans-scene but is on neither list. \
                 Decide whether it may only read scenes (READERS) or is a human tool that writes them (WRITERS)."
            );
        }
    }
    // Guard against the test passing because it looked in the wrong place.
    assert!(checked >= 6, "found only {checked} crates; is the workspace layout different?");
}

#[test]
fn the_detectors_see_every_way_a_dependency_and_the_feature_can_be_written() {
    // Checked against hand-written manifests, not against this crate's own
    // code, so a blind spot in a detector shows up here.
    for yes in [
        "[dependencies]\nsaans-scene = { path = \"../saans-scene\" }\n",
        "[dev-dependencies]\nsaans-scene={ path = \"../saans-scene\" }\n",
        "[dependencies]\nsaans-scene.workspace = true\n",
        "[dependencies.saans-scene]\npath = \"../saans-scene\"\n",
        "[target.'cfg(unix)'.dependencies.saans-scene]\npath = \"x\"\n",
    ] {
        assert!(depends_on_scene(yes), "missed a dependency: {yes}");
    }
    for no in [
        "[dependencies]\nsaans-sim = { path = \"../saans-sim\" }\n",
        "# saans-scene = { path = \"../saans-scene\" }\n",
        "description = \"talks about saans-scene in prose\"\n",
    ] {
        assert!(!depends_on_scene(no), "false dependency: {no}");
    }
    for yes in [
        "[dependencies]\nsaans-scene = { path = \"../saans-scene\", features = [\"write\"] }\n",
        "[dependencies.saans-scene]\npath = \"../saans-scene\"\nfeatures = [\"write\"]\n",
        "[features]\nedit = [\"saans-scene/write\"]\n[dependencies]\nsaans-scene = { path = \"../saans-scene\" }\n",
    ] {
        assert!(enables_write(yes), "missed write: {yes}");
    }
    for no in [
        "[dependencies]\nsaans-scene = { path = \"../saans-scene\" }\n",
        "[dependencies.saans-scene]\npath = \"../saans-scene\"\n[features]\nwrite = []\n",
        "[dependencies]\nsaans-scene = { path = \"../saans-scene\" }\n# features = [\"write\"]\n",
        "[dependencies]\nother = { path = \"x\", features = [\"write\"] }\nsaans-scene = { path = \"../saans-scene\" }\n",
    ] {
        assert!(!enables_write(no), "false write: {no}");
    }
}
