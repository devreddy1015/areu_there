//! The recorded Locate runs `--room-replay` loads (spec/livedemo.md §6.2):
//! read with saans-locate's records reader, then matched to an embedded
//! scene by the sha256 of its file. SIM: every record in them is.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use saans_locate::records::{read_records, Records, MAX_BYTES};

use super::engine::q_end;
use super::Embedded;
use crate::auth::sha256_hex;

/// One loaded file: the embedded scene its hash names, and its records.
pub(crate) struct Recorded {
    pub(crate) scene: Embedded,
    pub(crate) records: Records,
}

/// The recorded runs offered as REPLAY (§6), at most one per embedded
/// scene. Opaque: nothing in it is reachable from outside the crate but
/// the scenes' ids, and it can only be loaded from files and installed on
/// a gateway ([`super::RoomStreams::install_replays`]).
pub struct RoomReplays {
    runs: Vec<Arc<Recorded>>,
}

impl RoomReplays {
    /// Read and check `--room-replay` files (§6.2). Every refusal of
    /// saans-locate's reader, then: the scene's sha256 must match an
    /// embedded scene and no other file may cover that scene; the
    /// manifest's scene name and provenance must be that scene's; and when
    /// `roomsim_config.n_people` is unset, the first tick's truth must have
    /// the embedded scene's people count. The error names the file and
    /// the line, never a value from the file.
    pub fn load(paths: &[PathBuf]) -> Result<RoomReplays, String> {
        let mut runs: Vec<(Arc<Recorded>, &Path)> = Vec::new();
        for path in paths {
            let at = |line: usize, why: &str| format!("{}: line {line}: {why}", path.display());
            let file = File::open(path)
                .map_err(|e| format!("{}: cannot be read ({e})", path.display()))?;
            let records = read_records(BufReader::new(file), MAX_BYTES)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            let m = &records.manifest;
            let scene = Embedded::all()
                .find(|e| sha256_hex(e.text) == m.scene_sha256)
                .ok_or_else(|| {
                    at(
                        1,
                        "scene.sha256 matches no embedded scene (spec/livedemo.md §6.2)",
                    )
                })?;
            if let Some((_, other)) = runs.iter().find(|(r, _)| r.scene == scene) {
                return Err(at(
                    1,
                    &format!(
                        "scene: {} already covers this scene; one file per scene (spec/livedemo.md §6.2)",
                        other.display()
                    ),
                ));
            }
            let declared = scene
                .load()
                .ok_or_else(|| at(1, "scene: the embedded scene does not load"))?;
            if m.scene_name != declared.name || m.provenance != declared.provenance {
                return Err(at(
                    1,
                    "scene: its name or provenance is not the embedded scene's of that sha256",
                ));
            }
            if q_end(m.duration_s).is_none() {
                return Err(at(
                    1,
                    "duration_s: 40 * duration_s is not an integer (spec/livedemo.md §3.2)",
                ));
            }
            if m.roomsim_config.n_people.is_none() {
                if let Some(first) = records.ticks.first() {
                    if first.truth.len() != declared.people.len() {
                        // Records are ordered by t, the epoch first at equal t.
                        let line = 2 + records.epochs.iter().take_while(|e| e.t <= first.t).count();
                        return Err(at(
                            line,
                            &format!(
                                "truth: the first tick does not have the embedded scene's {} people, and roomsim_config.n_people is unset (spec/livedemo.md §6.2)",
                                declared.people.len()
                            ),
                        ));
                    }
                }
            }
            runs.push((Arc::new(Recorded { scene, records }), path));
        }
        Ok(RoomReplays {
            runs: runs.into_iter().map(|(r, _)| r).collect(),
        })
    }

    pub fn len(&self) -> usize {
        self.runs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    /// The ids of the embedded scenes covered, in the order loaded.
    pub fn scene_ids(&self) -> Vec<&'static str> {
        self.runs.iter().map(|r| r.scene.id).collect()
    }

    pub(crate) fn recorded(&self, scene: Embedded) -> Option<Arc<Recorded>> {
        self.runs.iter().find(|r| r.scene == scene).cloned()
    }
}

/// Counts and scene ids only: no record, and no line of a scene file.
impl std::fmt::Debug for RoomReplays {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoomReplays")
            .field("scenes", &self.scene_ids())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roomstream::testkit::*;
    use crate::roomstream::RoomStreams;
    use saans_locate::records::write_records;
    use saans_locate::variant::Variant;

    fn write_to(path: &Path, records: &Records) {
        let mut w = std::io::BufWriter::new(File::create(path).unwrap());
        write_records(&mut w, records).unwrap();
    }

    /// §6.2: a written run loads and covers its scene, once per gateway;
    /// each gateway-side refusal names the file's line and no value from
    /// it.
    #[test]
    fn recorded_runs_are_matched_to_embedded_scenes_and_refused_otherwise() {
        let bedroom = Embedded::find("bedroom").unwrap();
        let fx = sim_fixture("bedroom", Variant::Declared, 5.0);
        let records = records_of(&fx);
        let loaded = load_written(&records).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(loaded.scene_ids(), ["bedroom"]);
        assert_eq!(
            format!("{loaded:?}"),
            "RoomReplays { scenes: [\"bedroom\"] }"
        );
        assert!(loaded.recorded(Embedded::find("office").unwrap()).is_none());
        let room = RoomStreams::new();
        assert!(room.recorded(bedroom).is_none());
        room.install_replays(loaded).unwrap();
        assert!(room.recorded(bedroom).is_some());
        assert!(
            room.install_replays(RoomReplays { runs: Vec::new() })
                .is_err(),
            "once per gateway"
        );

        let mut unknown = records.clone();
        unknown.manifest.scene_sha256 = "0".repeat(64);
        let e = load_written(&unknown).unwrap_err();
        assert!(
            e.ends_with("line 1: scene.sha256 matches no embedded scene (spec/livedemo.md §6.2)"),
            "{e}"
        );

        let mut renamed = records.clone();
        renamed.manifest.scene_name = "zzq".into();
        let e = load_written(&renamed).unwrap_err();
        assert!(
            e.contains("line 1: scene: its name or provenance") && !e.contains("zzq"),
            "{e}"
        );

        // n_people unset: the first tick must have the scene's people.
        assert_eq!(records.manifest.roomsim_config.n_people, None);
        let mut crowded = records.clone();
        for k in &mut crowded.ticks {
            let p = k.truth[0];
            k.truth.push(p);
        }
        let line = 2 + crowded
            .epochs
            .iter()
            .take_while(|e| e.t <= crowded.ticks[0].t)
            .count();
        let e = load_written(&crowded).unwrap_err();
        assert!(
            e.contains(&format!(
                "line {line}: truth: the first tick does not have the embedded scene's 1 people"
            )),
            "{e}"
        );

        // One file per scene.
        let (a, b) = (temp_file("a.jsonl"), temp_file("b.jsonl"));
        write_to(&a, &records);
        write_to(&b, &records);
        let e = RoomReplays::load(&[a.clone(), b.clone()]).unwrap_err();
        assert!(
            e.starts_with(&format!(
                "{}: line 1: scene: {} already covers",
                b.display(),
                a.display()
            )),
            "{e}"
        );
        assert!(RoomReplays::load(std::slice::from_ref(&a)).is_ok());
        for p in [&a, &b] {
            let _ = std::fs::remove_file(p);
        }
        let e = RoomReplays::load(&[temp_file("missing.jsonl")]).unwrap_err();
        assert!(e.contains("missing.jsonl: cannot be read"), "{e}");
        assert!(RoomReplays::load(&[]).unwrap().is_empty());
    }
}
