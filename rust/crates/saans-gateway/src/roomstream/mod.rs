//! The room stream (spec/livedemo.md §2-§7): `GET /api/roomsim`, a
//! WebSocket on which every connection gets a run of its own. Every
//! number on it is SIM.
//!
//! ```text
//! handshake: Origin/Host ─► principal ─► can_see_position(None) ─► slot     (session.rs)
//! connection task (runtime) ── commands, std mpsc ──────────► thread `roomsim-session`
//!                           ◄── messages, tokio mpsc(1), blocking_send ──┘
//! thread: calibrating (RoomSimSetup, empty run) ─► Engine (engine.rs) ─► frames (frame.rs)
//! ```
//!
//! - **SIM**: Room Sim and Locate run in the session's thread on the
//!   embedded example scene the query names; frames carry truth and
//!   errors.
//! - **REPLAY**: a recorded Locate run that `--room-replay` loaded (§6,
//!   [`RoomReplays`]), labelled SIM, with truth deleted.
//! - **LIVE**: none. A session's mode is `sim` or `replay` whatever the
//!   gateway's own mode is (§7); nothing here can produce a LIVE frame.
//!
//! A frame goes only to the socket of the connection that asked for it.
//! Nothing here touches `Shared::live_tx`, `Shared::outputs`,
//! `Published`, the event log, the history or the disk (§2.5, ADR-008
//! decision 11). The only trace a request leaves is the audit `Denied`
//! record of a role refusal (§2.3), which carries no position.
//!
//! No scene's `Person.id` or `Phone.id` is sent, logged or kept: people
//! are indices, and the phone is the phone (ADR-008 decision 3).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use saans_scene::Scene;

use crate::roomsim::SCENES;
use crate::shared::Shared;

mod engine;
mod frame;
mod params;
mod replay;
mod session;
#[cfg(test)]
mod testkit;

pub use replay::RoomReplays;
pub(crate) use session::handler;

/// At most this many sessions run at once per gateway (§2.4). Sessions
/// that are still calibrating count, and so do REPLAY sessions.
pub const ROOMSIM_MAX_SESSIONS: usize = 4;

/// The length of a SIM session's run, SIM seconds (§3.1).
pub const ROOMSIM_RUN_S: f64 = 120.0;

/// The room stream's handle on [`Shared`] (spec/livedemo.md §2.5).
///
/// Opaque on purpose: it holds the per-gateway session counter and the
/// recorded runs `--room-replay` loaded, and no method of it yields a
/// frame, a record or a position.
pub struct RoomStreams {
    /// Session threads alive (§2.4); never a `static`, so two gateways in
    /// one process do not share a cap.
    sessions: AtomicUsize,
    /// Set once, by the binary, after the gateway starts (§6.2).
    replays: OnceLock<RoomReplays>,
}

impl RoomStreams {
    pub(crate) fn new() -> Self {
        Self {
            sessions: AtomicUsize::new(0),
            replays: OnceLock::new(),
        }
    }

    /// Offer these recorded runs on the room stream (`replay=1`, §6.3).
    /// Once per gateway: a second call is refused and gives them back.
    pub fn install_replays(&self, replays: RoomReplays) -> Result<(), RoomReplays> {
        self.replays.set(replays)
    }

    /// The recorded run of this embedded scene, if one was installed.
    fn recorded(&self, scene: Embedded) -> Option<Arc<replay::Recorded>> {
        self.replays.get()?.recorded(scene)
    }
}

/// A session slot (§2.4): taken in the handshake, moved into the
/// `on_upgrade` closure and then into the session thread, and released
/// when it is dropped there, i.e. when the thread exits. A failed upgrade
/// drops the closure, and the slot with it.
pub(crate) struct Slot {
    shared: Arc<Shared>,
}

impl Slot {
    /// A slot, or `None` when [`ROOMSIM_MAX_SESSIONS`] are taken. The
    /// increment is undone at once when the cap was already reached.
    pub(crate) fn take(shared: &Arc<Shared>) -> Option<Slot> {
        let prev = shared.room.sessions.fetch_add(1, Ordering::AcqRel);
        if prev >= ROOMSIM_MAX_SESSIONS {
            shared.room.sessions.fetch_sub(1, Ordering::AcqRel);
            return None;
        }
        Some(Slot {
            shared: Arc::clone(shared),
        })
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        self.shared.room.sessions.fetch_sub(1, Ordering::AcqRel);
    }
}

/// One of the example scenes compiled into the binary
/// (`crate::roomsim::SCENES`). No `Debug`: the file text carries the
/// people's and the phone's ids.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Embedded {
    pub(crate) index: usize,
    /// The scene's id in `/roomsim/scenes` (declared geometry, sent).
    pub(crate) id: &'static str,
    /// The file as embedded; its sha256 names it (ADR-008 decision 4).
    pub(crate) text: &'static str,
}

impl Embedded {
    pub(crate) fn all() -> impl Iterator<Item = Embedded> {
        SCENES
            .iter()
            .enumerate()
            .map(|(index, &(id, text))| Embedded { index, id, text })
    }

    pub(crate) fn find(id: &str) -> Option<Embedded> {
        Self::all().find(|e| e.id == id)
    }

    /// The scene, parsed and validated by saans-scene. `None` cannot
    /// happen for a shipped scene (saans-scene's tests load each one).
    pub(crate) fn load(self) -> Option<Scene> {
        Scene::from_json_str(self.id, self.text).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::Mode;

    #[test]
    fn slots_are_capped_per_gateway_and_returned_on_drop() {
        let a = Shared::new(Mode::Live);
        let b = Shared::new(Mode::Live);
        let held: Vec<Slot> = (0..ROOMSIM_MAX_SESSIONS)
            .map(|_| Slot::take(&a).expect("under the cap"))
            .collect();
        assert!(Slot::take(&a).is_none(), "the fifth session is refused");
        assert_eq!(
            a.room.sessions.load(Ordering::Acquire),
            ROOMSIM_MAX_SESSIONS,
            "the refusal undid its increment"
        );
        assert!(
            Slot::take(&b).is_some(),
            "another gateway has its own counter"
        );
        drop(held);
        assert_eq!(a.room.sessions.load(Ordering::Acquire), 0);
        assert!(Slot::take(&a).is_some());
    }

    #[test]
    fn every_embedded_scene_loads_and_is_found_by_its_id() {
        assert_eq!(Embedded::all().count(), SCENES.len());
        for e in Embedded::all() {
            assert!(e.load().is_some(), "{}", e.id);
            assert!(Embedded::find(e.id) == Some(e));
        }
        assert!(Embedded::find("nope").is_none());
    }
}
