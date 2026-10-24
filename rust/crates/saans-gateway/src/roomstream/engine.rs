//! The session engine (spec/livedemo.md §3.2-§3.4, §4, §5): the session
//! clock, the commands and the frames, with no socket and no wall clock.
//! The caller passes the time (`now`, since the engine started) into
//! every call, so tests drive a whole session in microseconds; the
//! session thread (session.rs) is a thin driver around it.
//!
//! **Clock.** Simulated time moves in quanta of 0.025 s: `q` counts them
//! and `T = float(2*q) * slot_s`, Room Sim's own formula, exact for the
//! only `slot_s` of Phase 5. A run's first frame is at `q = 0` (so a
//! restart returns `t` to 0, and the epoch at `t = 0` is in the first
//! `new`); then every 100 ms deadline a playing run takes
//! `q = min(q + 4*speed, q_end)`, advances its source to `T` and sends one
//! frame. A missed deadline is skipped, never caught up. The frame at
//! `q_end` says `ended`, and nothing follows it until a restart, whose
//! first frame goes at the next 100 ms deadline.
//!
//! **Pause.** `q` stops. The next deadline sends a frame with
//! `paused: true` and the same `t`, then one heartbeat a second (seq + 1,
//! the same `t`, empty `new`). Paused or ended for 10 minutes, the session
//! closes (§2.4).

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use saans_locate::phone::EpochRecord;
use saans_locate::pipeline::{PipelineError, TickRecord};
use saans_locate::records::SLOT_S;
use saans_locate::score::{score, FalseAlarms};
use saans_locate::stream::{Advance, LinkValue, RoomSimSetup, RoomSimStream};
use saans_locate::variant::Variant;
use saans_scene::Scene;
use serde_json::Value;

use super::frame::{error_message, run_block, scene_block, shows_truth, Frame};
use super::params::{Command, Speed};
use super::replay::Recorded;
use super::Embedded;
use crate::shared::Mode;

/// Wall time between two frames of a playing session (10 Hz).
pub(crate) const FRAME_PERIOD: Duration = Duration::from_millis(100);
/// Wall time between two heartbeats of a paused session.
pub(crate) const HEARTBEAT_PERIOD: Duration = Duration::from_secs(1);
/// Paused or ended this long, a session closes (§2.4).
pub(crate) const IDLE_LIMIT: Duration = Duration::from_secs(600);
/// `effective_speed` is measured over this much wall time (§3.2).
const RATE_WINDOW: Duration = Duration::from_secs(1);

/// The session time of `q` quanta, SIM seconds: `float(2*q) * slot_s`.
pub(crate) fn session_t(q: u64) -> f64 {
    (2 * q) as f64 * SLOT_S
}

/// `q_end = 40 * duration_s`, or `None` when that is not a positive
/// integer (§3.2; the records reader refuses such a duration).
pub(crate) fn q_end(duration_s: f64) -> Option<u64> {
    let q = 40.0 * duration_s;
    (q > 0.0 && q.fract() == 0.0 && q <= (1u64 << 52) as f64).then_some(q as u64)
}

/// Where a session's records come from. There is no LIVE source (§7).
pub(crate) enum Source<'s> {
    /// Room Sim and Locate, streamed from a calibrated setup (SIM). The
    /// stream is boxed: it is large, and a session holds one source.
    /// `None` after a restart, until the next frame builds a new one, so
    /// a restart costs nothing and a flood of them cannot hold back the
    /// frames.
    Sim {
        setup: &'s RoomSimSetup,
        stream: Option<Box<RoomSimStream<'s>>>,
    },
    /// A recorded run (REPLAY): the next tick and epoch to play.
    Replay {
        recorded: Arc<Recorded>,
        tick: usize,
        epoch: usize,
    },
}

impl Source<'_> {
    /// `sim` or `replay`, never `live`: the mode is the source's.
    fn mode(&self) -> Mode {
        match self {
            Source::Sim { .. } => Mode::Sim,
            Source::Replay { .. } => Mode::Replay,
        }
    }

    /// The records with `T_prev < t <= t_end`, `T_prev` the last `t_end`.
    fn advance_to(&mut self, t_end: f64) -> Result<Advance, PipelineError> {
        match self {
            Source::Sim { setup, stream } => {
                let stream = match stream {
                    Some(stream) => stream,
                    None => stream.insert(Box::new(setup.stream()?)),
                };
                stream.advance_to(t_end)
            }
            Source::Replay {
                recorded,
                tick,
                epoch,
            } => {
                let r = &recorded.records;
                let (k0, e0) = (*tick, *epoch);
                while r.ticks.get(*tick).is_some_and(|k| k.t <= t_end) {
                    *tick += 1;
                }
                while r.epochs.get(*epoch).is_some_and(|e| e.t <= t_end) {
                    *epoch += 1;
                }
                Ok(Advance {
                    ticks: r.ticks[k0..*tick].to_vec(),
                    links: r.links[k0..*tick].to_vec(),
                    epochs: r.epochs[e0..*epoch].to_vec(),
                })
            }
        }
    }

    /// Back to `t = 0`: a new stream from the same setup (no new
    /// calibration), built by the next frame, or the first record of the
    /// file.
    fn restart(&mut self) {
        match self {
            Source::Sim { stream, .. } => *stream = None,
            Source::Replay { tick, epoch, .. } => {
                *tick = 0;
                *epoch = 0;
            }
        }
    }
}

/// What a session fixes at its start (§3.3: a client reconnects to
/// change any of it).
pub(crate) struct Session {
    /// For the reference point of the statistics; its people's and
    /// phone's ids cleared.
    scene: Scene,
    scene_block: Value,
    variant: Variant,
    seed: u64,
    duration_s: f64,
    has_phone: bool,
    tick_hz: f64,
    links: Vec<(i64, i64)>,
    /// The empty run's false alarms `(wlm, rti)` (§5.3); `None` for the
    /// `empty` variant.
    empty_run: Option<(FalseAlarms, FalseAlarms)>,
}

/// `scene` without any `Person.id` or `Phone.id` (nothing here reads them).
fn without_ids(mut scene: Scene) -> Scene {
    for p in &mut scene.people {
        p.id = None;
    }
    if let Some(phone) = &mut scene.phone {
        phone.id = None;
    }
    scene
}

impl Session {
    /// A SIM session of the embedded scene `e` (parsed as `scene`), run
    /// by `setup` as `variant` with `seed`.
    pub(crate) fn sim(
        e: Embedded,
        scene: Scene,
        setup: &RoomSimSetup,
        variant: Variant,
        seed: u64,
        empty_run: Option<(FalseAlarms, FalseAlarms)>,
    ) -> Session {
        let scene = without_ids(scene);
        Session {
            scene_block: scene_block(e, &scene),
            scene,
            variant,
            seed,
            duration_s: setup.duration_s(),
            has_phone: setup.has_phone(),
            tick_hz: setup.tick_hz(),
            links: setup.links().to_vec(),
            empty_run,
        }
    }

    /// A REPLAY session of a recorded run: variant, seed, duration and
    /// empty run from its manifest (§6.3). `None` if its embedded scene
    /// does not load, which cannot happen for a shipped scene.
    pub(crate) fn replay(recorded: &Recorded) -> Option<Session> {
        let scene = without_ids(recorded.scene.load()?);
        let m = &recorded.records.manifest;
        Some(Session {
            scene_block: scene_block(recorded.scene, &scene),
            scene,
            variant: m.variant,
            seed: m.seed,
            duration_s: m.duration_s,
            has_phone: m.has_phone,
            tick_hz: m.locate_config.tick_hz,
            links: m.links.clone(),
            empty_run: m.empty_run,
        })
    }

    pub(crate) fn scene_block(&self) -> &Value {
        &self.scene_block
    }

    pub(crate) fn run_block(&self, index: u64) -> Value {
        run_block(index, self.variant, self.seed, self.duration_s)
    }
}

/// The play state a run starts in. During `calibrating`, `play`, `pause`
/// and `speed` set it and `restart` does nothing (§3.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Start {
    pub(crate) paused: bool,
    pub(crate) speed: Speed,
}

impl Default for Start {
    /// A session starts playing, at 1x (§3.2).
    fn default() -> Self {
        Start {
            paused: false,
            speed: Speed::DEFAULT,
        }
    }
}

impl Start {
    pub(crate) fn apply(&mut self, cmd: Command) {
        match cmd {
            Command::Play => self.paused = false,
            Command::Pause => self.paused = true,
            Command::Speed(s) => self.speed = s,
            Command::Restart => {}
        }
    }
}

/// What the session thread does next.
#[derive(Debug)]
pub(crate) enum Next {
    /// Send this frame, then ask again.
    Frame(Value),
    /// Nothing is due before this time; wait for it or for a command.
    Wait(Duration),
    /// Send this `error` message, close with `code` and end the session.
    Close { message: Value, code: u16 },
}

/// One run of a session: from `q = 0` to `q_end`, or to a restart.
struct Run {
    index: u64,
    q: u64,
    /// The previous frame's `t` in this run; `-inf` before its first.
    t_prev: f64,
    /// Its first frame, at `q = 0`, has been sent.
    begun: bool,
    ended: bool,
    /// The run's records with `t <= T` so far, in time order.
    ticks: Vec<TickRecord>,
    links: Vec<Vec<LinkValue>>,
    epochs: Vec<EpochRecord>,
    /// `(now, t)` of the frames sent since the run began or the last
    /// resume, trimmed to the last [`RATE_WINDOW`] plus one.
    played: VecDeque<(Duration, f64)>,
}

impl Run {
    fn new(index: u64) -> Run {
        Run {
            index,
            q: 0,
            t_prev: f64::NEG_INFINITY,
            begun: false,
            ended: false,
            ticks: Vec::new(),
            links: Vec::new(),
            epochs: Vec::new(),
            played: VecDeque::new(),
        }
    }
}

/// A session: its source, what it fixes, its clock and its current run.
pub(crate) struct Engine<'s> {
    source: Source<'s>,
    session: Session,
    q_end: u64,
    /// The next frame's `seq`: per connection, from 0, +1 per frame (§4).
    seq: u64,
    speed: Speed,
    paused: bool,
    /// When the next frame is due.
    due: Duration,
    /// When the last frame was computed.
    sent: Duration,
    /// Since when the session has been paused or ended (§2.4).
    idle_since: Option<Duration>,
    run: Run,
}

impl<'s> Engine<'s> {
    /// A session whose first run starts in `start`'s state; its first
    /// frame is due at `now`. Refused when `40 * duration_s` is not a
    /// positive integer.
    pub(crate) fn new(
        source: Source<'s>,
        session: Session,
        start: Start,
        now: Duration,
    ) -> Result<Self, &'static str> {
        let q_end = q_end(session.duration_s)
            .ok_or("duration_s: 40 * duration_s is not a positive integer")?;
        Ok(Engine {
            source,
            session,
            q_end,
            seq: 0,
            speed: start.speed,
            paused: start.paused,
            due: now,
            sent: now,
            idle_since: start.paused.then_some(now),
            run: Run::new(0),
        })
    }

    pub(crate) fn mode(&self) -> Mode {
        self.source.mode()
    }

    /// The current run's session time: `float(2*q) * slot_s`, and exactly
    /// `duration_s` at `q_end` (§3.2).
    fn t(&self) -> f64 {
        if self.run.q == self.q_end {
            self.session.duration_s
        } else {
            session_t(self.run.q)
        }
    }

    /// Apply one command (§3.3), in arrival order, before the next frame.
    pub(crate) fn command(&mut self, cmd: Command, now: Duration) {
        match cmd {
            Command::Play => {
                if self.paused {
                    self.paused = false;
                    if !self.run.ended {
                        self.idle_since = None;
                        if self.run.begun {
                            // effective_speed starts again from here.
                            self.run.played.clear();
                            self.run.played.push_back((now, self.t()));
                            self.due = now + FRAME_PERIOD;
                        }
                    }
                }
            }
            Command::Pause => {
                if !self.paused {
                    // The next deadline sends the frame that says so.
                    self.paused = true;
                    self.idle_since.get_or_insert(now);
                }
            }
            Command::Speed(s) => self.speed = s,
            // No frame sent yet: the session is still `calibrating` (or
            // `ready`) to the client, and its first run is at `q = 0`;
            // a restart does nothing (§3.3).
            Command::Restart if self.seq == 0 => {}
            Command::Restart => {
                self.run = Run::new(self.run.index + 1);
                self.source.restart();
                self.idle_since = self.paused.then_some(now);
                // The new run's first frame goes at the next 100 ms
                // deadline (at once if that has passed), not sooner: a
                // flood of restarts still costs one frame a deadline.
                self.due = self.due.min(self.sent + FRAME_PERIOD);
            }
        }
    }

    /// What is next at `now`: a frame that is due, a time to wait for, or
    /// the end of the session.
    pub(crate) fn next(&mut self, now: Duration) -> Next {
        let idle_end = self.idle_since.map(|since| since + IDLE_LIMIT);
        if let Some(end) = idle_end {
            if now >= end {
                return Next::Close {
                    message: error_message(self.mode(), "session", "idle for 10 minutes"),
                    code: 1000,
                };
            }
            if self.run.ended {
                return Next::Wait(end);
            }
        }
        if now < self.due {
            return Next::Wait(idle_end.map_or(self.due, |end| end.min(self.due)));
        }
        match self.frame(now) {
            Ok(frame) => Next::Frame(frame),
            Err(e) => Next::Close {
                message: error_message(self.mode(), e.param().unwrap_or("session"), &e.to_string()),
                code: 1011,
            },
        }
    }

    /// The frame due at `now` (§3.2): the run's first at `q = 0`, a step,
    /// or, paused, the same `t` again.
    fn frame(&mut self, now: Duration) -> Result<Value, PipelineError> {
        let due = self.due;
        let step = self.run.begun && !self.paused;
        if step {
            self.run.q = (self.run.q + self.speed.quanta()).min(self.q_end);
        }
        let first = !self.run.begun;
        self.run.begun = true;
        let t = self.t();
        let (new_ticks_from, new_epochs_from) = (self.run.ticks.len(), self.run.epochs.len());
        // Paused, `t` is the last one advanced to, and nothing is new.
        let advance = self.source.advance_to(t)?;
        self.run.ticks.extend(advance.ticks);
        self.run.links.extend(advance.links);
        self.run.epochs.extend(advance.epochs);
        if step && self.run.q == self.q_end {
            self.run.ended = true;
            self.idle_since.get_or_insert(now);
        }
        let effective_speed = if self.paused {
            None
        } else {
            if first {
                self.run.played.clear();
            }
            self.effective_speed(now, t)
        };
        let mode = self.mode();
        let frame = Frame {
            seq: self.seq,
            run: self.session.run_block(self.run.index),
            t,
            mode,
            scene: &self.session.scene_block,
            speed: self.speed.value(),
            effective_speed,
            paused: self.paused,
            ended: self.run.ended,
            links: &self.session.links,
            has_phone: self.session.has_phone,
            ticks: &self.run.ticks,
            tick_links: &self.run.links,
            epochs: &self.run.epochs,
            new_ticks_from,
            new_epochs_from,
            stats: self.stats(mode),
        }
        .to_json();
        debug_assert!(self.run.ticks[new_ticks_from..]
            .iter()
            .all(|k| self.run.t_prev < k.t && k.t <= t));
        self.run.t_prev = t;
        self.seq += 1;
        self.sent = now;
        let period = if self.paused {
            HEARTBEAT_PERIOD
        } else {
            FRAME_PERIOD
        };
        self.due = due + period;
        while self.due <= now {
            self.due += period;
        }
        Ok(frame)
    }

    /// Simulated seconds per wall second over the frames sent in the last
    /// wall second; `None` until a wall second has passed in this run or
    /// since the last resume (§3.2).
    fn effective_speed(&mut self, now: Duration, t: f64) -> Option<f64> {
        let played = &mut self.run.played;
        played.push_back((now, t));
        let cutoff = now.checked_sub(RATE_WINDOW)?;
        while played.len() >= 2 && played[1].0 <= cutoff {
            played.pop_front();
        }
        let &(w0, t0) = played.front()?;
        (w0 <= cutoff).then(|| (t - t0) / (now - w0).as_secs_f64())
    }

    /// §5's block over the run's records with `t <= T`, by the shared score
    /// function on every frame; REPLAY's without its error figures
    /// (§5.4). `empty_run` is the session's (§5.3) or the manifest's.
    fn stats(&self, mode: Mode) -> Value {
        let s = &self.session;
        let scores = score(
            &s.scene,
            &self.run.ticks,
            &self.run.epochs,
            s.has_phone,
            s.tick_hz,
        );
        let mut v = if shows_truth(mode) {
            scores.to_json()
        } else {
            scores.to_json_truthless()
        };
        if let Some((wlm, rti)) = &s.empty_run {
            for (key, fa) in [("wlm", wlm), ("rti", rti)] {
                if let Some(block) = v.get_mut(key).and_then(Value::as_object_mut) {
                    block.insert(
                        "empty_run".into(),
                        serde_json::to_value(fa).unwrap_or(Value::Null),
                    );
                }
            }
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roomstream::frame::state_message;
    use crate::roomstream::testkit::*;
    use saans_locate::pipeline::run_roomsim;
    use saans_locate::score::{false_alarms, PersonEstimator};
    use serde_json::json;

    #[test]
    fn the_clock_is_room_sims_and_ends_exactly() {
        for q in [0u64, 1, 4, 37, 800, 4799] {
            assert_eq!(session_t(q), (2 * q) as f64 * 0.0125);
        }
        assert_eq!(q_end(120.0), Some(4800));
        assert_eq!(
            session_t(4800),
            120.0,
            "T(q_end) is duration_s for the SIM run length"
        );
        assert_eq!(q_end(20.5), Some(820));
        for bad in [0.0, -1.0, 0.01, f64::NAN, f64::INFINITY, 1e300] {
            assert_eq!(q_end(bad), None, "{bad}");
        }
    }

    /// Steps of `4 * speed` quanta from `q = 0`, the clamp at `q_end`,
    /// then nothing: on a 20.5 s run at 8x the last step is 20, not 32.
    #[test]
    fn a_run_steps_by_its_speed_and_clamps_at_its_end() {
        let fx = sim_fixture("bedroom", Variant::Declared, 20.5);
        let mut e = sim_engine(
            &fx,
            Start {
                paused: false,
                speed: Speed::ALL[5],
            },
        );
        let frames = play_to_end(&mut e, Duration::ZERO).0;
        let qs: Vec<u64> = frames
            .iter()
            .map(|f| q_of(f["t"].as_f64().unwrap()))
            .collect();
        let mut want: Vec<u64> = (0..=25).map(|k| 32 * k).collect();
        want.push(820);
        assert_eq!(qs, want);
        for f in &frames {
            assert_eq!(
                f["t"].as_f64().unwrap(),
                session_t(q_of(f["t"].as_f64().unwrap()))
            );
        }
        assert_eq!(frames.last().unwrap()["t"], 20.5);
        assert!(frames.last().unwrap()["ended"].as_bool().unwrap());
        assert!(frames[..frames.len() - 1]
            .iter()
            .all(|f| f["ended"] == false));
        let seqs: Vec<u64> = frames.iter().map(|f| f["seq"].as_u64().unwrap()).collect();
        assert_eq!(seqs, (0..frames.len() as u64).collect::<Vec<_>>());

        // Every speed: q advances by exactly 4 * speed per frame.
        for s in Speed::ALL {
            let mut e = replay_engine(Start {
                paused: false,
                speed: s,
            });
            let mut now = Duration::ZERO;
            let ts: Vec<f64> = (0..6)
                .map(|_| next_frame(&mut e, &mut now)["t"].as_f64().unwrap())
                .collect();
            for (k, t) in ts.iter().enumerate() {
                assert_eq!(*t, session_t(k as u64 * s.quanta()), "{s:?}");
            }
        }
    }

    #[test]
    fn deadlines_are_100_ms_apart_and_missed_ones_are_skipped() {
        let mut e = replay_engine(Start::default());
        assert!(matches!(e.next(Duration::ZERO), Next::Frame(_)));
        assert!(matches!(e.next(Duration::ZERO), Next::Wait(d) if d == FRAME_PERIOD));
        assert!(matches!(e.next(ms(99)), Next::Wait(d) if d == FRAME_PERIOD));
        assert!(matches!(e.next(ms(100)), Next::Frame(_)));
        // 350 ms late: one frame, not four, and the next deadline is the
        // first one still ahead.
        let Next::Frame(f) = e.next(ms(550)) else {
            panic!("due")
        };
        assert_eq!(f["t"], session_t(8));
        assert!(matches!(e.next(ms(550)), Next::Wait(d) if d == ms(600)));
        let Next::Frame(f) = e.next(ms(600)) else {
            panic!("due")
        };
        assert_eq!(f["t"], session_t(12), "the clock does not catch up");
    }

    #[test]
    fn pause_sends_one_frame_then_a_heartbeat_a_second() {
        let mut e = replay_engine(Start::default());
        let mut now = Duration::ZERO;
        for _ in 0..3 {
            next_frame(&mut e, &mut now);
        }
        let before = next_frame(&mut e, &mut now);
        e.command(Command::Pause, now);
        let paused = next_frame(&mut e, &mut now);
        assert_eq!(
            now,
            ms(400),
            "the pause frame goes at the next 100 ms deadline"
        );
        assert_eq!(paused["paused"], true);
        assert_eq!(paused["t"], before["t"]);
        assert_eq!(paused["seq"], before["seq"].as_u64().unwrap() + 1);
        assert_eq!(paused["effective_speed"], Value::Null);
        let beat = next_frame(&mut e, &mut now);
        assert_eq!(now, ms(400) + HEARTBEAT_PERIOD, "then one a second");
        assert_eq!(beat["t"], before["t"]);
        assert_eq!(beat["seq"], paused["seq"].as_u64().unwrap() + 1);
        assert_eq!(beat["new"], json!({ "persons": [], "phone": [] }));
        assert_eq!(beat["persons"], paused["persons"]);
        assert_eq!(beat["stats"], paused["stats"]);
        e.command(Command::Play, now);
        let resumed = next_frame(&mut e, &mut now);
        assert_eq!(resumed["paused"], false);
        assert_eq!(
            resumed["t"],
            session_t(q_of(before["t"].as_f64().unwrap()) + 4)
        );
        assert_eq!(
            resumed["effective_speed"],
            Value::Null,
            "null until a wall second since the resume"
        );
    }

    #[test]
    fn effective_speed_is_null_for_a_second_then_measured() {
        let mut e = replay_engine(Start {
            paused: false,
            speed: Speed::ALL[4],
        });
        let mut now = Duration::ZERO;
        let frames: Vec<Value> = (0..15).map(|_| next_frame(&mut e, &mut now)).collect();
        for f in &frames[..10] {
            assert_eq!(f["effective_speed"], Value::Null, "{}", f["seq"]);
        }
        for f in &frames[10..] {
            let eff = f["effective_speed"].as_f64().unwrap();
            assert!((eff - 4.0).abs() < 1e-9, "{eff}");
        }
        // A slow host: frames 200 ms apart give half the speed.
        let mut e = replay_engine(Start::default());
        let mut last = Value::Null;
        for k in 0..12 {
            let Next::Frame(f) = e.next(ms(200 * k)) else {
                panic!("due at {k}")
            };
            last = f;
        }
        let eff = last["effective_speed"].as_f64().unwrap();
        assert!((eff - 0.5).abs() < 1e-9, "{eff}");
    }

    #[test]
    fn restart_starts_a_new_run_at_zero_and_keeps_the_play_state() {
        let mut e = replay_engine(Start::default());
        let mut now = Duration::ZERO;
        for _ in 0..30 {
            next_frame(&mut e, &mut now);
        }
        e.command(Command::Speed(Speed::ALL[3]), now);
        e.command(Command::Restart, now);
        let f = next_frame(&mut e, &mut now);
        assert_eq!(f["run"]["index"], 1);
        assert_eq!(f["t"], 0.0);
        assert_eq!(f["seq"], 30, "seq keeps counting");
        assert_eq!(f["speed"], 2.0);
        assert_eq!(f["state"], "warming_up");
        assert_eq!(f["persons"], Value::Null);
        assert_eq!(
            f["stats"]["wlm"]["ticks"], 0,
            "the statistics are the new run's"
        );
        let g = next_frame(&mut e, &mut now);
        assert_eq!(g["t"], session_t(8));

        // Paused, a restart sends the new run's first frame, paused, at 0.
        e.command(Command::Pause, now);
        next_frame(&mut e, &mut now);
        e.command(Command::Restart, now);
        let f = next_frame(&mut e, &mut now);
        assert_eq!(
            (
                f["run"]["index"].as_u64(),
                f["t"].as_f64(),
                f["paused"].as_bool()
            ),
            (Some(2), Some(0.0), Some(true))
        );
        let beat = next_frame(&mut e, &mut now);
        assert_eq!(beat["t"], 0.0);
    }

    /// §3.3: until the session's first frame (to the client it is still
    /// `calibrating`) a restart does nothing; after it, each one starts a
    /// run.
    #[test]
    fn a_restart_before_the_first_frame_does_nothing() {
        let mut e = replay_engine(Start::default());
        e.command(Command::Restart, Duration::ZERO);
        e.command(Command::Restart, Duration::ZERO);
        let mut now = Duration::ZERO;
        let f = next_frame(&mut e, &mut now);
        assert_eq!(
            (
                f["seq"].as_u64(),
                f["run"]["index"].as_u64(),
                f["t"].as_f64()
            ),
            (Some(0), Some(0), Some(0.0))
        );
        next_frame(&mut e, &mut now);
        e.command(Command::Restart, now);
        e.command(Command::Restart, now);
        let f = next_frame(&mut e, &mut now);
        assert_eq!(
            (f["run"]["index"].as_u64(), f["t"].as_f64()),
            (Some(2), Some(0.0))
        );
    }

    /// A restart's first frame goes at the next 100 ms deadline: never two
    /// frames within 100 ms, however many restarts come; and paused, 100
    /// ms after the last frame rather than at the next heartbeat.
    #[test]
    fn a_restart_keeps_the_100_ms_cadence() {
        let mut e = replay_engine(Start::default());
        let mut now = Duration::ZERO;
        for _ in 0..5 {
            next_frame(&mut e, &mut now);
        }
        assert_eq!(now, ms(400));
        for _ in 0..100 {
            e.command(Command::Restart, ms(410));
        }
        assert!(matches!(e.next(ms(450)), Next::Wait(d) if d == ms(500)));
        let Next::Frame(f) = e.next(ms(500)) else {
            panic!("due")
        };
        assert_eq!(
            (f["run"]["index"].as_u64(), f["t"].as_f64()),
            (Some(100), Some(0.0))
        );
        e.command(Command::Pause, ms(500));
        let Next::Frame(p) = e.next(ms(600)) else {
            panic!("due")
        };
        assert_eq!(p["paused"], true);
        e.command(Command::Restart, ms(650));
        assert!(matches!(e.next(ms(650)), Next::Wait(d) if d == ms(700)));
        let Next::Frame(f) = e.next(ms(700)) else {
            panic!("due")
        };
        assert_eq!(
            (
                f["run"]["index"].as_u64(),
                f["t"].as_f64(),
                f["paused"].as_bool()
            ),
            (Some(101), Some(0.0), Some(true))
        );
    }

    /// A SIM restart builds nothing: the next frame builds the new stream
    /// from the setup, so a flood of restarts costs one stream a frame,
    /// and the new run is the same run again.
    #[test]
    fn a_sim_restart_leaves_the_new_stream_to_the_next_frame() {
        let fx = sim_fixture("bedroom", Variant::WithWalker, 20.0);
        let mut e = sim_engine(&fx, Start::default());
        let mut now = Duration::ZERO;
        let first: Vec<Value> = (0..15).map(|_| next_frame(&mut e, &mut now)).collect();
        for _ in 0..1000 {
            e.command(Command::Restart, now);
        }
        assert!(matches!(e.source, Source::Sim { stream: None, .. }));
        let again: Vec<Value> = (0..15).map(|_| next_frame(&mut e, &mut now)).collect();
        assert!(matches!(
            e.source,
            Source::Sim {
                stream: Some(_),
                ..
            }
        ));
        assert_eq!(again[0]["run"]["index"], 1000);
        assert_eq!(again[0]["seq"], 15);
        for (a, b) in first.iter().zip(&again) {
            for key in [
                "t", "state", "persons", "links", "phone", "new", "truth", "stats",
            ] {
                assert_eq!(a[key], b[key], "{key} at t = {}", a["t"]);
            }
        }
        assert!(
            first.last().unwrap()["persons"].is_object(),
            "not vacuous: past warming_up"
        );
    }

    #[test]
    fn an_ended_run_sends_nothing_until_a_restart_and_idles_out() {
        let mut e = replay_engine(Start {
            paused: false,
            speed: Speed::ALL[5],
        });
        let (frames, now) = play_to_end(&mut e, Duration::ZERO);
        let last = frames.last().unwrap();
        assert_eq!(last["ended"], true);
        assert_eq!(last["t"], last["run"]["duration_s"]);
        // Nothing for 10 minutes, then the idle close.
        assert!(matches!(e.next(now + ms(100)), Next::Wait(d) if d == now + IDLE_LIMIT));
        e.command(Command::Play, now + ms(200));
        e.command(Command::Speed(Speed::DEFAULT), now + ms(300));
        assert!(
            matches!(e.next(now + ms(400)), Next::Wait(_)),
            "play does not restart an ended run"
        );
        match e.next(now + IDLE_LIMIT) {
            Next::Close { message, code } => {
                assert_eq!(code, 1000);
                assert_eq!(message["type"], "error");
                assert_eq!(message["param"], "session");
                assert_eq!(message["message"], "idle for 10 minutes");
                assert_eq!(message["mode"], "replay");
                assert_eq!(message["label"], "SIM");
            }
            other => panic!("{other:?}"),
        }
        // A restart before the limit starts again.
        let mut e = replay_engine(Start {
            paused: false,
            speed: Speed::ALL[5],
        });
        let (_, now) = play_to_end(&mut e, Duration::ZERO);
        e.command(Command::Restart, now + ms(500));
        let Next::Frame(f) = e.next(now + ms(500)) else {
            panic!("due")
        };
        assert_eq!(
            (f["t"].as_f64(), f["ended"].as_bool()),
            (Some(0.0), Some(false))
        );
        assert!(
            matches!(e.next(now + IDLE_LIMIT + ms(500)), Next::Frame(_)),
            "playing is not idle"
        );
    }

    #[test]
    fn a_paused_session_idles_out_after_10_minutes() {
        let mut e = replay_engine(Start {
            paused: true,
            speed: Speed::DEFAULT,
        });
        let mut now = Duration::ZERO;
        let f = next_frame(&mut e, &mut now);
        assert_eq!(
            (f["t"].as_f64(), f["paused"].as_bool()),
            (Some(0.0), Some(true)),
            "a run can start paused"
        );
        let mut beats = 0;
        loop {
            match e.next(now) {
                Next::Frame(_) => beats += 1,
                Next::Wait(d) => now = d,
                Next::Close { code, .. } => {
                    assert_eq!(code, 1000);
                    break;
                }
            }
        }
        assert_eq!(now, IDLE_LIMIT);
        assert_eq!(beats, 599, "a heartbeat a second");
    }

    /// Every record is in exactly one frame's `new`, the frame with
    /// `T_prev < t <= T`, `T_prev = -inf` on a run's first frame, which so
    /// carries the epoch at `t = 0`; and the latest blocks are the
    /// latest records with `t <= T`.
    #[test]
    fn new_holds_exactly_the_records_since_the_previous_frame() {
        let fx = sim_fixture("hall-l-shaped", Variant::WithWalker, 20.0);
        let all = run_roomsim(&fx.scene, &fx.rs, &Default::default(), "SIM").unwrap();
        assert!(
            all.epochs.first().is_some_and(|e| e.t == 0.0),
            "the hall's phone fires at t = 0"
        );
        for speed in [Speed::ALL[0], Speed::ALL[2], Speed::ALL[5]] {
            let mut e = sim_engine(
                &fx,
                Start {
                    paused: false,
                    speed,
                },
            );
            let mut now = Duration::ZERO;
            // A partial run, a restart, then a whole one.
            let mut runs: Vec<Vec<Value>> =
                vec![(0..7).map(|_| next_frame(&mut e, &mut now)).collect()];
            e.command(Command::Restart, now);
            runs.push(play_to_end(&mut e, now).0);
            for frames in &runs {
                let mut t_prev = f64::NEG_INFINITY;
                let (mut ticks, mut epochs) = (Vec::new(), Vec::new());
                for f in frames {
                    let t = f["t"].as_f64().unwrap();
                    for k in f["new"]["persons"].as_array().unwrap() {
                        let tk = k["t"].as_f64().unwrap();
                        assert!(t_prev < tk && tk <= t, "{t_prev} < {tk} <= {t}");
                        ticks.push(tk);
                    }
                    for p in f["new"]["phone"].as_array().unwrap() {
                        let te = p["t"].as_f64().unwrap();
                        assert!(t_prev < te && te <= t, "{t_prev} < {te} <= {t}");
                        epochs.push(te);
                    }
                    let latest = all.ticks.iter().rev().find(|k| k.t <= t);
                    assert_eq!(f["persons"]["t"].as_f64(), latest.map(|k| k.t));
                    assert_eq!(f["links"]["t"].as_f64(), latest.map(|k| k.t));
                    assert_eq!(
                        f["state"],
                        if latest.is_some() {
                            "running"
                        } else {
                            "warming_up"
                        }
                    );
                    let phone = all.epochs.iter().rev().find(|e| e.t <= t);
                    assert_eq!(f["phone"]["t"].as_f64(), phone.map(|e| e.t));
                    t_prev = t;
                }
                assert!(
                    frames[0]["new"]["phone"][0]["t"] == 0.0,
                    "the first frame carries the epoch at t = 0"
                );
                let n = ticks.len();
                assert_eq!(
                    ticks,
                    all.ticks[..n].iter().map(|k| k.t).collect::<Vec<_>>()
                );
                assert_eq!(
                    epochs,
                    all.epochs[..epochs.len()]
                        .iter()
                        .map(|e| e.t)
                        .collect::<Vec<_>>()
                );
                if frames.last().unwrap()["ended"] == true {
                    assert_eq!(n, all.ticks.len(), "a whole run sends every tick");
                    assert_eq!(epochs.len(), all.epochs.len(), "and every epoch");
                }
            }
        }
    }

    /// spec/livedemo.md §11: for the bedroom at 120 s, the final frame's
    /// `stats` equal `score(run_roomsim(…))` plus the `empty` variant's
    /// false alarms (§5.3, §10 "Equality with the session").
    #[test]
    fn the_final_stats_of_a_120_s_bedroom_session_equal_score_of_run_roomsim() {
        let fx = sim_fixture("bedroom", Variant::WithWalker, 120.0);
        let mut e = sim_engine(
            &fx,
            Start {
                paused: false,
                speed: Speed::ALL[5],
            },
        );
        let (frames, _) = play_to_end(&mut e, Duration::ZERO);
        let last = frames.last().unwrap();
        assert_eq!(
            (last["t"].as_f64(), last["ended"].as_bool()),
            (Some(120.0), Some(true))
        );

        let cfg = Default::default();
        let run = run_roomsim(&fx.scene, &fx.rs, &cfg, "SIM").unwrap();
        let empty = run_roomsim(
            &fx.scene,
            &Variant::Empty.apply(&fx.scene, &fx.base),
            &cfg,
            "SIM",
        )
        .unwrap();
        let tick_hz = fx.setup.tick_hz();
        let mut want = score(&fx.scene, &run.ticks, &run.epochs, run.has_phone, tick_hz).to_json();
        for which in PersonEstimator::ALL {
            want[which.as_str()]["empty_run"] =
                serde_json::to_value(false_alarms(&empty.ticks, which, tick_hz)).unwrap();
        }
        assert_eq!(last["stats"], want);
        assert!(
            last["stats"]["wlm"]["occupied"]["published"]
                .as_u64()
                .unwrap()
                > 0,
            "not vacuous"
        );
        assert_eq!(
            fx.empty_run,
            Some((
                false_alarms(&empty.ticks, PersonEstimator::Wlm, tick_hz),
                false_alarms(&empty.ticks, PersonEstimator::Rti, tick_hz),
            ))
        );
    }

    /// §5.4 and §11: a REPLAY engine fed a file written by saans-locate's
    /// writer gives the same statistics with the error figures deleted,
    /// the manifest's empty run, and no truth anywhere.
    #[test]
    fn a_replay_of_a_written_file_gives_the_same_truthless_stats() {
        let fx = sim_fixture("bedroom", Variant::WithWalker, 20.0);
        let mut sim = sim_engine(
            &fx,
            Start {
                paused: false,
                speed: Speed::ALL[5],
            },
        );
        let sim_last = play_to_end(&mut sim, Duration::ZERO).0.pop().unwrap();

        let recorded = recorded_from_file(&fx);
        let mut rep = Engine::new(
            Source::Replay {
                recorded: Arc::clone(&recorded),
                tick: 0,
                epoch: 0,
            },
            Session::replay(&recorded).unwrap(),
            Start {
                paused: false,
                speed: Speed::ALL[5],
            },
            Duration::ZERO,
        )
        .unwrap();
        assert_eq!(rep.mode(), Mode::Replay);
        let frames = play_to_end(&mut rep, Duration::ZERO).0;
        let rep_last = frames.last().unwrap();

        let tick_hz = fx.setup.tick_hz();
        let run = run_roomsim(&fx.scene, &fx.rs, &Default::default(), "SIM").unwrap();
        let mut want = score(&fx.scene, &run.ticks, &run.epochs, true, tick_hz).to_json_truthless();
        let (wlm, rti) = fx.empty_run.unwrap();
        want["wlm"]["empty_run"] = serde_json::to_value(wlm).unwrap();
        want["rti"]["empty_run"] = serde_json::to_value(rti).unwrap();
        assert_eq!(rep_last["stats"], want);
        for key in ["median_m", "p90_m", "inside_radius", "ref"] {
            assert!(
                sim_last["stats"]["wlm"]["occupied"].get(key).is_some(),
                "SIM keeps {key}"
            );
            assert!(
                rep_last["stats"]["wlm"]["occupied"].get(key).is_none(),
                "REPLAY deletes {key}"
            );
            assert!(
                rep_last["stats"]["rtt"].get(key).is_none(),
                "REPLAY deletes the phone's {key}"
            );
        }
        // Same records, same frames, but for mode, truth and the error
        // figures.
        assert_eq!(rep_last["persons"], sim_last["persons"]);
        assert_eq!(rep_last["links"], sim_last["links"]);
        assert_eq!(rep_last["phone"], sim_last["phone"]);
        assert_eq!(rep_last["run"], sim_last["run"]);
        assert_eq!(rep_last["scene"], sim_last["scene"]);
        for f in &frames {
            assert_eq!(f["mode"], "replay");
            assert!(f.get("truth").is_none());
            assert_no_truth_outside_truth(f);
            assert_eq!(
                key_paths(f, "blocked"),
                Vec::<String>::new(),
                "no blocked in REPLAY"
            );
        }
    }

    /// §4: SIM frames carry truth; `blocked` lives only inside
    /// `truth.phone`; every position outside truth has its radius.
    #[test]
    fn sim_frames_carry_truth_and_no_bare_position() {
        let fx = sim_fixture("hall-l-shaped", Variant::WithWalker, 20.0);
        let mut e = sim_engine(
            &fx,
            Start {
                paused: false,
                speed: Speed::ALL[5],
            },
        );
        let frames = play_to_end(&mut e, Duration::ZERO).0;
        let mut scored = 0;
        for f in &frames {
            assert_eq!(f["mode"], "sim");
            assert_eq!(f["label"], "SIM");
            let truth = f.get("truth").expect("SIM frames carry truth");
            assert_no_truth_outside_truth(f);
            for path in key_paths(f, "blocked") {
                assert_eq!(path, "$.truth.phone.blocked");
            }
            if !truth["persons"].is_null() {
                let people = truth["persons"]["people"].as_array().unwrap();
                assert_eq!(people.len(), 3, "the hall's two people and the walker");
                for score in [&truth["persons"]["wlm"], &truth["persons"]["rti"]] {
                    let keys: Vec<&str> = score
                        .as_object()
                        .unwrap()
                        .keys()
                        .map(String::as_str)
                        .collect();
                    assert!(
                        keys == ["error_m", "person"] || keys == ["unscored"],
                        "{score}"
                    );
                    scored += usize::from(keys.len() == 2);
                }
                assert_eq!(
                    truth["persons"]["t"], f["persons"]["t"],
                    "truth is the record's own"
                );
            }
            if !truth["phone"].is_null() {
                assert_eq!(truth["phone"]["t"], f["phone"]["t"]);
                for score in [&truth["phone"]["rtt"], &truth["phone"]["rssi"]] {
                    let keys: Vec<&str> = score
                        .as_object()
                        .unwrap()
                        .keys()
                        .map(String::as_str)
                        .collect();
                    assert!(keys == ["error_m"] || keys == ["unscored"], "{score}");
                }
            }
        }
        assert!(scored > 0, "some estimate was scored");
    }

    /// ADR-008 decision 3: for every shipped scene and variant, no
    /// `Person.id` or `Phone.id` is a string value in any state, frame or
    /// error. A walk of the parsed JSON, not a text search: the shipped
    /// phone's id "phone" is also a frame key.
    #[test]
    fn no_person_or_phone_id_is_sent_for_any_scene_or_variant() {
        let mut checked = 0;
        for e in Embedded::all() {
            let ids = scene_ids(e);
            assert!(
                ids.iter().any(|i| i == "phone") && ids.len() >= 2,
                "{}: not vacuous",
                e.id
            );
            for variant in Variant::ALL {
                let fx = sim_fixture(e.id, variant, 20.0);
                let mut eng = sim_engine(
                    &fx,
                    Start {
                        paused: false,
                        speed: Speed::ALL[5],
                    },
                );
                let mut messages = vec![
                    state_message(
                        "calibrating",
                        Mode::Sim,
                        eng.session.scene_block(),
                        &eng.session.run_block(0),
                    ),
                    error_message(
                        Mode::Sim,
                        "session",
                        "4 room simulations are running; try again later",
                    ),
                ];
                let (frames, now) = play_to_end(&mut eng, Duration::ZERO);
                messages.extend(frames);
                eng.command(Command::Pause, now);
                eng.command(Command::Restart, now);
                let mut t = now;
                messages.push(next_frame(&mut eng, &mut t));
                messages.push(next_frame(&mut eng, &mut t));
                if let Next::Close { message, .. } = eng.next(t + IDLE_LIMIT) {
                    messages.push(message);
                }
                for m in &messages {
                    assert_no_string_value(m, &ids, e.id);
                    checked += 1;
                }
            }
        }
        assert!(checked > 9 * 25);
    }

    #[test]
    fn a_running_session_engine_never_reports_live() {
        let mut e = replay_engine(Start::default());
        assert_eq!(e.mode(), Mode::Replay);
        let mut now = Duration::ZERO;
        assert_eq!(next_frame(&mut e, &mut now)["mode"], "replay");
        let fx = sim_fixture("office", Variant::Empty, 20.0);
        let mut e = sim_engine(&fx, Start::default());
        let mut now = Duration::ZERO;
        assert_eq!(e.mode(), Mode::Sim);
        let f = next_frame(&mut e, &mut now);
        assert_eq!(f["mode"], "sim");
        // Variant `empty`: no empty run is kept, and the key is absent.
        assert!(f["stats"]["wlm"].get("empty_run").is_none());
        assert!(f["stats"]["wlm"].get("unoccupied").is_some());
    }

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }
}
