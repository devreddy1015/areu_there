//! The room stream's handshake, connection task and session thread
//! (spec/livedemo.md §2, §3.0).
//!
//! **Handshake**, in this order: Origin and Host (403, not audited), the
//! principal as for `/api/live` (401, audited by `principal()`),
//! `can_see_position(None)` (403, audited `Denied`), then a session slot.
//! A query error, or no free slot, still completes the upgrade, so the
//! page can be told why: an `error` message, then close 1008 or 1013.
//!
//! **Connection task** (on the runtime): parses commands, forwards them
//! over a `std::sync::mpsc` channel, and writes the thread's
//! pre-serialised messages to the socket. A client that accepts nothing
//! for 10 s (a write that waits that long, or a ping it leaves unanswered
//! that long: see [`Liveness`]) is closed with 1008.
//!
//! **Session thread** (`roomsim-session`, a plain OS thread: never a
//! runtime worker, which carry UDP ingest and the tick, and never
//! `spawn_blocking`, which shutdown would wait for): it owns the setup,
//! the stream and the clock, and sends every message over a tokio channel
//! of capacity 1 with `blocking_send`. It computes a frame only once the
//! previous message was taken, so no frame is dropped and the commands
//! that arrive meanwhile apply to the next one. It exits within 0.5 s of
//! the socket closing (the channels disconnect) or the gateway stopping
//! (the cancel flag), calibration included, and releases its slot as it
//! exits.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{
    sync_channel, Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError,
};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use axum::extract::ws::rejection::WebSocketUpgradeRejection;
use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::{RawQuery, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use saans_locate::phone::Config;
use saans_locate::pipeline::PipelineError;
use saans_locate::score::{false_alarms, FalseAlarms, PersonEstimator};
use saans_locate::stream::RoomSimSetup;
use saans_locate::variant::Variant;
use saans_roomsim::Config as RoomSimConfig;
use serde_json::{json, Value};
use tokio::sync::{mpsc, watch};
use tokio::time::{timeout, MissedTickBehavior};

use super::engine::{Engine, Next, Session, Source, Start};
use super::frame::{error_message, run_block, scene_block, state_message, LABEL};
use super::params::{parse_command, parse_query, requested_mode, resolve, Command, Job, Refusal};
use super::replay::Recorded;
use super::{Embedded, Slot, ROOMSIM_MAX_SESSIONS, ROOMSIM_RUN_S};
use crate::audit::Action;
use crate::http::{error, principal, AppState};
use crate::shared::Mode;
use crate::timefmt::now_unix_ms;

const PATH: &str = "/api/roomsim";
/// The thread's name, as `ps -L` and a debugger show it.
pub(crate) const THREAD_NAME: &str = "roomsim-session";
/// The largest message a client may send (§2.4); commands are tiny.
const MAX_MESSAGE: usize = 1024;
/// A client that accepts nothing for this long is closed with 1008
/// (§3.2): a write that waits this long, or a ping unanswered this long.
const ACCEPT_TIMEOUT: Duration = Duration::from_secs(10);
/// At most one ping this often, and only after a message was sent.
const PING_EVERY: Duration = Duration::from_secs(1);
/// How long a close waits to be sent and answered.
const CLOSE_WAIT: Duration = Duration::from_secs(1);
/// Commands queued for the thread; more at once are refused, not queued.
const COMMAND_QUEUE: usize = 64;
/// The longest the thread waits without looking at its cancel flag.
const POLL: Duration = Duration::from_millis(100);
/// The empty run is advanced this many SIM seconds at a time, so a stop
/// is noticed between steps (the records do not depend on the steps).
const EMPTY_RUN_STEP_S: f64 = 1.0;

const BUSY: &str = "4 room simulations are running; try again later";
const _: () = assert!(ROOMSIM_MAX_SESSIONS == 4, "BUSY names the cap");

// ===================================================================
// Origin and Host (§2.2)
// ===================================================================

/// The one value of a header given exactly once, as text.
fn single(h: &HeaderMap, name: header::HeaderName) -> Option<&str> {
    let mut all = h.get_all(name).into_iter();
    match (all.next(), all.next()) {
        (Some(v), None) => v.to_str().ok(),
        _ => None,
    }
}

/// The host part of a `Host` value (`name`, `name:port`, `[v6]` or
/// `[v6]:port`), or `None` when the rest is not a port.
fn host_name(host: &str) -> Option<&str> {
    let (name, rest) = if host.starts_with('[') {
        let end = host.find(']')?;
        (&host[..=end], &host[end + 1..])
    } else {
        let end = host.find(':').unwrap_or(host.len());
        (&host[..end], &host[end..])
    };
    let port_ok = rest.is_empty()
        || rest
            .strip_prefix(':')
            .is_some_and(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()));
    (port_ok && !name.is_empty()).then_some(name)
}

/// §2.2: `Origin` is present and not `null`; with its `scheme://`
/// removed it equals `Host` as sent, ASCII case-insensitively, with no
/// default port filled in; and on a gateway with authentication off the
/// host is `localhost`, `127.0.0.1` or `[::1]`, which closes DNS
/// rebinding for this endpoint.
pub(crate) fn origin_matches(origin: &str, host: &str, auth_required: bool) -> bool {
    let Some((scheme, rest)) = origin.split_once("://") else {
        return false;
    };
    let web = scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https");
    if !web || !rest.eq_ignore_ascii_case(host) {
        return false;
    }
    let Some(name) = host_name(host) else {
        return false;
    };
    auth_required
        || name.eq_ignore_ascii_case("localhost")
        || name == "127.0.0.1"
        || name == "[::1]"
}

fn same_origin(h: &HeaderMap, auth_required: bool) -> bool {
    match (single(h, header::ORIGIN), single(h, header::HOST)) {
        (Some(origin), Some(host)) => origin_matches(origin, host, auth_required),
        _ => false,
    }
}

// ===================================================================
// The handshake (§2.1-§2.4)
// ===================================================================

/// `GET /api/roomsim`.
pub(crate) async fn handler(
    upgrade: Result<WebSocketUpgrade, WebSocketUpgradeRejection>,
    State(app): State<AppState>,
    h: HeaderMap,
    RawQuery(query): RawQuery,
) -> Response {
    // 1. Before authentication, and not audited: a cross-site page, or a
    //    rebound name on an auth-off gateway, learns nothing more.
    if !same_origin(&h, app.auth.required()) {
        return error(StatusCode::FORBIDDEN, "the room stream needs a same-origin Origin (and, without authentication, a loopback Host)");
    }
    // 2. The principal, as for /api/live (a 401 is audited there).
    let who = match principal(&app, &h, PATH) {
        Ok(p) => p,
        Err(r) => return r,
    };
    // 3. Simulated positions: admin and safety only (ADR-008 d10).
    if !who.can_see_position(None) {
        app.audit.record(
            now_unix_ms(),
            Action::Denied,
            &who.name,
            Some(who.role.as_str()),
            json!({ "path": PATH, "why": "position" }),
        );
        return error(
            StatusCode::FORBIDDEN,
            "the room stream needs the admin or safety role",
        );
    }
    let ws = match upgrade {
        Ok(ws) => ws,
        Err(not_a_websocket) => return not_a_websocket.into_response(),
    };
    // 4. A slot, held from here until the session thread exits.
    let slot = Slot::take(&app.shared);
    let mode = requested_mode(query.as_deref());
    let job = parse_query(query.as_deref())
        .and_then(|q| resolve(q, |scene| app.shared.room.recorded(scene)));
    let stop = app.stop.clone();
    ws.max_message_size(MAX_MESSAGE)
        .max_frame_size(MAX_MESSAGE)
        .on_upgrade(move |socket| connection(socket, slot, mode, job, stop))
}

// ===================================================================
// The connection task
// ===================================================================

/// What the session thread sends the connection task.
#[derive(Debug)]
pub(crate) enum Out {
    /// A pre-serialised message.
    Text(String),
    /// Close the socket with this code; nothing follows.
    Close(u16),
}

async fn connection(
    mut socket: WebSocket,
    slot: Option<Slot>,
    mode: Mode,
    job: Result<Job, Refusal>,
    mut stop: watch::Receiver<bool>,
) {
    let Some(slot) = slot else {
        // No free slot: say so, and build nothing (§2.4).
        return refuse(&mut socket, error_message(mode, "session", BUSY), 1013).await;
    };
    let job = match job {
        Ok(job) => job,
        Err(r) => {
            drop(slot);
            return refuse(&mut socket, error_message(mode, r.param, r.message), 1008).await;
        }
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let (commands, command_rx) = sync_channel(COMMAND_QUEUE);
    let (out_tx, mut out) = mpsc::channel(1);
    if spawn_session(job, slot, Arc::clone(&cancel), command_rx, out_tx).is_err() {
        let m = error_message(mode, "session", "the room simulation could not be started");
        return refuse(&mut socket, m, 1011).await;
    }
    let end = pump(&mut socket, mode, &commands, &mut out, &mut stop).await;
    // The thread first: it exits within 0.5 s of this, whatever the
    // client does with the close below.
    cancel.store(true, Ordering::Release);
    drop(commands);
    drop(out);
    match end {
        End::Close(code) => close_with(&mut socket, code).await,
        End::ClosedByClient => {
            // The library queues the close reply and sends it on the next
            // read (as in `http::ws_session`).
            let _ = timeout(CLOSE_WAIT, socket.recv()).await;
        }
        End::Gone => {}
    }
}

/// How a connection ends.
enum End {
    /// Send a close frame with this code.
    Close(u16),
    /// The client sent its close frame; it awaits the reply.
    ClosedByClient,
    /// The socket failed, or the session ended without a close code.
    Gone,
}

/// Whether the client still takes what it is sent (§3.2: one that
/// accepts no frame for [`ACCEPT_TIMEOUT`] is closed with 1008).
///
/// A socket write returns once the kernel has buffered the message, and
/// the kernel buffers megabytes, so the writes notice a client that
/// stopped reading only much later. Its pongs do not lie: a pong comes
/// back only once the client has read everything sent before the ping,
/// and browsers send it by themselves. So within a second of sending a
/// message the connection pings, numbering the pings; a pong answers every
/// ping up to its number (a client may answer only the latest, RFC 6455
/// §5.5.3). A ping that waits [`ACCEPT_TIMEOUT`] with no pong in between
/// means the client took nothing in that time. A slow client, whose pongs
/// come late but keep coming, is kept, and a session that sends nothing
/// (an ended run) pings nothing, so only its idle rule closes it.
struct Liveness {
    /// Pings sent, numbered from 1, and the highest number a pong
    /// answered.
    sent: u64,
    answered: u64,
    /// A message was sent since the last ping.
    unconfirmed: bool,
    /// Since when a ping has waited with no pong: when the first ping
    /// was sent while none was waiting, or the last pong that left some.
    waiting_since: Option<Instant>,
}

impl Liveness {
    fn new() -> Self {
        Liveness {
            sent: 0,
            answered: 0,
            unconfirmed: false,
            waiting_since: None,
        }
    }

    /// A message was sent.
    fn sent_message(&mut self) {
        self.unconfirmed = true;
    }

    /// The next ping's payload, if a message was sent since the last one.
    fn ping(&mut self, now: Instant) -> Option<[u8; 8]> {
        if !std::mem::take(&mut self.unconfirmed) {
            return None;
        }
        if self.answered == self.sent {
            self.waiting_since = Some(now);
        }
        self.sent += 1;
        Some(self.sent.to_be_bytes())
    }

    /// A pong; one that answers no ping waiting is ignored.
    fn pong(&mut self, payload: &[u8], now: Instant) {
        let Ok(n) = <[u8; 8]>::try_from(payload).map(u64::from_be_bytes) else {
            return;
        };
        if self.answered < n && n <= self.sent {
            self.answered = n;
            self.waiting_since = (n < self.sent).then_some(now);
        }
    }

    /// A ping has waited [`ACCEPT_TIMEOUT`] with no pong.
    fn stalled(&self, now: Instant) -> bool {
        self.waiting_since
            .is_some_and(|since| now.saturating_duration_since(since) >= ACCEPT_TIMEOUT)
    }
}

/// Forward messages and commands until the session or the socket ends.
/// Not biased: a client that is slow to read still gets its commands
/// read, and one that floods commands still gets its frames.
async fn pump(
    socket: &mut WebSocket,
    mode: Mode,
    commands: &SyncSender<Command>,
    out: &mut mpsc::Receiver<Out>,
    stop: &mut watch::Receiver<bool>,
) -> End {
    let mut live = Liveness::new();
    let mut pings = tokio::time::interval(PING_EVERY);
    pings.set_missed_tick_behavior(MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            _ = stopping(stop) => return End::Close(1001),
            _ = pings.tick() => {
                let now = Instant::now();
                if live.stalled(now) {
                    return End::Close(1008);
                }
                if let Some(n) = live.ping(now) {
                    if let Err(end) = deliver(socket, Message::Ping(n.to_vec().into()), stop).await {
                        return end;
                    }
                }
            }
            msg = out.recv() => match msg {
                Some(Out::Text(text)) => {
                    if let Err(end) = deliver(socket, Message::Text(text.into()), stop).await {
                        return end;
                    }
                    live.sent_message();
                }
                Some(Out::Close(code)) => return End::Close(code),
                None => return End::Gone,
            },
            incoming = socket.recv() => match incoming {
                Some(Ok(Message::Text(text))) => {
                    let refused = match parse_command(text.as_str()) {
                        Ok(cmd) => match commands.try_send(cmd) {
                            Ok(()) => None,
                            Err(TrySendError::Full(_)) => Some(Refusal { param: "cmd", message: "too many commands at once; this one was not applied" }),
                            // The thread has ended; its last messages are
                            // still to come on `out`.
                            Err(TrySendError::Disconnected(_)) => None,
                        },
                        Err(r) => Some(r),
                    };
                    if let Some(r) = refused {
                        // The session continues (§3.3).
                        let m = error_message(mode, r.param, r.message).to_string();
                        if let Err(end) = deliver(socket, Message::Text(m.into()), stop).await {
                            return end;
                        }
                        live.sent_message();
                    }
                }
                Some(Ok(Message::Pong(p))) => live.pong(&p, Instant::now()),
                // Commands are JSON text (§2.4).
                Some(Ok(Message::Binary(_))) => return End::Close(1008),
                Some(Ok(Message::Close(_))) => return End::ClosedByClient,
                Some(Ok(_)) => {}
                None | Some(Err(_)) => return End::Gone,
            },
        }
    }
}

/// Resolves once the gateway's stop signal is set, or its sender is gone.
/// Its output holds no borrow of `stop`, so `select!` arms may use it.
async fn stopping(stop: &mut watch::Receiver<bool>) {
    let _ = stop.wait_for(|s| *s).await;
}

/// Send one message; `Err` with how the connection ends when the gateway
/// stops, the socket fails, or the client accepts nothing for
/// [`ACCEPT_TIMEOUT`] (§3.2: closed with 1008).
async fn deliver(
    socket: &mut WebSocket,
    message: Message,
    stop: &mut watch::Receiver<bool>,
) -> Result<(), End> {
    tokio::select! {
        biased;
        _ = stopping(stop) => Err(End::Close(1001)),
        sent = timeout(ACCEPT_TIMEOUT, socket.send(message)) => match sent {
            Ok(Ok(())) => Ok(()),
            Ok(Err(_)) => Err(End::Gone),
            Err(_) => Err(End::Close(1008)),
        },
    }
}

/// Close with `code`, then give the client a moment to answer, so it sees
/// the code rather than an abnormal closure (RFC 6455 §5.5.1).
async fn close_with(socket: &mut WebSocket, code: u16) {
    let frame = Message::Close(Some(CloseFrame {
        code,
        reason: "".into(),
    }));
    if let Ok(Ok(())) = timeout(CLOSE_WAIT, socket.send(frame)).await {
        let _ = timeout(CLOSE_WAIT, async {
            while let Some(Ok(m)) = socket.recv().await {
                if matches!(m, Message::Close(_)) {
                    break;
                }
            }
        })
        .await;
    }
}

/// One `error` message, then close with `code`: a refusal after the
/// upgrade (§2.4, §3.1).
async fn refuse(socket: &mut WebSocket, message: Value, code: u16) {
    if let Ok(Ok(())) = timeout(
        CLOSE_WAIT,
        socket.send(Message::Text(message.to_string().into())),
    )
    .await
    {
        close_with(socket, code).await;
    }
}

// ===================================================================
// The session thread
// ===================================================================

/// Start a session's thread. It owns `slot` and drops it as it exits; if
/// the thread cannot be started, the slot is dropped with the closure.
pub(crate) fn spawn_session(
    job: Job,
    slot: Slot,
    cancel: Arc<AtomicBool>,
    commands: Receiver<Command>,
    out: mpsc::Sender<Out>,
) -> std::io::Result<JoinHandle<()>> {
    std::thread::Builder::new()
        .name(THREAD_NAME.into())
        .spawn(move || {
            let _slot = slot;
            match job {
                Job::Sim {
                    scene,
                    variant,
                    seed,
                } => run_sim(scene, variant, seed, &cancel, &commands, &out),
                Job::Replay { recorded } => run_replay(recorded, &cancel, &commands, &out),
            }
        })
}

/// Send a message; false once the connection task is gone.
fn send(out: &mpsc::Sender<Out>, message: Value) -> bool {
    out.blocking_send(Out::Text(message.to_string())).is_ok()
}

/// Send an `error` message and close with `code`.
fn fail(out: &mpsc::Sender<Out>, mode: Mode, param: &str, message: &str, code: u16) {
    if send(out, error_message(mode, param, message)) {
        let _ = out.blocking_send(Out::Close(code));
    }
}

/// [`RoomSimSetup::new`]'s `keep_going`, and the empty run's: false once
/// the session is cancelled or its commands disconnect; meanwhile the
/// commands that arrive set the state the run starts in (§3.3).
fn keep_going(cancel: &AtomicBool, commands: &Receiver<Command>, start: &mut Start) -> bool {
    if cancel.load(Ordering::Acquire) {
        return false;
    }
    // A queue's worth at most, as in `drive`; the rest wait for the next
    // poll.
    for _ in 0..COMMAND_QUEUE {
        match commands.try_recv() {
            Ok(cmd) => start.apply(cmd),
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => return false,
        }
    }
    true
}

/// The empty run of `setup`'s scene and seed (§5.3), through
/// `setup.empty_room()`, which shares its calibration: the `unoccupied`
/// false alarms of WLM and RTI over the whole run. Its records are not
/// kept. `Ok(None)` when `keep_going` says stop.
pub(crate) fn empty_run(
    setup: &RoomSimSetup,
    keep_going: &mut dyn FnMut() -> bool,
) -> Result<Option<(FalseAlarms, FalseAlarms)>, PipelineError> {
    let empty = setup.empty_room()?;
    let mut stream = empty.stream()?;
    let mut ticks = Vec::new();
    let mut k = 0u32;
    while !stream.finished() {
        if !keep_going() {
            return Ok(None);
        }
        k += 1;
        let t = EMPTY_RUN_STEP_S * f64::from(k);
        let t_end = if t < empty.duration_s() {
            t
        } else {
            f64::INFINITY
        };
        ticks.extend(stream.advance_to(t_end)?.ticks);
    }
    let hz = empty.tick_hz();
    Ok(Some((
        false_alarms(&ticks, PersonEstimator::Wlm, hz),
        false_alarms(&ticks, PersonEstimator::Rti, hz),
    )))
}

/// A SIM session: `calibrating` (the calibration and the empty run), then
/// the engine.
fn run_sim(
    e: Embedded,
    variant: Variant,
    seed: u64,
    cancel: &AtomicBool,
    commands: &Receiver<Command>,
    out: &mpsc::Sender<Out>,
) {
    let mode = Mode::Sim;
    let Some(scene) = e.load() else {
        return fail(out, mode, "scene", "the embedded scene does not load", 1011);
    };
    let base = RoomSimConfig {
        seed,
        duration_s: ROOMSIM_RUN_S,
        ..RoomSimConfig::default()
    };
    let rs = variant.apply(&scene, &base);
    let calibrating = state_message(
        "calibrating",
        mode,
        &scene_block(e, &scene),
        &run_block(0, variant, seed, ROOMSIM_RUN_S),
    );
    if !send(out, calibrating) {
        return;
    }
    let mut start = Start::default();
    let mut go = || keep_going(cancel, commands, &mut start);
    let refused = |err: PipelineError| {
        fail(
            out,
            mode,
            err.param().unwrap_or("scene"),
            &err.to_string(),
            1008,
        )
    };
    let setup = match RoomSimSetup::new(&scene, &rs, &Config::default(), LABEL, &mut go) {
        Ok(Some(setup)) => setup,
        Ok(None) => return,
        Err(err) => return refused(err),
    };
    let empty = match variant {
        Variant::Empty => None,
        Variant::Declared | Variant::WithWalker => match empty_run(&setup, &mut go) {
            Ok(Some(fa)) => Some(fa),
            Ok(None) => return,
            Err(err) => return refused(err),
        },
    };
    let stream = match setup.stream() {
        Ok(stream) => stream,
        Err(err) => return refused(err),
    };
    let session = Session::sim(e, scene, &setup, variant, seed, empty);
    match Engine::new(
        Source::Sim {
            setup: &setup,
            stream: Some(Box::new(stream)),
        },
        session,
        start,
        Duration::ZERO,
    ) {
        Ok(engine) => drive(engine, cancel, commands, out),
        Err(why) => fail(out, mode, "duration_s", why, 1011),
    }
}

/// A REPLAY session: `ready`, then the engine on the recorded run.
fn run_replay(
    recorded: Arc<Recorded>,
    cancel: &AtomicBool,
    commands: &Receiver<Command>,
    out: &mpsc::Sender<Out>,
) {
    let mode = Mode::Replay;
    let Some(session) = Session::replay(&recorded) else {
        return fail(out, mode, "scene", "the embedded scene does not load", 1011);
    };
    if !send(
        out,
        state_message("ready", mode, session.scene_block(), &session.run_block(0)),
    ) {
        return;
    }
    let mut start = Start::default();
    if !keep_going(cancel, commands, &mut start) {
        return;
    }
    match Engine::new(
        Source::Replay {
            recorded,
            tick: 0,
            epoch: 0,
        },
        session,
        start,
        Duration::ZERO,
    ) {
        Ok(engine) => drive(engine, cancel, commands, out),
        Err(why) => fail(out, mode, "duration_s", why, 1011),
    }
}

/// The session clock: the engine, on this thread's wall clock. A frame is
/// computed only once the connection task has taken the previous message
/// (§3.2), and the commands that arrived meanwhile are applied first.
fn drive(
    mut engine: Engine<'_>,
    cancel: &AtomicBool,
    commands: &Receiver<Command>,
    out: &mpsc::Sender<Out>,
) {
    // Only to wait for the channel to empty: no tasks, no drivers.
    let Ok(rt) = tokio::runtime::Builder::new_current_thread().build() else {
        return;
    };
    let t0 = Instant::now();
    loop {
        if cancel.load(Ordering::Acquire) {
            return;
        }
        // The previous message was taken (the permit is dropped at once:
        // this thread is the only sender, so `blocking_send` below does
        // not wait). A closed connection ends the wait at once.
        if rt.block_on(out.reserve()).is_err() {
            return;
        }
        // A queue's worth at most, so a flood of commands cannot hold back
        // the frames; every command queued before the drain is applied.
        for _ in 0..COMMAND_QUEUE {
            match commands.try_recv() {
                Ok(cmd) => engine.command(cmd, t0.elapsed()),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return,
            }
        }
        match engine.next(t0.elapsed()) {
            Next::Frame(frame) => {
                if !send(out, frame) {
                    return;
                }
            }
            Next::Wait(until) => {
                match commands.recv_timeout(until.saturating_sub(t0.elapsed()).min(POLL)) {
                    Ok(cmd) => engine.command(cmd, t0.elapsed()),
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => return,
                }
            }
            Next::Close { message, code } => {
                if send(out, message) {
                    let _ = out.blocking_send(Out::Close(code));
                }
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roomstream::engine::session_t;
    use crate::roomstream::testkit::*;
    use crate::shared::Shared;

    #[test]
    fn origin_must_be_the_host_as_sent() {
        let ok = [
            ("http://127.0.0.1:8080", "127.0.0.1:8080", false),
            ("http://localhost:8080", "localhost:8080", false),
            ("HTTP://LocalHost:8080", "localhost:8080", false),
            ("http://[::1]:8080", "[::1]:8080", false),
            ("http://localhost", "localhost", false),
            ("https://gw.example", "gw.example", true),
            ("http://evil.example:8080", "evil.example:8080", true),
        ];
        for (origin, host, auth) in ok {
            assert!(
                origin_matches(origin, host, auth),
                "{origin} / {host} / auth {auth}"
            );
        }
        let refused = [
            ("null", "127.0.0.1:8080", true),
            ("", "127.0.0.1:8080", true),
            ("127.0.0.1:8080", "127.0.0.1:8080", true),
            ("http://127.0.0.1:8081", "127.0.0.1:8080", true),
            ("http://127.0.0.1", "127.0.0.1:8080", true),
            ("http://127.0.0.1:80", "127.0.0.1", true),
            ("http://evil.example", "127.0.0.1:8080", true),
            ("http://127.0.0.1:8080/", "127.0.0.1:8080", true),
            ("file://127.0.0.1:8080", "127.0.0.1:8080", true),
            ("ws://127.0.0.1:8080", "127.0.0.1:8080", true),
            ("http://[::1]", "::1", true),
            // DNS rebinding: both carry the attacker's name.
            ("http://evil.example:8080", "evil.example:8080", false),
            ("http://127.0.0.2:8080", "127.0.0.2:8080", false),
            ("http://localhost.evil:8080", "localhost.evil:8080", false),
            ("http://[::1]x:8080", "[::1]x:8080", false),
            ("http://127.0.0.1:80a", "127.0.0.1:80a", false),
            ("http://127.0.0.1:", "127.0.0.1:", false),
        ];
        for (origin, host, auth) in refused {
            assert!(
                !origin_matches(origin, host, auth),
                "{origin} / {host} / auth {auth}"
            );
        }
        let mut h = HeaderMap::new();
        h.insert(header::HOST, "127.0.0.1:1".parse().unwrap());
        assert!(!same_origin(&h, false), "no Origin");
        h.insert(header::ORIGIN, "http://127.0.0.1:1".parse().unwrap());
        assert!(same_origin(&h, false));
        h.append(header::ORIGIN, "http://127.0.0.1:1".parse().unwrap());
        assert!(!same_origin(&h, false), "two Origins");
    }

    /// The parts of a session the tests drive: its channels and thread.
    struct Running {
        shared: Arc<Shared>,
        cancel: Arc<AtomicBool>,
        commands: SyncSender<Command>,
        out: mpsc::Receiver<Out>,
        thread: JoinHandle<()>,
    }

    fn start(shared: &Arc<Shared>, job: Job) -> Running {
        let slot = Slot::take(shared).expect("a free slot");
        let cancel = Arc::new(AtomicBool::new(false));
        let (commands, command_rx) = sync_channel(COMMAND_QUEUE);
        let (out_tx, out) = mpsc::channel(1);
        let thread = spawn_session(job, slot, Arc::clone(&cancel), command_rx, out_tx).unwrap();
        Running {
            shared: Arc::clone(shared),
            cancel,
            commands,
            out,
            thread,
        }
    }

    impl Running {
        fn next(&mut self) -> Value {
            match self.out.blocking_recv() {
                Some(Out::Text(t)) => serde_json::from_str(&t).unwrap(),
                other => panic!("{other:?}"),
            }
        }

        /// How long the thread takes to exit after `end`, and that it
        /// released its slot.
        fn exit_time(self, end: impl FnOnce(&AtomicBool)) -> Duration {
            let Running {
                shared,
                cancel,
                commands,
                out,
                thread,
            } = self;
            let t = Instant::now();
            end(&cancel);
            drop((commands, out));
            thread.join().unwrap();
            let took = t.elapsed();
            assert_eq!(
                shared.room.sessions.load(Ordering::Acquire),
                0,
                "the slot is released"
            );
            took
        }
    }

    fn bedroom(variant: Variant) -> Job {
        Job::Sim {
            scene: Embedded::find("bedroom").unwrap(),
            variant,
            seed: 0,
        }
    }

    /// §2.5, §7: a running session puts nothing on `live_tx` or `outputs`
    /// and publishes nothing, even on a gateway whose own mode is live,
    /// and its frames say `sim`. Its thread is named, and exits within
    /// 0.5 s of its connection going away.
    #[test]
    fn a_running_session_puts_nothing_on_live_tx_or_outputs() {
        let shared = Shared::new(Mode::Live);
        let mut live = shared.live_tx.subscribe();
        let mut outputs = shared.outputs.subscribe();
        let mut s = start(&shared, bedroom(Variant::WithWalker));
        assert_eq!(s.thread.thread().name(), Some(THREAD_NAME));
        let first = s.next();
        assert_eq!(
            (first["type"].as_str(), first["state"].as_str()),
            (Some("state"), Some("calibrating"))
        );
        assert_eq!(first["mode"], "sim");
        // Commands sent while calibrating set the start state.
        s.commands
            .send(Command::Speed(crate::roomstream::params::Speed::ALL[5]))
            .unwrap();
        let mut frames = Vec::new();
        while frames.len() < 12 {
            let f = s.next();
            assert_eq!(f["type"], "frame", "{f}");
            frames.push(f);
        }
        assert!(frames.iter().all(|f| f["mode"] == "sim"));
        assert!(
            frames[1..].iter().all(|f| f["speed"] == 8.0),
            "the speed set while calibrating"
        );
        assert!(frames.iter().any(|f| f["state"] == "running"));
        assert!(matches!(
            live.try_recv(),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty)
        ));
        assert!(matches!(
            outputs.try_recv(),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty)
        ));
        assert!(shared.read().state_msg.is_none() && shared.read().spaces.is_empty());
        assert!(shared.query_events(None, None, 10).is_empty());
        assert_eq!(shared.room.sessions.load(Ordering::Acquire), 1);
        let took = s.exit_time(|_| {});
        assert!(took < Duration::from_millis(500), "exited after {took:?}");
    }

    /// §3.0: a stop is noticed during calibration too, within 0.5 s, by
    /// the cancel flag (the gateway stopping) or by the channels closing.
    #[test]
    fn a_calibrating_session_stops_within_half_a_second() {
        let shared = Shared::new(Mode::Live);
        for by_flag in [true, false] {
            let hall = Job::Sim {
                scene: Embedded::find("hall-l-shaped").unwrap(),
                variant: Variant::WithWalker,
                seed: 0,
            };
            let mut s = start(&shared, hall);
            assert_eq!(s.next()["state"], "calibrating");
            std::thread::sleep(Duration::from_millis(50));
            let took = if by_flag {
                // The flag alone, the channels kept open.
                let Running {
                    shared,
                    cancel,
                    commands,
                    out,
                    thread,
                } = s;
                let t = Instant::now();
                cancel.store(true, Ordering::Release);
                thread.join().unwrap();
                let took = t.elapsed();
                drop((commands, out));
                assert_eq!(shared.room.sessions.load(Ordering::Acquire), 0);
                took
            } else {
                s.exit_time(|_| {})
            };
            assert!(
                took < Duration::from_millis(500),
                "by flag {by_flag}: exited after {took:?}"
            );
        }
    }

    #[test]
    fn a_replay_session_is_ready_at_once_and_plays_the_file() {
        let shared = Shared::new(Mode::Live);
        let mut s = start(
            &shared,
            Job::Replay {
                recorded: bedroom_recorded(),
            },
        );
        let ready = s.next();
        assert_eq!(
            (
                ready["state"].as_str(),
                ready["mode"].as_str(),
                ready["label"].as_str()
            ),
            (Some("ready"), Some("replay"), Some("SIM"))
        );
        assert_eq!(ready["run"]["variant"], "with_walker");
        let f = s.next();
        assert_eq!(
            (f["t"].as_f64(), f["mode"].as_str()),
            (Some(0.0), Some("replay"))
        );
        assert!(f.get("truth").is_none());
        let took = s.exit_time(|c| c.store(true, Ordering::Release));
        assert!(took < Duration::from_millis(500), "{took:?}");
    }

    /// §3.2: a frame is computed only once the previous one was taken. A
    /// reader that stops for 450 ms every 3 frames misses none (`seq` has
    /// no gaps, `t` steps by exactly 4 quanta at 1x), and a command sent
    /// while a frame waits in the channel applies to the frame after it.
    #[test]
    fn a_slow_reader_misses_no_frame_and_its_commands_apply_to_the_next() {
        let shared = Shared::new(Mode::Live);
        let mut s = start(
            &shared,
            Job::Replay {
                recorded: bedroom_recorded(),
            },
        );
        assert_eq!(s.next()["state"], "ready");
        for k in 0..12u64 {
            if k % 3 == 1 {
                std::thread::sleep(Duration::from_millis(450));
            }
            let f = s.next();
            assert_eq!(
                (f["seq"].as_u64(), f["t"].as_f64()),
                (Some(k), Some(session_t(4 * k)))
            );
        }
        // Frame 12 is computed at its deadline and waits in the channel;
        // a pause sent meanwhile applies to frame 13, at the same t.
        std::thread::sleep(Duration::from_millis(550));
        s.commands.send(Command::Pause).unwrap();
        std::thread::sleep(Duration::from_millis(450));
        let waited = s.next();
        let paused = s.next();
        let seen = |f: &Value| (f["seq"].as_u64(), f["t"].as_f64(), f["paused"].as_bool());
        assert_eq!(seen(&waited), (Some(12), Some(session_t(48)), Some(false)));
        assert_eq!(seen(&paused), (Some(13), Some(session_t(48)), Some(true)));
        let took = s.exit_time(|c| c.store(true, Ordering::Release));
        assert!(took < Duration::from_millis(500), "{took:?}");
    }

    /// A client that floods commands (restarts, the dearest) gets its
    /// frames at the clock's pace: neither held back nor hurried.
    #[test]
    fn a_flood_of_restarts_does_not_hold_back_the_frames() {
        let shared = Shared::new(Mode::Live);
        let mut s = start(&shared, bedroom(Variant::WithWalker));
        assert_eq!(s.next()["state"], "calibrating");
        assert_eq!(s.next()["seq"], 0);
        let flooding = Arc::new(AtomicBool::new(true));
        let flood = {
            let (commands, flooding) = (s.commands.clone(), Arc::clone(&flooding));
            std::thread::spawn(move || {
                let t = Instant::now();
                let mut sent = 0u64;
                while t.elapsed() < Duration::from_secs(2) {
                    sent += u64::from(commands.try_send(Command::Restart).is_ok());
                }
                flooding.store(false, Ordering::Release);
                sent
            })
        };
        let (mut seq, mut during) = (0, 0);
        while flooding.load(Ordering::Acquire) {
            let f = s.next();
            seq += 1;
            assert_eq!(f["seq"], seq);
            during += usize::from(flooding.load(Ordering::Acquire));
        }
        let sent = flood.join().unwrap();
        assert!(sent > 1000, "not a flood: {sent} restarts");
        assert!(
            (5..=25).contains(&during),
            "{during} frames in 2 s of {sent} restarts"
        );
        let took = s.exit_time(|c| c.store(true, Ordering::Release));
        assert!(took < Duration::from_millis(500), "{took:?}");
    }

    /// §3.2's 10 s rule, on an injected clock: a ping unanswered for 10 s
    /// with no pong in between, and only then.
    #[test]
    fn a_client_is_stalled_once_a_ping_waits_10_s_with_no_pong() {
        let t0 = Instant::now();
        let at = |s: u64| t0 + Duration::from_secs(s);
        let mut l = Liveness::new();
        // Nothing sent, nothing pinged: an ended run waits for its idle
        // rule.
        assert_eq!(l.ping(at(1)), None);
        assert!(!l.stalled(at(3600)));
        // A client that answers.
        for k in 1..=30 {
            l.sent_message();
            let p = l.ping(at(k)).unwrap();
            assert_eq!(u64::from_be_bytes(p), k);
            l.pong(&p, at(k));
            assert!(!l.stalled(at(k + 60)));
        }
        // It stops reading: the first ping it leaves starts the 10 s.
        for k in 31..=45 {
            l.sent_message();
            assert!(l.ping(at(k)).is_some());
            assert_eq!(l.stalled(at(k)), k >= 41, "{k}");
        }
        // A pong for a ping never sent, or of another size, is no answer.
        l.pong(&999u64.to_be_bytes(), at(45));
        l.pong(&[0, 31], at(45));
        assert!(l.stalled(at(45)));
        // A slow client: a late pong is progress, and the next 10 s run
        // from it.
        l.pong(&31u64.to_be_bytes(), at(46));
        assert!(!l.stalled(at(55)));
        assert!(l.stalled(at(56)));
        // A pong for the latest ping answers them all; an older one after
        // it changes nothing.
        l.pong(&45u64.to_be_bytes(), at(56));
        l.pong(&40u64.to_be_bytes(), at(57));
        assert!(!l.stalled(at(1000)));
        // The last message of an ended run is pinged once; unanswered, it
        // stalls the client 10 s later with nothing more sent.
        l.sent_message();
        assert!(l.ping(at(1000)).is_some());
        assert_eq!(l.ping(at(1001)), None, "nothing new, no ping");
        assert!(!l.stalled(at(1009)));
        assert!(l.stalled(at(1010)));
    }
}
