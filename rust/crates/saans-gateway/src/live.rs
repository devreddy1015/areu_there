//! Live mode: one UDP socket, a 0.5 s tick, the HTTP server, clean
//! shutdown. Also the in-process handle the integration tests drive.
//!
//! The core task owns the [`Hub`] (and so the engine) outright. It waits
//! on, in priority order: shutdown, a datagram, the tick, and control
//! commands. HTTP handlers only read what the hub publishes.
//!
//! [`ClockMode::Manual`] replaces the 0.5 s timer with explicit
//! [`Gateway::advance_to`] calls, so tests can run an hour of gateway time
//! in milliseconds while still going through real UDP and HTTP.

use crate::config::GatewayConfig;
use crate::engine::TICK_S;
use crate::hub::{local_epoch_s, tick_until, Hub, Wall};
use crate::shared::Shared;
use crate::timefmt::now_unix_ms;
use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::{TcpListener, UdpSocket};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClockMode {
    /// `t_s` follows a monotonic clock; ticks every 0.5 s.
    Live,
    /// `t_s` starts at `t0` and moves only on `advance_to` (tests).
    Manual { t0: f64 },
}

enum Control {
    /// Drain queued datagrams, tick up to t, set the clock to t.
    AdvanceTo(f64, oneshot::Sender<u64>),
    /// Drain queued datagrams; reply with the datagram count.
    Drain(oneshot::Sender<u64>),
}

enum Clock {
    Live { start: Instant, anchor: f64 },
    Manual(f64),
}

impl Clock {
    fn now(&self) -> f64 {
        match self {
            Clock::Live { start, anchor } => anchor + start.elapsed().as_secs_f64(),
            Clock::Manual(t) => *t,
        }
    }
}

/// A running gateway.
pub struct Gateway {
    pub udp_addr: SocketAddr,
    pub http_addr: SocketAddr,
    pub shared: Arc<Shared>,
    control: mpsc::Sender<Control>,
    stop: watch::Sender<bool>,
    core: JoinHandle<()>,
    http: JoinHandle<()>,
    /// MQTT and alert-routing tasks.
    outputs: Vec<JoinHandle<()>>,
}

impl Gateway {
    /// Bind both sockets and start. Port 0 in the config picks a free port
    /// (tests); the bound addresses are in `udp_addr` / `http_addr`.
    pub async fn start(cfg: GatewayConfig, clock: ClockMode) -> io::Result<Gateway> {
        let udp_bind = cfg
            .udp_socket_addr()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.0))?;
        let http_bind = cfg
            .http_socket_addr()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.0))?;
        // Deliberately no SO_REUSEADDR: a second listener on the port must
        // fail loudly, not silently split the datagrams (recorder.py).
        let in_use = |what: &str, addr: SocketAddr, e: io::Error| {
            io::Error::new(e.kind(), format!("cannot bind {what} {addr}: {e}"))
        };
        let std_sock =
            std::net::UdpSocket::bind(udp_bind).map_err(|e| in_use("UDP", udp_bind, e))?;
        std_sock.set_nonblocking(true)?;
        let drain_sock = std_sock.try_clone()?;
        let sock = UdpSocket::from_std(std_sock)?;
        let listener = TcpListener::bind(http_bind)
            .await
            .map_err(|e| in_use("HTTP", http_bind, e))?;
        let (udp_addr, http_addr) = (sock.local_addr()?, listener.local_addr()?);

        let shared = Shared::with_history("live", cfg.history);
        let cfg_arc = Arc::new(cfg.clone());
        let alerts = if cfg.alert_routes.is_empty() {
            None
        } else {
            let dir = if cfg.data_dir.as_os_str().is_empty() {
                None
            } else {
                Some(cfg.data_dir.as_path())
            };
            Some(crate::alerts::AlertService::new(&cfg, dir)?)
        };
        let (clock, wall) = match clock {
            ClockMode::Live => {
                let anchor = local_epoch_s(now_unix_ms(), cfg.utc_offset_minutes);
                (
                    Clock::Live {
                        start: Instant::now(),
                        anchor,
                    },
                    Wall::System,
                )
            }
            ClockMode::Manual { t0 } => (
                Clock::Manual(t0),
                Wall::Anchored {
                    t0,
                    unix_ms_at_t0: now_unix_ms(),
                },
            ),
        };
        let t0 = clock.now();
        // Read once, here: an unset or malformed key variable stops startup
        // rather than leaving authentication silently off.
        let frame_keys = crate::frameauth::load_site_keys(&cfg)?;
        let mut hub = Hub::new(cfg, Some(shared.clone()), true, wall, t0);
        hub.engine.set_frame_keys(frame_keys);
        if !hub.engine.frame_auth_required() && !udp_bind.ip().is_loopback() {
            hub.notes.push(format!(
                "FeatureFrames are not authenticated (no [[frame_key]]): anything that can reach udp {udp_bind} can send them. See ADR-012 and README \"Operating requirements\""
            ));
        }
        for n in &hub.notes {
            crate::log(n);
        }
        hub.publish(t0);

        let (stop, stop_rx) = watch::channel(false);
        let (control, control_rx) = mpsc::channel(16);
        // Output consumers subscribe before the core starts, so they see
        // every event from the first tick.
        let mut outputs = Vec::new();
        if let Some(a) = &alerts {
            outputs.push(a.spawn(shared.outputs.subscribe(), stop_rx.clone()));
        }
        if let Some(m) = &cfg_arc.mqtt {
            crate::mqtt::require_env(m)?;
            outputs.push(crate::mqtt::spawn_with_status(
                &cfg_arc,
                shared.outputs.subscribe(),
                stop_rx.clone(),
                shared.mqtt.clone(),
            ));
        }
        let core = tokio::spawn(run_core(
            hub,
            sock,
            drain_sock,
            clock,
            control_rx,
            stop_rx.clone(),
        ));
        let app = crate::http::router(crate::http::HttpDeps::new(
            shared.clone(),
            stop_rx.clone(),
            cfg_arc,
            alerts,
        ));
        let mut http_stop = stop_rx;
        let http = tokio::spawn(async move {
            let serve = axum::serve(listener, app).with_graceful_shutdown(async move {
                let _ = http_stop.wait_for(|s| *s).await;
            });
            if let Err(e) = serve.await {
                crate::log(&format!("HTTP server error: {e}"));
            }
        });
        Ok(Gateway {
            udp_addr,
            http_addr,
            shared,
            control,
            stop,
            core,
            http,
            outputs,
        })
    }

    /// Manual clock: process queued datagrams, then run ticks up to `t_s`.
    /// Returns the number of datagrams received so far.
    pub async fn advance_to(&self, t_s: f64) -> u64 {
        let (tx, rx) = oneshot::channel();
        if self
            .control
            .send(Control::AdvanceTo(t_s, tx))
            .await
            .is_err()
        {
            return 0;
        }
        rx.await.unwrap_or(0)
    }

    /// Process queued datagrams; returns the number received so far.
    pub async fn drain(&self) -> u64 {
        let (tx, rx) = oneshot::channel();
        if self.control.send(Control::Drain(tx)).await.is_err() {
            return 0;
        }
        rx.await.unwrap_or(0)
    }

    /// Stop, persist history, close WebSockets, wait for both tasks.
    pub async fn shutdown(self) {
        let _ = self.stop.send(true);
        let _ = self.core.await;
        if tokio::time::timeout(Duration::from_secs(5), self.http)
            .await
            .is_err()
        {
            crate::log("HTTP server did not stop within 5 s");
        }
        for task in self.outputs {
            if tokio::time::timeout(Duration::from_secs(5), task)
                .await
                .is_err()
            {
                crate::log("an output task (MQTT / alerts) did not stop within 5 s");
            }
        }
    }
}

fn drain(sock: &std::net::UdpSocket, buf: &mut [u8], hub: &mut Hub, t: f64) {
    // A datagram that loopback delivered before this command is already
    // queued; read it now even if the reactor has not noticed it yet.
    while let Ok((n, src)) = sock.recv_from(buf) {
        hub.ingest(&buf[..n], t, Some(src));
    }
}

async fn run_core(
    mut hub: Hub,
    sock: UdpSocket,
    drain_sock: std::net::UdpSocket,
    mut clock: Clock,
    mut control: mpsc::Receiver<Control>,
    mut stop: watch::Receiver<bool>,
) {
    let live = matches!(clock, Clock::Live { .. });
    let mut buf = vec![0u8; 65_536];
    let mut drain_buf = vec![0u8; 65_536];
    let mut ticker = tokio::time::interval_at(
        tokio::time::Instant::now() + Duration::from_millis(500),
        Duration::from_millis(500),
    );
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut next_tick = clock.now() + TICK_S;
    loop {
        tokio::select! {
            biased;
            _ = stop.changed() => break,
            r = sock.recv_from(&mut buf) => match r {
                Ok((n, src)) => {
                    let t = clock.now();
                    hub.ingest(&buf[..n], t, Some(src));
                }
                Err(e) => {
                    crate::log(&format!("UDP receive error: {e}"));
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            },
            _ = ticker.tick(), if live => hub.tick(clock.now()),
            Some(cmd) = control.recv() => {
                drain(&drain_sock, &mut drain_buf, &mut hub, clock.now());
                match cmd {
                    Control::AdvanceTo(t, ack) => {
                        if let Clock::Manual(now) = &mut clock {
                            while next_tick <= t + 1e-9 {
                                *now = next_tick;
                                tick_until(&mut hub, &mut next_tick, *now);
                            }
                            *now = now.max(t);
                        }
                        let _ = ack.send(hub.engine.stats.datagrams);
                    }
                    Control::Drain(ack) => {
                        let _ = ack.send(hub.engine.stats.datagrams);
                    }
                }
            }
        }
    }
    let t = clock.now();
    hub.shutdown(t);
}
