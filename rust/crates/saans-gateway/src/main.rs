//! `saans-gateway --config saans.toml` (live) or
//! `saans-gateway --config saans.toml --replay rec.pcap [--speed 0] [--events-out ev.jsonl]`.

use clap::Parser;
use saans_gateway::{log, ClockMode, Gateway, GatewayConfig, ReplayOptions};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Parser)]
#[command(name = "saans-gateway", version, about = "Saans gateway: camera-free WiFi health and safety sensing")]
struct Args {
    /// Configuration file (TOML; see saans.example.toml).
    #[arg(long, short, default_value = "saans.toml")]
    config: PathBuf,
    /// Replay a pcap recording instead of listening on UDP.
    #[arg(long, value_name = "FILE.pcap")]
    replay: Option<PathBuf>,
    /// Replay speed: 0 = as fast as possible (deterministic), 1 = real time.
    #[arg(long, default_value_t = 0.0, requires = "replay")]
    speed: f64,
    /// Also write the replay's events as JSON lines.
    #[arg(long, value_name = "FILE.jsonl", requires = "replay")]
    events_out: Option<PathBuf>,
    /// Replay only datagrams sent to this UDP port (default: all).
    #[arg(long, requires = "replay")]
    port: Option<u16>,
    /// Serve the API and dashboard during the replay, and after it until Ctrl-C.
    #[arg(long, requires = "replay")]
    serve: bool,
    /// Validate the configuration and exit.
    #[arg(long)]
    check: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let cfg = match GatewayConfig::load(&args.config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("saans-gateway: {e}");
            return ExitCode::from(2);
        }
    };
    if args.check {
        println!("{}: OK ({} spaces)", args.config.display(), cfg.spaces.len());
        return ExitCode::SUCCESS;
    }
    if !(args.speed >= 0.0 && args.speed.is_finite()) {
        eprintln!("saans-gateway: --speed must be >= 0");
        return ExitCode::from(2);
    }
    let rt = match tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("saans-gateway: cannot start runtime: {e}");
            return ExitCode::FAILURE;
        }
    };
    let result = match args.replay.clone() {
        Some(pcap) => rt.block_on(run_replay(cfg, args, pcap)),
        None => rt.block_on(run_live(cfg)),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("saans-gateway: {e}");
            ExitCode::FAILURE
        }
    }
}

async fn run_live(cfg: GatewayConfig) -> Result<(), Box<dyn std::error::Error>> {
    let spaces = cfg.spaces.len();
    let data_dir = cfg.data_dir.clone();
    let gw = Gateway::start(cfg, ClockMode::Live).await?;
    log(&format!(
        "listening for sensors on udp {}, dashboard on http://{}/ ({spaces} spaces, data in {})",
        gw.udp_addr,
        gw.http_addr,
        data_dir.display()
    ));
    shutdown_signal().await;
    log("stopping: saving history");
    gw.shutdown().await;
    log("stopped");
    Ok(())
}

async fn run_replay(cfg: GatewayConfig, args: Args, pcap: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let opts = ReplayOptions { pcap, speed: args.speed, events_out: args.events_out, port: args.port };
    if !args.serve {
        let summary = saans_gateway::replay(cfg, &opts, None).await?;
        print!("{summary}");
        return Ok(());
    }
    // Serve while replaying: the HTTP side reads what the replay publishes.
    let shared = saans_gateway::shared::Shared::new("replay");
    let http_addr = cfg.http_socket_addr()?;
    let listener = tokio::net::TcpListener::bind(http_addr)
        .await
        .map_err(|e| std::io::Error::new(e.kind(), format!("cannot bind HTTP {http_addr}: {e}")))?;
    log(&format!("replay dashboard on http://{}/", listener.local_addr()?));
    let (stop, stop_rx) = tokio::sync::watch::channel(false);
    // Replay serving shows the recording's state; alerts and MQTT stay off.
    let app = saans_gateway::http::router(saans_gateway::http::HttpDeps::new(
        shared.clone(),
        stop_rx.clone(),
        std::sync::Arc::new(cfg.clone()),
        None,
    ));
    let mut http_stop = stop_rx;
    let server = tokio::spawn(async move {
        let _ = axum::serve(listener, app).with_graceful_shutdown(async move {
            let _ = http_stop.wait_for(|s| *s).await;
        })
        .await;
    });
    let summary = tokio::select! {
        r = saans_gateway::replay(cfg, &opts, Some(shared)) => Some(r?),
        _ = shutdown_signal() => None,
    };
    if let Some(s) = summary {
        print!("{s}");
        log("replay finished; still serving the final state until Ctrl-C");
        shutdown_signal().await;
    }
    let _ = stop.send(true);
    let _ = tokio::time::timeout(std::time::Duration::from_secs(5), server).await;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = term => {},
    }
}
