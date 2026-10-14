//! `saans-gateway --config saans.toml` (live),
//! `saans-gateway --config saans.toml --replay rec.pcap [--speed 0] [--events-out ev.jsonl]`,
//! or `saans-gateway --config saans.toml --simulate` (the sensing
//! explorer: a SYNTHETIC room through the real pipeline).
//! `saans-gateway --export-site DIR` writes the public site as static files.

use clap::{ArgGroup, Parser};
use saans_gateway::{log, ClockMode, Gateway, GatewayConfig, ReplayOptions, SimGateway, SimulateOptions};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Parser)]
#[command(name = "saans-gateway", version, about = "Saans gateway: camera-free WiFi health and safety sensing")]
// `--speed` paces whichever of the two offline sources is running, so it
// requires one of them rather than naming just `--replay`.
#[command(group(ArgGroup::new("source").args(["replay", "simulate"])))]
struct Args {
    /// Configuration file (TOML; see saans.example.toml).
    #[arg(long, short, default_value = "saans.toml")]
    config: PathBuf,
    /// Replay a pcap recording instead of listening on UDP.
    #[arg(long, value_name = "FILE.pcap")]
    replay: Option<PathBuf>,
    /// Run the sensing explorer: a SIMULATED room driven through the
    /// real DSP, fusion and detectors. Serves the API and dashboard the
    /// way live mode does and binds no UDP port, because there are no
    /// sensors. Every value it publishes is synthetic.
    #[arg(long)]
    simulate: bool,
    /// Replay / simulate speed: 0 = as fast as possible, 1 = real time,
    /// 10 = ten times faster. Default: 0 for --replay (deterministic),
    /// 1 for --simulate.
    #[arg(long, requires = "source")]
    speed: Option<f64>,
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
    /// Print an Argon2id PHC string for a password, to paste into
    /// `[[user]].password_hash`, and exit. The password is read from
    /// stdin so it never reaches the shell history or the process list.
    #[arg(long)]
    hash_password: bool,
    /// Write the public site into DIR as static files and exit: the copy
    /// published on GitHub Pages. DIR is created, and refused unless
    /// empty. No config is read; nothing the gateway computes is in it.
    #[arg(long, value_name = "DIR")]
    export_site: Option<PathBuf>,
}

fn main() -> ExitCode {
    let args = Args::parse();
    // Before the config: hashing a password is what you do *to write* one.
    if args.hash_password {
        return hash_password();
    }
    // Before the config too: the site is the same in every deployment.
    if let Some(dir) = &args.export_site {
        return match saans_gateway::site::export(dir) {
            Ok(files) => {
                println!("{}: {} files", dir.display(), files.len());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("saans-gateway: --export-site: {e}");
                ExitCode::FAILURE
            }
        };
    }
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
    // Replay defaults to "as fast as possible" because its output is a
    // summary; the explorer defaults to real time because its output is
    // a room someone is watching.
    let speed = args.speed.unwrap_or(if args.simulate { 1.0 } else { 0.0 });
    if !(speed >= 0.0 && speed.is_finite()) {
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
    let result = match (args.simulate, args.replay.clone()) {
        (true, _) => rt.block_on(run_simulate(cfg, speed)),
        (false, Some(pcap)) => rt.block_on(run_replay(cfg, args, speed, pcap)),
        (false, None) => rt.block_on(run_live(cfg)),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("saans-gateway: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `saans-gateway --hash-password`, the command `config.rs` names when it
/// rejects a plaintext password. Reads one line from stdin and prints the
/// PHC string; the salt is 16 bytes from the OS, not a seeded PRNG.
fn hash_password() -> ExitCode {
    use argon2::password_hash::{PasswordHasher, SaltString};
    use std::io::BufRead;
    eprint!("password: ");
    let mut line = String::new();
    if std::io::stdin().lock().read_line(&mut line).is_err() || line.trim().is_empty() {
        eprintln!("saans-gateway: no password on stdin");
        return ExitCode::from(2);
    }
    let mut salt = [0u8; 16];
    if getrandom::getrandom(&mut salt).is_err() {
        eprintln!("saans-gateway: no entropy available");
        return ExitCode::FAILURE;
    }
    let Ok(salt) = SaltString::encode_b64(&salt) else {
        eprintln!("saans-gateway: cannot encode salt");
        return ExitCode::FAILURE;
    };
    match argon2::Argon2::default().hash_password(line.trim_end_matches(['\n', '\r']).as_bytes(), &salt) {
        Ok(h) => {
            println!("{h}");
            ExitCode::SUCCESS
        }
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

/// The sensing explorer. Serving is not optional here the way it is for
/// a replay: a simulate run with nobody watching has no output at all.
async fn run_simulate(cfg: GatewayConfig, speed: f64) -> Result<(), Box<dyn std::error::Error>> {
    let gw = SimGateway::start(cfg, SimulateOptions { speed }).await?;
    let sc = gw.sim.scenario();
    log(&format!(
        "SIMULATED {:?} scenario: a {:.0} m x {:.0} m room, {} nodes, {} link(s), {:.0} Hz CSI. Explorer on http://{}/",
        sc.name,
        sc.room.size_m.0,
        sc.room.size_m.1,
        sc.nodes.len(),
        sc.links.len(),
        sc.sample_rate_hz,
        gw.http_addr,
    ));
    log("every value this mode publishes is SYNTHETIC: nothing here measured a real room, body or node");
    log("the room starts empty on purpose — the detectors calibrate on an empty room first, so the person appears a couple of minutes in (--speed 10 to hurry it along)");
    shutdown_signal().await;
    log("stopping");
    gw.shutdown().await;
    log("stopped");
    Ok(())
}

async fn run_replay(cfg: GatewayConfig, args: Args, speed: f64, pcap: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let opts = ReplayOptions { pcap, speed, events_out: args.events_out, port: args.port };
    if !args.serve {
        let summary = saans_gateway::replay(cfg, &opts, None).await?;
        print!("{summary}");
        return Ok(());
    }
    // Serve while replaying: the HTTP side reads what the replay publishes.
    let shared = saans_gateway::shared::Shared::with_history("replay", cfg.history);
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
