//! Simulation speed: an 8-hour night, 3 links x 56 subcarriers at 20 Hz.
//! SYNTHETIC. `cargo run --release -p saans-sim --example speed`
use saans_sim::{scenarios, Simulator};
use std::time::Instant;

fn main() {
    let sc = scenarios::scenario("night-8h", scenarios::room_rich_nlos(), scenarios::three_links(), scenarios::night(8.0, 4.0, (10.0, 40.0)), 1);
    let t0 = Instant::now();
    let mut sim = Simulator::new(&sc);
    let build = t0.elapsed().as_secs_f64();
    let (mut ticks, mut frames, mut checksum) = (0u64, 0u64, 0.0f64);
    let t1 = Instant::now();
    while let Some(tick) = sim.step() {
        ticks += 1;
        frames += tick.frames.len() as u64;
        if let Some(f) = tick.frames.first() {
            checksum += f.frame.amplitudes[0];
        }
    }
    let run = t1.elapsed().as_secs_f64();
    println!(
        "{}: {} links x {} subcarriers, {:.1} h simulated: {ticks} ticks, {frames} frames in {run:.2} s ({:.0} ticks/s, {:.0}x real time; setup {:.1} ms) [checksum {checksum:.3}]",
        sc.name,
        sc.links.len(),
        sc.n_subcarriers,
        sim.duration_s() / 3600.0,
        ticks as f64 / run,
        sim.duration_s() / run,
        build * 1e3
    );
}
