//! Behaviour of the simulator beyond Python parity: determinism, packet
//! loss, truth bookkeeping, motion bounds and the signal signatures the
//! detectors rely on. All SYNTHETIC.

use saans_dsp::BiquadBandpass;
use saans_sim::scenario::{Activity, Scenario, Segment, TruthEvent, TruthKind};
use saans_sim::scenarios::{self, CHAIR, DESK};
use saans_sim::{Simulator, MARGIN_M};

fn seg(duration_s: f64, activity: Activity) -> Segment {
    Segment { duration_s, activity }
}

fn seated(duration_s: f64) -> Segment {
    seg(duration_s, Activity::Seated { at: CHAIR, bpm: 15.0, bpm_end: None })
}

fn quiet(mut sc: Scenario) -> Scenario {
    sc.loss_prob = 0.0;
    sc
}

/// Amplitude series per subcarrier for link 0 (requires no loss).
fn amplitude_series(sim: &mut Simulator) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut t = Vec::new();
    let mut series: Vec<Vec<f64>> = Vec::new();
    while let Some(tick) = sim.step() {
        let f = &tick.frames[0];
        if series.is_empty() {
            series = vec![Vec::new(); f.frame.amplitudes.len()];
        }
        for (k, &a) in f.frame.amplitudes.iter().enumerate() {
            series[k].push(a);
        }
        t.push(tick.truth.t_s);
    }
    (t, series)
}

/// Mean square of `x` band-passed to [lo, hi] Hz, over t in [t0, t1).
fn band_power(t: &[f64], x: &[f64], lo: f64, hi: f64, t0: f64, t1: f64) -> f64 {
    let mut f = BiquadBandpass::new(lo, hi, 20.0);
    let (mut s, mut n) = (0.0, 0);
    for (&ti, &xi) in t.iter().zip(x) {
        let y = f.process(xi);
        if ti >= t0 && ti < t1 {
            s += y * y;
            n += 1;
        }
    }
    s / n as f64
}

#[test]
fn deterministic_under_a_seed() {
    let mk = |seed| scenarios::scenario("det", scenarios::room_rich_nlos(), scenarios::three_links(), scenarios::collapse(), seed);
    let run = |sc: &Scenario, n: usize| -> Vec<saans_sim::Tick> { Simulator::new(sc).take(n).collect() };
    let a = run(&mk(5), 4000);
    let b = run(&mk(5), 4000);
    assert_eq!(a, b, "same seed must give identical ticks");
    assert_eq!(Simulator::new(&mk(5)).truth_events(), Simulator::new(&mk(5)).truth_events());
    let c = run(&mk(6), 4000);
    assert_ne!(a[3000].frames[0].frame.amplitudes, c[3000].frames[0].frame.amplitudes);
}

#[test]
fn loss_rate_and_sequences() {
    let sc = scenarios::scenario("loss", scenarios::room_clear_los(), scenarios::three_links(), scenarios::empty(1.0), 11);
    let mut sim = Simulator::new(&sc);
    let n_ticks = 10_000u32;
    let mut received = [0u32; 3];
    let mut first: [Option<u32>; 3] = [None; 3];
    let mut last: [Option<u32>; 3] = [None; 3];
    let mut gaps = [0u32; 3];
    for _ in 0..n_ticks {
        let tick = sim.step().unwrap();
        for lf in &tick.frames {
            let i = sc.links.iter().position(|&l| l == (lf.tx, lf.rx)).unwrap();
            received[i] += 1;
            assert_eq!(lf.frame.sample_index, lf.seq as u64);
            assert_eq!(lf.frame.node_id, lf.rx);
            let dt = lf.t_s - tick.truth.t_s;
            assert!((0.0..0.002).contains(&dt), "jitter {dt}");
            if let Some(p) = last[i] {
                assert!(lf.seq > p);
                gaps[i] += lf.seq - p - 1;
            }
            first[i].get_or_insert(lf.seq);
            last[i] = Some(lf.seq);
        }
    }
    for i in 0..3 {
        let lost = n_ticks - received[i];
        let rate = lost as f64 / n_ticks as f64;
        // 2% +- ~4.3 sigma (sigma = 0.0014).
        assert!((rate - 0.02).abs() < 0.006, "link {i}: loss {rate}");
        // Every loss is visible in the sequence numbers.
        let (f, l) = (first[i].unwrap(), last[i].unwrap());
        assert_eq!(lost, f + gaps[i] + (n_ticks - 1 - l), "link {i}");
    }
}

fn assert_close(a: f64, b: f64, what: &str) {
    assert!((a - b).abs() < 1e-6, "{what}: {a} vs {b}");
}

#[test]
fn truth_events_match_the_timeline() {
    let fall_to = (3.4, 4.2);
    let timeline = vec![
        seg(120.0, Activity::Absent),
        seg(10.0, Activity::WalkTo { to: CHAIR, speed_mps: 1.0 }),
        seated(60.0),
        seg(20.0, Activity::BreathingPause { at: CHAIR }),
        seated(30.0),
        seg(200.0, Activity::Fidget { at: CHAIR, bpm: 14.0, every_s: 15.0, burst_s: 3.0 }),
        seg(60.8, Activity::Fall { to: fall_to, bpm: 18.0 }),
        seg(30.0, Activity::SitDown { at: DESK, bpm: 16.0 }),
        seg(60.0, Activity::Absent),
    ];
    let sc = quiet(scenarios::scenario("truth", scenarios::room_clear_los(), scenarios::two_links(), timeline, 21));
    let mut sim = Simulator::new(&sc);
    let ev = sim.truth_events();
    let of = |k: TruthKind| -> Vec<&TruthEvent> { ev.iter().filter(|e| e.kind == k).collect() };

    assert_close(sim.duration_s(), 590.8, "duration");
    let occ = of(TruthKind::Occupied);
    assert_eq!(occ.len(), 1);
    assert_close(occ[0].t_start_s, 120.0, "occupied start");
    assert_close(occ[0].t_end_s, 530.8, "occupied end");

    let pauses = of(TruthKind::BreathingPause);
    assert_eq!(pauses.len(), 1);
    assert_close(pauses[0].t_start_s, 190.0, "pause start");
    assert_close(pauses[0].t_end_s, 210.0, "pause end");

    let falls = of(TruthKind::Fall);
    assert_eq!(falls.len(), 1);
    assert_close(falls[0].t_start_s, 440.0, "fall start");
    assert_close(falls[0].t_end_s, 440.8, "fall end");

    let sits = of(TruthKind::SitDown);
    assert_eq!(sits.len(), 1);
    // Walk from the fall spot to 0.4 m short of the desk, then the drop.
    let d = (fall_to.0 - DESK.0).hypot(fall_to.1 - DESK.1);
    assert_close(sits[0].t_start_s, 500.8 + (d - 0.4), "sit-down start");
    assert_close(sits[0].t_end_s - sits[0].t_start_s, 0.6, "sit-down length");

    let walks = of(TruthKind::Walking);
    let door_to_chair = (0.6f64 - CHAIR.0).hypot(0.6 - CHAIR.1);
    assert_close(walks[0].t_start_s, 120.0, "walk in start");
    assert_close(walks[0].t_end_s, 120.0 + door_to_chair, "walk in end");
    assert_eq!(walks.len(), 2, "{walks:?}");

    let bursts = of(TruthKind::Burst);
    assert!(bursts.len() >= 5 && bursts.len() <= 25, "{} bursts", bursts.len());
    for b in &bursts {
        assert!(b.t_start_s >= 240.0 && b.t_end_s <= 440.0);
        assert_close(b.t_end_s - b.t_start_s, 3.0, "burst length");
    }
    assert!(ev.windows(2).all(|w| w[0].t_start_s <= w[1].t_start_s), "sorted");

    // Per-tick truth agrees with the intervals.
    let inside = |k: TruthKind, t: f64| ev.iter().any(|e| e.kind == k && t >= e.t_start_s && t < e.t_end_s);
    let mut fall_moving = 0;
    while let Some(tick) = sim.step() {
        let tr = tick.truth;
        let t = tr.t_s;
        assert_eq!(tr.present, inside(TruthKind::Occupied, t), "present @ {t}");
        assert_eq!(tr.in_pause, inside(TruthKind::BreathingPause, t), "pause @ {t}");
        assert_eq!(tr.in_burst, inside(TruthKind::Burst, t), "burst @ {t}");
        let moving = inside(TruthKind::Walking, t) || inside(TruthKind::Fall, t) || inside(TruthKind::SitDown, t);
        assert_eq!(tr.still, tr.present && !moving && !tr.in_burst, "still @ {t}");
        assert_eq!(tr.breathing, tr.present && !tr.in_pause);
        if inside(TruthKind::Fall, t) {
            fall_moving += 1;
        }
        if (250.0..440.0).contains(&t) && tr.breathing {
            assert_eq!(tr.bpm, 14.0);
        }
        if (441.0..500.0).contains(&t) {
            assert_eq!(tr.position, fall_to);
            assert_eq!(tr.bpm, 18.0);
        }
    }
    assert_eq!(fall_moving, 16, "0.8 s at 20 Hz");
}

#[test]
fn body_stays_inside_the_room_on_long_walks() {
    let timeline = vec![seg(10.0, Activity::Absent), seg(2.0 * 3600.0, Activity::Walk { speed_mps: 1.4 })];
    let sc = scenarios::scenario("walk", scenarios::room_clear_los(), scenarios::three_links(), timeline, 4);
    let (w, h) = sc.room.size_m;
    let sim = Simulator::new(&sc);
    let ev = sim.truth_events();
    let walks: Vec<_> = ev.iter().filter(|e| e.kind == TruthKind::Walking).collect();
    assert_eq!(walks.len(), 1, "one continuous walk");
    assert_close(walks[0].t_end_s - walks[0].t_start_s, 7200.0, "walk length");
    let (mut lo, mut hi) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
    let mut t = 10.0;
    while t < sim.duration_s() {
        let p = sim.truth_at(t).position;
        assert!(p.0 >= MARGIN_M - 1e-9 && p.0 <= w - MARGIN_M + 1e-9 && p.1 >= MARGIN_M - 1e-9 && p.1 <= h - MARGIN_M + 1e-9, "t={t}: {p:?}");
        lo = (lo.0.min(p.0), lo.1.min(p.1));
        hi = (hi.0.max(p.0), hi.1.max(p.1));
        t += 0.05;
    }
    // ...and it actually explores the room.
    assert!(lo.0 < 1.0 && lo.1 < 1.0 && hi.0 > w - 1.0 && hi.1 > h - 1.0, "{lo:?} {hi:?}");
    // A short walk at constant speed covers speed * time of path.
    let (mut path, mut prev) = (0.0, sim.truth_at(10.0).position);
    let mut t = 10.0;
    while t < 70.0 {
        t += 0.05;
        let p = sim.truth_at(t).position;
        path += (p.0 - prev.0).hypot(p.1 - prev.1);
        prev = p;
    }
    assert!((path - 84.0).abs() < 1.0, "path {path}");
}

#[test]
fn walking_has_far_more_motion_band_energy_than_sitting() {
    let mut ratios = Vec::new();
    for seed in [1, 2, 3] {
        let energy = |timeline: Vec<Segment>| {
            let sc = quiet(scenarios::scenario("e", scenarios::room_clear_los(), scenarios::one_link(), timeline, seed));
            let (t, series) = amplitude_series(&mut Simulator::new(&sc));
            series.iter().map(|x| band_power(&t, x, 1.0, 4.0, 10.0, 120.0)).sum::<f64>() / series.len() as f64
        };
        let walk = energy(vec![seg(120.0, Activity::Walk { speed_mps: 1.0 })]);
        let sit = energy(vec![seated(120.0)]);
        println!("seed {seed}: 1-4 Hz power walking {walk:.3}, sitting {sit:.3}, ratio {:.1}", walk / sit);
        ratios.push(walk / sit);
    }
    assert!(ratios.iter().all(|&r| r > 10.0), "{ratios:?}");
}

#[test]
fn pause_zeroes_the_breathing_modulation() {
    // A seat that is not a blind spot (checked below with the ground truth).
    let at = (2.6, 3.4);
    let sit = |d| seg(d, Activity::Seated { at, bpm: 15.0, bpm_end: None });
    let timeline = vec![sit(60.0), seg(40.0, Activity::BreathingPause { at }), sit(60.0)];
    let mut sc = quiet(scenarios::scenario("pause", scenarios::room_clear_los(), scenarios::one_link(), timeline, 8));
    sc.noise_std = 0.6;
    let mut sim = Simulator::new(&sc);
    let sens = sim.sensitivity((0, 1));
    let (k, best) = sens.iter().enumerate().fold((0, 0.0), |acc, (k, &s)| if s > acc.1 { (k, s) } else { acc });
    assert!(best > 0.5, "this seat is in a blind spot (best sensitivity {best})");
    let (t, series) = amplitude_series(&mut sim);
    let x = &series[k];
    let breathing = band_power(&t, x, 0.1, 0.5, 25.0, 60.0);
    let paused = band_power(&t, x, 0.1, 0.5, 75.0, 100.0);
    let after = band_power(&t, x, 0.1, 0.5, 125.0, 160.0);
    println!("0.1-0.5 Hz power on subcarrier {k} (sens {best:.2}): breathing {breathing:.4}, paused {paused:.5}, after {after:.4}");
    assert!(paused < 0.02 * breathing, "pause power {paused} vs breathing {breathing}");
    assert!(after > 0.3 * breathing, "breathing must resume");
}

#[test]
fn sensitivity_and_rssi_are_sane() {
    let sc = quiet(scenarios::scenario("s", scenarios::room_clear_los(), scenarios::three_links(), scenarios::collapse(), 3));
    let mut sim = Simulator::new(&sc);
    let mut rssi = Vec::new();
    let mut before = Vec::new();
    while let Some(tick) = sim.step() {
        if tick.truth.t_s < 100.0 {
            rssi.extend(tick.frames.iter().filter(|f| (f.tx, f.rx) == (0, 1)).map(|f| f.frame.rssi_dbm));
        }
        // Walking in, then lying after the fall: different positions.
        if (tick.truth.t_s - 125.0).abs() < 1e-9 {
            before = sim.sensitivity((0, 1));
        }
    }
    let after = sim.sensitivity((0, 1));
    for s in before.iter().chain(&after) {
        assert!((0.0..=1.0 + 1e-12).contains(s));
    }
    assert_ne!(before, after, "sensitivity follows the body");
    assert!(rssi.iter().all(|&r| (-52..=-38).contains(&r)), "{:?}", &rssi[..10]);
}

#[test]
fn ready_made_scenarios_build_and_label() {
    let layouts = [scenarios::one_link(), scenarios::two_links(), scenarios::three_links()];
    let rooms = [scenarios::room_clear_los(), scenarios::room_attenuated_los(), scenarios::room_rich_nlos()];
    let cases: Vec<(Vec<Segment>, TruthKind)> = vec![
        (scenarios::reader(2.0), TruthKind::Occupied),
        (scenarios::night(1.0, 6.0, (10.0, 40.0)), TruthKind::BreathingPause),
        (scenarios::collapse(), TruthKind::Fall),
        (scenarios::sit_down(), TruthKind::SitDown),
        (scenarios::tachypnea(), TruthKind::Occupied),
        (scenarios::lone_worker(5.0), TruthKind::Burst),
    ];
    for (i, (tl, kind)) in cases.into_iter().enumerate() {
        let sc = scenarios::scenario("x", rooms[i % 3].clone(), layouts[i % 3].clone(), tl, i as u64);
        let mut sim = Simulator::new(&sc);
        assert!(sim.truth_events().iter().any(|e| e.kind == kind), "case {i}: no {kind:?}");
        assert!(sim.truth_events().iter().all(|e| e.t_start_s >= 95.0), "case {i}: event inside the settle period");
        let first = sim.step().unwrap();
        assert!(!first.truth.present);
        assert!(first.frames.len() <= sc.links.len());
    }
    assert_close(Simulator::new(&scenarios::scenario("n", scenarios::room_clear_los(), scenarios::one_link(), scenarios::night(8.0, 4.0, (10.0, 40.0)), 1)).duration_s(), 28800.0, "8 h");
    // Tachypnea truth reports the fast rate.
    let tl = scenarios::tachypnea();
    let fast_start: f64 = tl[..3].iter().map(|s| s.duration_s).sum();
    let sim = Simulator::new(&scenarios::scenario("t", scenarios::room_clear_los(), scenarios::one_link(), tl, 1));
    assert_close(sim.truth_at(fast_start + 150.0).bpm, 28.0, "mid tachypnea");
    assert_close(sim.truth_at(fast_start - 1.0).bpm, 16.0, "before");
}

#[test]
fn body_on_top_of_a_node_stays_finite() {
    // Random seats can land on an antenna; the far-field formula is clamped.
    let layout = scenarios::three_links();
    let on_node = layout.nodes[0].pos;
    let timeline = vec![seg(5.0, Activity::Absent), seg(5.0, Activity::WalkTo { to: on_node, speed_mps: 1.0 }), seg(5.0, Activity::Fall { to: on_node, bpm: 15.0 })];
    let sc = scenarios::scenario("node", scenarios::room_rich_nlos(), layout, timeline, 2);
    let mut sim = Simulator::new(&sc);
    while let Some(tick) = sim.step() {
        for f in &tick.frames {
            assert!(f.frame.amplitudes.iter().chain(&f.frame.phases).all(|x| x.is_finite()), "t={}", tick.truth.t_s);
        }
    }
    for &l in &sc.links {
        assert!(sim.sensitivity(l).iter().all(|s| s.is_finite()));
    }
}

// ------------------------------------------------- explorer: manual body

/// Seated at `CHAIR` from t = 0, one link, no packet loss. Short, because
/// these tests only need a handful of ticks.
fn explorer_scenario(seed: u64) -> Scenario {
    quiet(scenarios::scenario("explorer", scenarios::room_clear_los(), scenarios::one_link(), vec![seated(20.0)], seed))
}

fn nth_tick(sim: &mut Simulator, n: usize) -> saans_sim::Tick {
    sim.nth(n - 1).expect("scenario is long enough")
}

#[test]
fn position_override_moves_the_body_and_clearing_it_returns_to_the_plan() {
    let sc = explorer_scenario(7);
    let plan_tick = nth_tick(&mut Simulator::new(&sc), 100);
    assert_close(plan_tick.truth.position.0, CHAIR.0, "plan x");
    assert_close(plan_tick.truth.position.1, CHAIR.1, "plan y");

    // Pinned somewhere else: the truth says so and the CSI changes.
    let mut moved = Simulator::new(&sc);
    moved.set_position_override(Some(DESK));
    assert_eq!(moved.position_override(), Some(DESK));
    let moved_tick = nth_tick(&mut moved, 100);
    assert_eq!(moved_tick.truth.position, DESK, "truth follows the override");
    assert!(
        moved_tick.frames[0].frame.amplitudes.iter().zip(&plan_tick.frames[0].frame.amplitudes).any(|(a, b)| (a - b).abs() > 1e-9),
        "moving the body must change the channel"
    );

    // Cleared mid-run: back on the plan, and bit-identical to never having
    // touched it -- the override takes no RNG draws and no sequence numbers.
    let mut cleared = Simulator::new(&sc);
    cleared.set_position_override(Some(DESK));
    let _ = nth_tick(&mut cleared, 50);
    cleared.set_position_override(None);
    assert_eq!(cleared.position_override(), None);
    let after = nth_tick(&mut cleared, 50);
    assert_eq!(after.truth, plan_tick.truth, "truth is the plan's again");
    assert_eq!(after.frames, plan_tick.frames, "CSI is the plan's again, exactly");
}

#[test]
fn position_override_is_clamped_into_the_walkable_area() {
    let sc = explorer_scenario(7);
    let (w, h) = sc.room.size_m;
    let mut sim = Simulator::new(&sc);
    for p in [(1e6, -1e6), (-3.0, 400.0), (f64::NAN, 2.0)] {
        sim.set_position_override(Some(p));
        let got = sim.position_override().expect("still set");
        assert!(
            got.0 >= MARGIN_M - 1e-9 && got.0 <= w - MARGIN_M + 1e-9 && got.1 >= MARGIN_M - 1e-9 && got.1 <= h - MARGIN_M + 1e-9,
            "{p:?} clamped to {got:?}, outside the walkable area"
        );
    }
    sim.set_position_override(Some((1e6, -1e6)));
    assert_eq!(sim.position_override(), Some((w - MARGIN_M, MARGIN_M)));
}

#[test]
fn an_absent_body_leaves_exactly_the_empty_room() {
    let sc = explorer_scenario(5);
    assert!(Simulator::new(&sc).present(), "a new simulator has a body in it");

    let present = nth_tick(&mut Simulator::new(&sc), 60);
    let mut empty = Simulator::new(&sc);
    empty.set_present(false);
    assert!(!empty.present());
    let gone = nth_tick(&mut empty, 60);

    assert!(present.truth.present && present.truth.breathing);
    assert!(!gone.truth.present, "truth must say nobody is there");
    assert!(!gone.truth.breathing && !gone.truth.still && !gone.truth.in_pause && !gone.truth.in_burst);
    assert_eq!(gone.truth.bpm, 0.0);
    assert!(
        gone.frames[0].frame.amplitudes.iter().zip(&present.frames[0].frame.amplitudes).any(|(a, b)| (a - b).abs() > 1e-9),
        "removing the body must change the amplitudes"
    );

    // The stronger claim: it is the SAME CSI as a scenario in which nobody
    // ever entered -- static channel plus the same noise draws, with no
    // torso, arm or chest motion left over anywhere.
    let mut never = Simulator::new(&quiet(scenarios::scenario("empty", scenarios::room_clear_los(), scenarios::one_link(), scenarios::empty(0.1), 5)));
    let unoccupied = nth_tick(&mut never, 60);
    assert_eq!(gone.frames, unoccupied.frames, "an emptied room must be indistinguishable from an empty one");

    // And the body comes back when asked.
    empty.set_present(true);
    let back = empty.step().expect("more ticks");
    assert!(back.truth.present && back.truth.breathing);
}

#[test]
fn sensitivity_at_the_current_position_is_sensitivity() {
    let sc = quiet(scenarios::scenario("sens", scenarios::room_rich_nlos(), scenarios::three_links(), vec![seated(20.0)], 3));
    let mut sim = Simulator::new(&sc);
    let tick = nth_tick(&mut sim, 40);
    for &l in &sc.links {
        // Same physics path, so bit-identical, not merely close.
        assert_eq!(sim.sensitivity(l), sim.sensitivity_at(l, tick.truth.position), "link {l:?} at the plan's position");
    }
    sim.set_position_override(Some(DESK));
    let _ = sim.step().expect("more ticks");
    for &l in &sc.links {
        assert_eq!(sim.sensitivity(l), sim.sensitivity_at(l, DESK), "link {l:?} under an override");
    }
    // It is a what-if map: taking the body away does not change it.
    let before: Vec<Vec<f64>> = sc.links.iter().map(|&l| sim.sensitivity(l)).collect();
    sim.set_present(false);
    let after: Vec<Vec<f64>> = sc.links.iter().map(|&l| sim.sensitivity(l)).collect();
    assert_eq!(before, after);
}

#[test]
fn coverage_has_the_advertised_shape_and_units() {
    let sc = explorer_scenario(7);
    let (room_w, room_h) = sc.room.size_m;
    for res_m in [0.25, 0.4, 1.0] {
        let cov = Simulator::new(&sc).coverage((0, 1), res_m);
        let (w, h) = ((room_w / res_m).ceil() as usize, (room_h / res_m).ceil() as usize);
        assert_eq!((cov.w, cov.h), (w, h), "res {res_m}");
        assert_eq!(cov.res_m, res_m);
        assert_eq!(cov.origin_m, (res_m / 2.0, res_m / 2.0));
        assert_eq!(cov.values.len(), w * h, "row-major, w*h values");
        assert!(cov.values.iter().all(|v| v.is_finite() && *v >= 0.0), "res {res_m}: non-finite or negative cell");
        assert!(cov.median_raw > 0.0 && cov.median_raw <= 1.0, "raw median {} is outside sensitivity's [0, 1]", cov.median_raw);
        // "1.0 = this room's median" is the advertised unit.
        let mut sorted = cov.values.clone();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        let median = if n % 2 == 1 { sorted[n / 2] } else { 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]) };
        assert!((median - 1.0).abs() < 1e-12, "res {res_m}: median is {median}, not 1.0");
    }
    // Coverage of the room, not of the walkable area: the first cell is
    // half a cell in from the corner.
    let cov = Simulator::new(&sc).coverage((0, 1), 0.25);
    assert_eq!(cov.origin_m, (0.125, 0.125));
    assert!(cov.origin_m.0 + (cov.w - 1) as f64 * cov.res_m <= room_w);
    assert!(cov.origin_m.1 + (cov.h - 1) as f64 * cov.res_m <= room_h);
}

#[test]
fn coverage_rejects_a_bad_resolution() {
    let sim = Simulator::new(&explorer_scenario(7));
    for bad in [0.0, -0.25, f64::NAN, f64::INFINITY] {
        assert!(sim.try_coverage((0, 1), bad).is_err(), "res_m {bad} must be rejected");
    }
    // 6 m x 5 m at 1 mm is 30 million cells.
    assert!(sim.try_coverage((0, 1), 0.001).is_err(), "an absurd grid must be rejected");
    assert!(sim.try_coverage((0, 1), 0.05).is_ok(), "a sane fine grid must be served");
}

#[test]
fn coverage_shows_fresnel_nulls_not_a_distance_falloff() {
    // A vertical cut at x = 3.0 m, from 0.4 m off the line of sight
    // (nodes are at y = 2.5) to the far wall, and more than 1.5 m from
    // either antenna. Along it every distance-only quantity -- range to
    // each node, total path, their product -- varies smoothly and
    // monotonically, so a smooth falloff model can cross a threshold at
    // most once. Interference does not: the body's echo walks through
    // the static channel's phase, nulling and peaking every half
    // wavelength (6.2 cm at 2.437 GHz).
    let res_m = 0.05;
    let (cut_x, y_lo, y_hi) = (3.0, 2.9, 4.6);
    let (lo, hi) = (0.5, 0.9);
    for seed in 0..8u64 {
        let sc = explorer_scenario(seed);
        let cov = Simulator::new(&sc).coverage((0, 1), res_m);
        let ix = ((cut_x - cov.origin_m.0) / cov.res_m).round() as usize;
        let cut: Vec<f64> = (0..cov.h)
            .map(|iy| (cov.origin_m.1 + iy as f64 * cov.res_m, cov.values[iy * cov.w + ix]))
            .filter(|(y, _)| (y_lo..=y_hi).contains(y))
            .map(|(_, v)| v)
            .collect();
        assert!(cut.len() > 30, "seed {seed}: the cut is too short to say anything");

        let (mn, mx) = (cut.iter().copied().fold(f64::INFINITY, f64::min), cut.iter().copied().fold(0.0, f64::max));
        assert!(mn <= 0.4, "seed {seed}: faintest cell on the cut is {mn:.3} of the median -- no null");
        assert!(mx >= 1.0, "seed {seed}: brightest cell on the cut is only {mx:.3} of the median");

        // Adjacent cells 5 cm apart differing several-fold is the thing a
        // distance falloff cannot do at all.
        let adjacent = cut.windows(2).map(|w| w[0].max(w[1]) / w[0].min(w[1]).max(1e-12)).fold(0.0, f64::max);
        assert!(adjacent >= 3.0, "seed {seed}: neighbouring cells differ by at most {adjacent:.1}x -- the map is smooth");

        // Count full swings between "dark" and "bright" along the cut.
        let (mut crossings, mut state) = (0, 0i32);
        for &v in &cut {
            let s = if v < lo { -1 } else if v > hi { 1 } else { 0 };
            if s != 0 && s != state {
                crossings += i32::from(state != 0);
                state = s;
            }
        }
        assert!(crossings >= 5, "seed {seed}: only {crossings} dark/bright swings along a {:.1} m cut", y_hi - y_lo);
    }
}
