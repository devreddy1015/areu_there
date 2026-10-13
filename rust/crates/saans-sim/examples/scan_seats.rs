//! Ground-truth breathing sensitivity at candidate seats across room seeds.
//! SYNTHETIC. `cargo run --release --example scan_seats [x y]...`
//! Prints, per room and layout, how often the seat is a whole-band blind
//! spot (max sensitivity over subcarriers < 0.2) on the first link and on
//! every link at once.
use saans_sim::scenario::{Activity, Point, Segment};
use saans_sim::scenarios::{self, BED, CHAIR, DESK};
use saans_sim::Simulator;

fn scan(seat: Point) -> Vec<(f64, f64, f64)> {
    let mut out = Vec::new();
    for room in [scenarios::room_clear_los(), scenarios::room_rich_nlos()] {
        for layout in [scenarios::one_link(), scenarios::three_links()] {
            let n = 200;
            let (mut blind0, mut blind_all, mut sum_best) = (0, 0, 0.0);
            for seed in 0..n {
                let tl = vec![Segment { duration_s: 1.0, activity: Activity::Seated { at: seat, bpm: 15.0, bpm_end: None } }];
                let sc = scenarios::scenario("scan", room.clone(), layout.clone(), tl, seed);
                let sim = Simulator::new(&sc);
                let per_link: Vec<f64> = sc.links.iter().map(|&l| sim.sensitivity(l).into_iter().fold(0.0, f64::max)).collect();
                let best = per_link.iter().cloned().fold(0.0, f64::max);
                blind0 += (per_link[0] < 0.2) as u32;
                blind_all += (best < 0.2) as u32;
                sum_best += best;
            }
            out.push((blind0 as f64 / n as f64, blind_all as f64 / n as f64, sum_best / n as f64));
        }
    }
    out
}

fn main() {
    let args: Vec<f64> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let seats: Vec<(String, Point)> = if args.len() >= 2 {
        args.as_chunks::<2>().0.iter().map(|&[x, y]| (format!("({x}, {y})"), (x, y))).collect()
    } else {
        vec![("CHAIR".into(), CHAIR), ("BED".into(), BED), ("DESK".into(), DESK)]
    };
    println!("seat            clear/1link   clear/3links  rich/1link    rich/3links   (link0 blind %, all-links blind %, mean best)");
    for (name, seat) in seats {
        let r = scan(seat);
        let cells: Vec<String> = r.iter().map(|(a, b, c)| format!("{:4.1}/{:4.1}/{:.2}", 100.0 * a, 100.0 * b, c)).collect();
        println!("{name:15} {}", cells.join("  "));
    }
}
