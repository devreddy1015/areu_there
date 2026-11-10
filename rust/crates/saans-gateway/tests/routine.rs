//! `/api/routine` (issue #39): the home's daily routine, served when some
//! space has a room, and only to the roles that see health alerts in every
//! room it reads. The day logic itself is unit-tested in
//! `saans-health/src/routine.rs`; this checks the gateway's side.

mod support;

use serde_json::{json, Value};
use support::*;

const HOME: &str =
    "[[space]]\nid = \"bedroom\"\nroom = \"bedroom\"\nlinks = [{ rx = 3, tx = 4 }]\n\
                    [[space]]\nid = \"hall\"\nlinks = [{ rx = 1, tx = 2 }]\n";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn without_a_room_there_is_no_routine() {
    let g = TestGw::start(config(
        "routine-none",
        "[[space]]\nid = \"hall\"\nlinks = [{ rx = 1, tx = 2 }]\n",
    ))
    .await;
    assert_eq!(g.get("/api/routine").await.status, 404);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_routine_is_served_to_the_roles_that_see_health_alerts() {
    let mut g = TestGw::start(config("routine", &format!("{}{HOME}", auth_toml()))).await;
    g.run(5.0, |i| {
        vec![occupied_frame(3, 4, i, None), empty_frame(1, 2, i)]
    })
    .await;
    for (role, want) in [
        ("admin", 200),
        ("safety", 200),
        ("ops", 403),
        ("occupancy", 403),
    ] {
        let r = get_with(
            g.http(),
            "/api/routine",
            &[("Authorization", &bearer(role))],
        )
        .await;
        assert_eq!(r.status, want, "{role}: {}", r.body);
    }
    let r = get_with(
        g.http(),
        "/api/routine",
        &[("Authorization", &bearer("admin"))],
    )
    .await;
    let v: Value = serde_json::from_str(&r.body).unwrap();
    assert_eq!(
        v["rooms"],
        json!([{ "space_id": "bedroom", "room": "bedroom" }])
    );
    assert!(v["today"]["observed_hours"].as_u64().unwrap() >= 1, "{v}");
    assert_eq!(v["today"]["date"].as_str().unwrap().len(), 10, "{v}");
    assert!(
        v["days"].as_array().unwrap().is_empty(),
        "no day has closed"
    );
    assert!(v["note"].as_str().unwrap().contains("diagnoses nothing"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unknown_value_is_null_on_the_wire_never_zero() {
    // Five seconds of an afternoon, with a bathroom whose link never
    // reports: no night and no morning has been seen through. This pins
    // how unknown goes out: null and `got_up_known: false`, never 0. When
    // a value becomes known is unit-tested in saans-health.
    let home = format!(
        "{HOME}[[space]]\nid = \"bath\"\nroom = \"bathroom\"\nlinks = [{{ rx = 5, tx = 6 }}]\n"
    );
    let mut g = TestGw::start(config("routine-unseen", &home)).await;
    g.run(5.0, |i| {
        vec![occupied_frame(3, 4, i, None), empty_frame(1, 2, i)]
    })
    .await;
    let v = g.json("/api/routine").await;
    assert_eq!(v["today"]["night_bathroom_visits"], Value::Null, "{v}");
    assert_eq!(v["today"]["got_up_min"], Value::Null, "{v}");
    assert_eq!(v["today"]["got_up_known"], json!(false), "{v}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn motion_in_a_room_measured_empty_is_not_an_active_hour() {
    // Gross motion well above `BOUT_INDEX` (index 10).
    let moving = |mut f: saans_dsp::features::FeatureFrame| {
        f.gross_motion = 0.5;
        f
    };
    let mut g = TestGw::start(config("routine-active", HOME)).await;
    // The hall's link reports it in a room it measures as empty (noise, or
    // a neighbour through the wall): that is not an active hour.
    g.run(2.0, |i| {
        vec![occupied_frame(3, 4, i, None), moving(empty_frame(1, 2, i))]
    })
    .await;
    let v = g.json("/api/routine").await;
    assert_eq!(v["today"]["active_hours"], json!(0), "{v}");
    // Someone moving in the bedroom is. The body is rebuilt once a minute.
    g.run(61.0, |i| {
        vec![
            moving(occupied_frame(3, 4, i + 10, None)),
            empty_frame(1, 2, i + 10),
        ]
    })
    .await;
    let v = g.json("/api/routine").await;
    assert_eq!(v["today"]["active_hours"], json!(1), "{v}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_safety_user_scoped_to_some_rooms_does_not_get_the_whole_home() {
    // Every space's motion feeds the routine's active hours, so seeing the
    // bedroom is not enough: the hall's motion would leak through it.
    let sha = saans_gateway::auth::sha256_hex("scoped-safety-token");
    let scoped = format!(
        "[[token]]\nname = \"scoped\"\nrole = \"safety\"\nspaces = [\"bedroom\"]\nsha256 = \"{sha}\"\n"
    );
    let mut g = TestGw::start(config(
        "routine-scoped",
        &format!("{}{scoped}{HOME}", auth_toml()),
    ))
    .await;
    g.run(2.0, |i| {
        vec![occupied_frame(3, 4, i, None), empty_frame(1, 2, i)]
    })
    .await;
    let r = get_with(
        g.http(),
        "/api/routine",
        &[("Authorization", "Bearer scoped-safety-token")],
    )
    .await;
    assert_eq!(r.status, 403, "{}", r.body);
}
