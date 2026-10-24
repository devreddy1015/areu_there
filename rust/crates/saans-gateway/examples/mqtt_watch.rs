//! A broker and a subscriber in one process, for watching what the
//! gateway publishes without installing mosquitto. Development only: no
//! authentication, no persistence, loopback.
//!
//! ```text
//! cargo run -p saans-gateway --example mqtt_watch -- 1883
//! ```
//!
//! then point a gateway's `[mqtt]` at `127.0.0.1:1883`. Prints one line per
//! message: local time, `R` if the broker delivered it as retained, topic,
//! payload.

use rumqttc::{AsyncClient, Event, MqttOptions, Packet, QoS};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Duration;

#[tokio::main]
async fn main() {
    let port: u16 = std::env::args()
        .nth(1)
        .map(|p| p.parse().expect("usage: mqtt_watch [port]"))
        .unwrap_or(1883);
    let addr: SocketAddr = ([127, 0, 0, 1], port).into();
    let server = rumqttd::ServerSettings {
        name: "v4".into(),
        listen: addr,
        tls: None,
        next_connection_delay_ms: 1,
        connections: rumqttd::ConnectionSettings {
            connection_timeout_ms: 60_000,
            max_payload_size: 1 << 20,
            max_inflight_count: 100,
            auth: None,
            external_auth: None,
            dynamic_filters: true,
        },
    };
    let config = rumqttd::Config {
        router: rumqttd::RouterConfig {
            max_connections: 100,
            max_outgoing_packet_count: 200,
            max_segment_size: 1 << 20,
            max_segment_count: 10,
            ..Default::default()
        },
        v4: Some(HashMap::from([("1".to_string(), server)])),
        ..Default::default()
    };
    std::thread::spawn(move || rumqttd::Broker::new(config).start().expect("broker"));
    eprintln!("mqtt_watch: broker on {addr}, printing every message");

    let mut o = MqttOptions::new("mqtt-watch", "127.0.0.1", port);
    o.set_max_packet_size(1 << 20, 1 << 20);
    let (client, mut el) = AsyncClient::new(o, 16);
    client.subscribe("#", QoS::AtLeastOnce).await.unwrap();
    loop {
        match el.poll().await {
            Ok(Event::Incoming(Packet::Publish(p))) => {
                let now = saans_gateway::timefmt::now_unix_ms();
                let time = saans_gateway::timefmt::iso8601(now, 0);
                println!(
                    "{} {} {} {}",
                    &time[11..23],
                    if p.retain { "R" } else { "-" },
                    p.topic,
                    String::from_utf8_lossy(&p.payload)
                );
            }
            Ok(_) => {}
            // The broker thread may not be listening yet.
            Err(_) => tokio::time::sleep(Duration::from_millis(200)).await,
        }
    }
}
