# Running the gateway as a service

For a gateway that stays up: started at boot, restarted if it crashes,
logging to the journal. Any Linux machine with systemd. It has run only
on an x86-64 laptop so far; a small always-on box (ARM included) should
work, but none has been tried.

| File | Installed as |
|---|---|
| `deploy/systemd/saans-gateway.service` | `/etc/systemd/system/saans-gateway.service` |
| `deploy/systemd/saans-gateway.env.example` | `/etc/saans/saans.env`, mode 0600, filled in |
| `rust/crates/saans-gateway/saans.example.toml` | `/etc/saans/saans.toml`, edited |
| `rust/target/release/saans-gateway` | `/usr/local/bin/saans-gateway` |

## Install

```bash
cd rust && cargo build --release -p saans-gateway
sudo install -m 0755 target/release/saans-gateway /usr/local/bin/
sudo install -d -m 0755 /etc/saans
sudo install -m 0644 crates/saans-gateway/saans.example.toml /etc/saans/saans.toml
sudo install -m 0600 ../deploy/systemd/saans-gateway.env.example /etc/saans/saans.env
sudo install -m 0644 ../deploy/systemd/saans-gateway.service /etc/systemd/system/
```

Then edit `/etc/saans/saans.toml`:

- `data_dir = "/var/lib/saans"`: the one directory the service can write.
- Your spaces and their links: which RX board watches which room.
- Anything else from the example you want: signed frames, MQTT, alerts.

Put the secrets it names into `/etc/saans/saans.env`, and only those.
Then:

```bash
sudo /usr/local/bin/saans-gateway --config /etc/saans/saans.toml --check
sudo systemctl daemon-reload
sudo systemctl enable --now saans-gateway
```

## Operate

```bash
systemctl status saans-gateway
journalctl -u saans-gateway -f          # the gateway's log lines, and why it stopped
sudo systemctl restart saans-gateway    # after editing saans.toml or saans.env
```

- **The dashboard** is at `http://127.0.0.1:8080/app` on the gateway
  machine itself. To open it from elsewhere, set `http_addr` to a network
  address. The gateway then requires sign-in, and refuses to start until
  a `[[user]]` or `[[token]]` exists (README, "Who may see what").
- **The sensors** send to UDP 5566. Allow that port from the sensor
  network only. Frames are not authenticated unless `[[frame_key]]` is
  set (README, "Operating requirements").
- **Stopping** saves history, replay counters and the event log, and MQTT
  publishes `offline`. Allow it the 30 s the unit gives it.
- **If it will not start**, the journal says why, naming the variable or
  setting, never a secret's value. After five failed starts in two
  minutes it stays stopped, so a broken config does not loop. Fix it,
  then run `sudo systemctl reset-failed saans-gateway` and start it.

## What the service is allowed to do

It runs as a user made up at each start (`DynamicUser=yes`) that owns
only `/var/lib/saans`. It cannot write anywhere else, cannot see home
directories, has no capabilities, and can make only the system calls an
ordinary network service needs. `systemd-analyze security` rates the
unit's exposure 1.2 on its 0-10 scale (lower is safer).

`/var/lib/saans` holds event logs, the audit log, `history.json` and
`frame_counters.json`: health data under ADR-001 and ADR-005. Back it up
as such, and never into this repository.

## Update

```bash
cd rust && git pull && cargo build --release -p saans-gateway
sudo install -m 0755 target/release/saans-gateway /usr/local/bin/
sudo systemctl restart saans-gateway
```

## Remove

```bash
sudo systemctl disable --now saans-gateway
sudo rm /etc/systemd/system/saans-gateway.service /usr/local/bin/saans-gateway
sudo systemctl daemon-reload
```

`/etc/saans` and `/var/lib/saans` (in fact `/var/lib/private/saans`,
because of `DynamicUser`) stay until you delete them.

## How this was checked

On 2026-10-24, without installing it system-wide:

- `systemd-analyze verify` on the unit: clean.
- `systemd-analyze security --offline=yes`: exposure 1.2, "OK".
- **The same unit run as a temporary user service.** Its own user, the
  state and config directories, and `ProtectHome=yes` were left out,
  because those need root. Every other restriction was kept, including
  the system-call filter. Under it the gateway:
  - started, answered `/api/health`, and ingested FeatureFrames over UDP;
  - restarted 5 s after a `SIGKILL`;
  - stopped cleanly on `systemctl stop` (history saved, exit 0);
  - with a key variable missing, failed five times naming the variable,
    then stopped with `start-limit-hit`.

Not checked: the unit installed as a system service with its own user,
which needs root on a machine you choose.
