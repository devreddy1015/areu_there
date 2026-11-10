# Saans relay protocol, v1 (draft)

**Status:** draft, 2026-11-06. **Nothing implements it yet.** It
specifies the decisions in ADR-016 (the relay,
[PR #54](https://github.com/devreddy1015/areu_there/pull/54)) and
ADR-017 (the phone app,
[PR #55](https://github.com/devreddy1015/areu_there/pull/55)), and it is
the contract the stub relay for issue #40 and both apps will build
against. Until the first implementation merges, it can change freely.
After that, any change to bytes or meaning needs a new version (§9).

Each value marked **[proposal]** is a starting number. The ADRs or the
spec in PR #52 own it.

## 0. Parties and keys

| Party | Holds | Keys |
|---|---|---|
| **Gateway** (G), one per home | the alerts and every authorisation decision | `G.kem`: an X25519 key pair, for sealing to G. `G.sig`: an Ed25519 key pair, for signing. The private halves are read from environment variables at the moment of use (CONTRIBUTING, "Secrets") |
| **Relay** (R) | routing data, sealed envelopes, delivery receipts and the escalation clock | TLS only. It holds no key that opens an envelope |
| **Device** (D), an enrolled phone | one family member's view of one home | `D.kem` (X25519) and `D.sig` (Ed25519), created on the phone and kept in the Android Keystore or the iOS Keychain. The private halves never leave it |

**Identifiers.** Each is 16 random bytes, written as unpadded base64url.

- `home_id`, `device_id` and `contact_id` are fixed at enrolment.
- `msg_id` is new for every envelope.
- `alert_ref` is new for every alert. It is random, not the gateway's
  alert number, so that the relay cannot count a home's alerts from the
  numbering.

**Time.** In clear headers, RFC 3339 in UTC (`2026-11-06T09:12:00Z`).
Inside the seal, RFC 3339 with the home's offset
(`2026-11-06T14:42:00+05:30`), because the phone shows it.

## 1. The seal

**HPKE (RFC 9180), base mode,** with one suite:

| | Id | Name |
|---|---|---|
| KEM | `0x0020` | DHKEM(X25519, HKDF-SHA256) |
| KDF | `0x0001` | HKDF-SHA256 |
| AEAD | `0x0003` | ChaCha20-Poly1305 |

Both phone libraries ship this suite:

- CryptoKit as `HPKE.Ciphersuite.Curve25519_SHA256_ChachaPoly`
  ([Apple](https://developer.apple.com/tutorials/data/documentation/cryptokit/hpke/ciphersuite.json));
- Tink as `DHKEM_X25519_HKDF_SHA256_HKDF_SHA256_CHACHA20_POLY1305`
  ([Tink](https://developers.google.com/tink/hybrid)).

An envelope is sealed to **one** recipient. A message for three phones
is three envelopes.

**The envelope**, as JSON:

```json
{
  "hdr": "<base64url of the header's exact JSON bytes>",
  "enc": "<base64url of the 32-byte HPKE encapsulated key>",
  "ct":  "<base64url of the HPKE ciphertext>"
}
```

**The header** is clear. The relay reads it, and nothing outside it:

```json
{"v":1,"home":"<home_id>","msg":"<msg_id>","to":"<device_id or \"gateway\">",
 "class":"urgent","alert":"<alert_ref>","at":"2026-11-06T09:12:00Z","exp":"2026-11-07T09:12:00Z"}
```

- **`alert`** is present only on `urgent`, `heads_up`, `update` and
  `ack`.
  - It is the alert's random `alert_ref`, and the relay needs it in the
    clear: escalation, its repeats, receipts and the stop on an
    acknowledgement are all kept per alert.
  - It tells the relay which messages belong together, and nothing about
    what they say.
  - The acknowledgement's action ("checking", "all fine", "false alarm")
    stays inside the seal.
- **`class`** is one of the following:
  - `urgent`, `heads_up`, `update`, `home_now`, `summary` and `test`,
    which go from gateway to device;
  - `ack`, `home_now_request` and `enrol`, which go from device to
    gateway.
- **`exp`** is when the relay may drop an envelope that has not been
  collected: 24 h after `at` [proposal, ADR-016].

**Sealing.**

```text
info      = "saans relay v1"                       (ASCII)
aad       = hdr_bytes                              (the exact bytes base64url-encoded in "hdr")
body      = payload JSON bytes                     (§3)
signature = Ed25519(sender.sig, "saans relay v1 sig" || hdr_bytes || body)
plaintext = signature (64 bytes) || body
enc, ct   = HPKE.Seal(recipient.kem public key, info, aad, plaintext)
```

**Opening, in this order. Refuse at the first failure:**

1. `v` is 1, `to` names this recipient, and `exp` has not passed.
2. HPKE.Open with `aad` = `hdr_bytes`. A change to any byte of the
   header, including `class` and `to`, fails here.
3. Split the plaintext into the 64-byte signature and the body. Verify
   the signature against the sender's `sig` public key. That key is
   `G.sig` for anything addressed to a device, and the enrolled
   `D.sig` for anything addressed to the gateway.
4. Only now parse the body.
5. Check `seq` (§3). A gap is reported to the user. A repeat or a
   smaller value is refused as a replay.

**Why the signature sits inside the seal.** The seal alone shows only
that someone held the recipient's public key, and anyone can hold that.
The signature shows the gateway wrote the message. Putting it inside
hides even that much from the relay.

**Size.** FCM data messages and APNs payloads are both limited to 4,096
bytes
([FCM](https://firebase.google.com/docs/cloud-messaging/customize-messages/set-message-type),
[APNs](https://developer.apple.com/documentation/usernotifications/generating-a-remote-notification)).

- An envelope travels inside a push only if its JSON is at most
  **3,000 bytes**.
- A larger one stays in the device's mailbox (§2.2), and the push
  carries only the header fields.
- Alerts and updates fit. Summaries are never pushed.

## 2. Transport

HTTPS only, TLS 1.3, JSON bodies. Every path begins `/v1`.

**Request signing.** Every request from a gateway or a device carries
two headers:

```text
Saans-Date: 2026-11-06T09:12:00Z
Saans-Signature: <key id>:<base64url Ed25519 signature over
                  "saans relay v1 req" || METHOD || " " || PATH_AND_QUERY || "\n" || Saans-Date || "\n" || SHA-256(body)>
```

- `<key id>` is `g:<home_id>` or `d:<device_id>`.
- The relay refuses a date more than 300 s from its own clock. It also
  refuses a `msg` it already holds, which makes a resend harmless.
- There is no bearer token, so there is no shared secret for a log to
  leak. ADR-012 makes the same choice for frames, and CONTRIBUTING,
  "Secrets", explains why.

### 2.1 Gateway ↔ relay

| Request | Purpose |
|---|---|
| `POST /v1/homes/{home}/envelopes` | A batch of envelopes from G, for devices. Each one is stored, then pushed or held according to its class (§4). The response lists the accepted `msg` ids |
| `GET /v1/homes/{home}/inbox?after={cursor}&wait=25` | A long poll for envelopes addressed to G, plus receipts and delivery events (§5). **Each poll is the home's heartbeat.** The relay holds the request up to `wait` seconds, which is the gateway's Telegram hold (`POLL_TIMEOUT_S` = 25). Passing `after` acknowledges everything up to that cursor, as Telegram's `offset` does |
| `PUT /v1/homes/{home}/alerts/{alert_ref}` | `{"state":"acknowledged" \| "resolved"}`: stops the relay's escalation for that alert. G sends it when an alert is handled anywhere, the dashboard included |
| `PUT /v1/homes/{home}/devices/{device}` | Registers or updates a device: `{"sig_pub":"…","push":{"platform":"android" \| "ios","token":"…"},"primary":true}`. Only the gateway can register a device |
| `DELETE /v1/homes/{home}/devices/{device}` | Removes a device. The relay deletes its envelopes at once |
| `PUT /v1/homes/{home}/plan` | The escalation plan and the fallback contacts (§4.2) |
| `DELETE /v1/homes/{home}` | Withdraws the home. The relay erases everything it holds for it, its log included (DPDP s.8(7); ADR-016) |

### 2.2 Device ↔ relay

| Request | Purpose |
|---|---|
| `GET /v1/devices/{device}/mailbox?after={cursor}&wait=0..25` | Envelopes for this device. The app calls it whenever a push arrives, whenever it is opened, and while it waits for a `home_now` answer |
| `POST /v1/devices/{device}/receipts` | `{"msg":"…","at":"…","opened":true}`: the phone has the envelope, and `opened` says whether it decrypted. This is the receipt that stops the 2-minute SMS (§4.1) |
| `POST /v1/homes/{home}/outbox` | An envelope addressed to the gateway: `ack`, `home_now_request` |
| `PUT /v1/devices/{device}/push` | A new push token after the platform rotates it |
| `GET /v1/devices/{device}/status` | `{"relay_time":"…","home_last_heard":"…","home_silent":false}`. The app shows "Saans cannot reach its service" when this fails, and "No contact with the home since …" when `home_silent` is true |

## 3. Payloads (inside the seal)

Every payload has `"t"` (its type) and `"seq"`.

- **`seq`** is a u64 counter per (sender, recipient) pair, starting at 1.
  A gap shows that a message is missing, and is shown to the user as "a
  message from home may be missing".

**Gateway to device.**

| `t` (header `class`) | Fields |
|---|---|
| `alert` (`urgent` or `heads_up`) | `alert_ref`, `level`, `kind` (an event type such as `"long_lie"` or `"space_blind"`), `room` (the family's label), `opened_at`, `state` (`"open"`), `step` (escalation step at send) |
| `update` | `alert_ref`, `state` (`"acknowledged"`, `"resolved"` or `"open"`), `action` (`"checking"`, `"all_fine"` or `"false_alarm"`; absent when the gateway changed it), `by` (a device label or `"home"`), `at`, `note` (optional; for example `"movement seen 06:31"`) |
| `home_now` | `request`, `status` (`"home"`, `"away"`, `"not_seen"` or `"unknown"`), `last_room`, `last_movement_at`, `rooms` (a list of `{"room", "sensing": "ok" \| "blind" \| "link_down" \| "calibration_stale"}`), `as_of` |
| `summary` | `date`, then the items #39 defines. Sent only if the monitored person has allowed it |
| `test` | `sent_at`. It is shown as a test, end to end (ADR-017, Decision 6) |

**Never in any payload:**

- evidence sentences, confidences, breathing or heart numbers (ADR-016,
  Decision 4);
- raw CSI, FeatureFrames, history series, or the event or audit logs.

The phone writes the words a person reads, from `kind` and `room`. So a
change of wording needs no change to the protocol.

**Device to gateway.**

| `t` (header `class`) | Fields |
|---|---|
| `ack` | `alert_ref`, `action` (`"checking"`, `"all_fine"` or `"false_alarm"`), `at`. The gateway records it as acknowledged (`checking`) or resolved (the other two), by the device's label, and writes it to the audit log as a dashboard acknowledgement would be |
| `home_now_request` | `request` (a random id), `at`. The gateway answers with `home_now` to this device only, and writes each request to the log the monitored person can read (spec §4.2 in PR #52) |
| `enrol` | `code`, `label`, `kem_pub`, `sig_pub`, `platform`, `push_token` (§6) |

## 4. What the relay does with each class

| Class | Push | Escalation | Kept until |
|---|---|---|---|
| `urgent` | high priority (§7), repeated every 120 s [proposal] until the alert is acknowledged | yes (§4.1) | acknowledged, or `exp` |
| `heads_up` | high priority, once. The phone holds it through the person's quiet hours | no | collected, or `exp` |
| `update` | high priority. Updates are rare, and the relay cannot see what one says | no. Escalation stops when an `ack` envelope arrives from a device for that `alert`, or when the gateway sends `PUT …/alerts/{alert_ref}` | collected, or `exp` |
| `ack` (to the gateway) | none: it waits in the gateway's inbox | it stops escalation for its `alert` | collected by the gateway, or `exp` |
| `home_now`, `summary` | none: mailbox only | no | collected, or `exp` |
| `test` | as `urgent`, but its escalation goes no further than the 2-minute SMS check, which is reported rather than sent | no | collected, or `exp` |

### 4.1 Escalation of an urgent alert

The plan comes from `PUT …/plan`. The relay runs the clock, so the plan
carries on if the home goes dark after raising the alert (ADR-016,
Decision 6). The default plan [proposal, after spec §4.4 in PR #52]:

| After | Action |
|---|---|
| 0 s | push to the primary devices, then repeat |
| 120 s | an SMS to each primary contact whose devices have sent no receipt for this alert |
| 300 s | push to every device, and an SMS to every family contact |
| 600 s | a voice call to each nearby contact. Pressing 1 acknowledges the alert as `checking`, by `"<contact label> (call)"` |
| afterwards | the alert stays open, and pushes repeat to every device until it is acknowledged |

**The text outside the seal is fixed and generic** (ADR-016). The SMS
and call text is:

```text
Saans: urgent alert from <home label>. Open the Saans app or call home.
```

`<home label>` is the family's choice of name. It is stored at the relay
in the clear because the SMS needs it, and it may be left blank, in
which case the text says "your Saans home".

The relay tells the gateway about every step it takes, as a delivery
event in its inbox (§5).

### 4.2 The plan

```json
{"steps":[{"after_s":0,"push":"primary","repeat_s":120},
          {"after_s":120,"sms":"primary_without_receipt"},
          {"after_s":300,"push":"all","sms":"family"},
          {"after_s":600,"call":"nearby"}],
 "contacts":[{"contact":"<contact_id>","role":"family" | "nearby","phone":"+91…","label":"…","confirmed_at":"…"}],
 "home_label":"…","silence_after_s":600}
```

- **A contact is used only once `confirmed_at` is set.** That is the time
  the person confirmed, by answering a test call or SMS, that they agree
  to be called (ADR-016, "DPDP Act 2023").
- **Phone numbers are personal data.** They are in the clear at the
  relay because SMS and calls need them. They are deleted with the home.

## 5. Silence, receipts and delivery events

- **Silence.** If no poll arrives for `silence_after_s` (600 s
  [proposal]), the relay sends a `no_contact` push to every device. It
  also sends an SMS to every family contact, because a phone may have
  stopped the app.
  - The relay writes this message itself. It is not sealed, and it
    carries only `{"class":"no_contact","since":"…"}`.
  - The relay holds no gateway key, so it cannot write a sealed message,
    and it knows nothing about the rooms to put in one.
  - When polls resume, a `contact_restored` push follows.
- **Delivery events.** Through the inbox, the gateway learns of each
  receipt, push sent, SMS sent, call outcome and expiry:
  `{"t":"delivery","msg"|"alert_ref":"…","what":"receipt" | "push" | "sms" | "call" | "expired","detail":"…","at":"…"}`.
  The gateway records them in the alert's `deliveries`, as it does for
  Telegram today (`alerts.rs`, `Delivery`).
- **Gaps and forgeries.**
  - A relay that drops a message shows up as a `seq` gap at the phone.
  - A relay that changes a header fails HPKE.Open.
  - A relay that invents a message fails the signature.
  - A relay that delays a message shows up in `at`, which the phone
    displays.

## 6. Enrolment

1. **The QR code.** On the local network, the gateway's dashboard shows a
   QR code containing:

   ```json
   {"v":1,"relay":"https://…","home":"<home_id>","g_kem":"…","g_sig":"…","code":"<16 random bytes>","exp":"…"}
   ```

   It expires after 10 minutes [proposal]. The QR code never leaves the
   home except through the phone's camera.
2. **The request.** The phone creates `D.kem` and `D.sig`. It sends an
   `enrol` envelope, sealed to `g_kem`, through
   `POST /v1/homes/{home}/outbox`. The relay cannot read the code
   inside.
3. **The approval.** The gateway checks the code: once only, and before
   it expires. It then shows "<label>'s phone asks to receive alerts.
   Allow?" on the local dashboard, for the monitored person to approve
   (ADR-016).
4. **The registration.** On approval, the gateway registers the device
   with `PUT …/devices/{device}` and sends a `test` envelope. Setup on
   the phone ends when that test arrives (ADR-017, Decision 6).
5. **Removal** happens at the gateway, by `DELETE …/devices/{device}`.
   The phone learns of it from its next status call, and erases its
   keys.

## 7. Push formats

**FCM HTTP v1, data message** (Android). All `data` values are strings:

```json
{"message":{"token":"…","android":{"priority":"HIGH","ttl":"3600s"},
 "data":{"v":"1","home":"…","msg":"…","class":"urgent","env":"<envelope JSON, if ≤ 3,000 bytes>"}}}
```

- **A data message, not a notification message,** so that the app
  decrypts before anything is shown (ADR-017).
- **The app must post a visible notification for every `urgent` and
  `heads_up` message.** A high-priority message that shows nothing may
  be deprioritised (ADR-017, Context).
- **TTL:** 3,600 s for `urgent` [proposal], so that a phone switched
  back on hours later does not ring for a stale alert. 86,400 s for the
  rest.
- **No `collapse_key`** on `urgent`.
- **FCM's newer `RemoteNotification` type** is in preview. It is to be
  revisited when it is generally available.

**APNs** (iOS). Sent with `apns-push-type: alert`, `apns-priority: 10`,
and `apns-expiration` matching `exp`:

```json
{"aps":{"alert":{"title":"Saans","body":"Urgent alert. Open the app."},
        "mutable-content":1,"interruption-level":"time-sensitive","relevance-score":1,
        "sound":"default"},
 "saans":{"v":1,"home":"…","msg":"…","class":"urgent","env":"…"}}
```

- **The generic `alert` text** is what iOS shows if the Notification
  Service Extension cannot decrypt in time (ADR-016, Decision 3).
- **Once Apple grants the critical-alert entitlement,**
  `interruption-level` becomes `"critical"` and `sound` becomes
  `{"critical":1,"name":"default","volume":1.0}`
  ([APNs payload keys](https://developer.apple.com/documentation/usernotifications/generating-a-remote-notification)).
- **`heads_up`** uses `"active"`.

## 8. Retention at the relay

| What | Kept |
|---|---|
| Envelopes | until collected and, for `urgent`, acknowledged, or until `exp`, whichever comes first |
| Delivery log (receipts, pushes, SMS, calls) | 30 days [proposal], then reduced to counts per home per day |
| Devices, plan, contacts, home label | until removed at the gateway or the home withdraws |
| Last poll time | one value per home, overwritten |

The relay keeps nothing else, and does not log request bodies.

## 9. Versions

- `v` in the header, the QR code and the push payload, together with
  `/v1` in every path, is the protocol version.
- A change to bytes or meaning makes it 2. A relay serves both versions
  while any enrolled device still speaks 1.

## 10. Test vectors

The repository's golden-vector rule (`spec/wire.md` §3) applies, with
one difference: **HPKE sealing is randomised.** Tink and CryptoKit do
not let a caller fix the ephemeral key. So:

- **The primitive** is checked against RFC 9180's published test vectors
  for suite (0x0020, 0x0001, 0x0003), in base mode, by every
  implementation that can be given a fixed ephemeral key. The gateway's
  can.
- **The envelope** vectors (`spec/golden/relay_envelope*`) fix the keys,
  the header bytes and the payload, and give `enc` and `ct`. They are
  generated by the gateway, because the Python reference has no X25519
  in its standard library (ADR-017). Every implementation must:
  - **open** each vector to exactly the given body;
  - **refuse** it when any byte of `hdr`, `enc`, `ct` or the signature
    is flipped;
  - **round-trip** its own seal through another implementation's open.
- **Request signing** vectors fix a key, a date, a method, a path and a
  body, and give the exact signature.
