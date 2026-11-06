# ADR-016: The family alert path: an outbound-only relay that carries sealed alerts

**Status:** accepted, 2026-11-06. **Decided, not built.** Nothing in the
code changes with this ADR. The stub relay and its end-to-end test,
which is the second half of issue #40, are owed, and wait for the
consolidation (#32) to merge. The phone app this relies on is decided
in ADR-017 (#41). The elder-care spec and ADR-015 that this ADR
answers are in
[PR #52](https://github.com/devreddy1015/areu_there/pull/52), not merged
when this was written.

## Context

### What exists

The gateway already has an alert service (`saans-gateway/src/alerts.rs`,
`SAANS_PLAN.md` §10.3):

- **Routing:** `[[alert_route]]`s match events to channels. One alert is
  open per (space, kind); repeats update it instead of paging again. The
  states are open, acknowledged and resolved, kept in `alerts.jsonl`.
- **Escalation:** once, after `ack_timeout_min`, to `escalate_to`.
- **Channels:**
  - Telegram: `sendMessage` with Acknowledge and Resolve buttons, read
    back by long-polling `getUpdates`, so there is no inbound port
    (`poll_telegram`).
  - A webhook, signed with HMAC-SHA256.
- **The privacy boundary is one function, `outbound`.** By default a
  message carries the alert's kind, space and state, and nothing else.
  Evidence, confidence and numbers go out only if the route sets
  `include_health_details`.
- **Delivery is attempted once.** A failure is logged and counted, and
  not retried.
- **None of it has met a real Telegram bot or paging system.** The tests
  stand in local servers for both (`docs/EVIDENCE.md` §2.6).

### What elder care asks of it

ADR-015 makes the first customer an older person living alone, with
alerts going to family. This is what the spec in PR #52 asks for. Every
timing in it is marked there as a proposal.

- **Reach:** the gateway sits behind home NAT with no public address, and
  the family is anywhere.
- **Three levels:**
  - *urgent* (`Check`, e.g. a long lie): repeated until someone
    acknowledges it, and never held for quiet hours;
  - *heads-up* (`Advisory`);
  - *summary* (`Info`): shown in the app once a day, never a
    notification.
- **Acknowledgement:** "I'm checking", "All fine" and "False alarm", with
  every contact seeing who is acting.
- **Escalation:** primary contacts at 0 min, every family contact at
  5 min, and at 10 min a nearby contact by phone call. Saans never calls
  emergency services.
- **Silence:** "No contact with the home since 14:05", after 10 min.
- **On request:** a "home now" answer, and the daily summary if the
  monitored person allows it.

ADR-015 adds two constraints. ADR-011 stands, so a relay that stores
summaries for many homes would be a version of the control plane that
ADR-011 rules out. And end-to-end encryption from the gateway to the
family's phones "is the option to beat".

### Three facts that shape every option

1. **No platform confirms that a push reached a phone, let alone a
   person.**
   - Firebase says a message id "doesn't mean that the message was
     already delivered to the device. Instead, it means that it was
     accepted for delivery"
     ([FCM](https://firebase.google.com/docs/cloud-messaging/customize-messages/setting-message-lifespan)).
   - Apple: "As a best-effort service, APNs may reorder notifications",
     and "in some cases, not delivered"
     ([APNs](https://developer.apple.com/documentation/usernotifications/sending-notification-requests-to-apns)).
   - Telegram's `sendMessage` returns the message "On success", meaning
     accepted. No update type reports delivery or reading
     ([Bot API](https://core.telegram.org/bots/api)).
   - So an urgent alert counts as delivered only when the app says the
     phone has it, and as handled only when a person acknowledges it.
2. **Indian Android phones stop background apps.**
   - dontkillmyapp ranks Xiaomi second, OnePlus third and Samsung
     fourth among the worst offenders ([dontkillmyapp](https://dontkillmyapp.com/)).
   - Xiaomi: "In default settings, background processing simply does
     not work right" ([Xiaomi](https://dontkillmyapp.com/xiaomi)).
   - OnePlus, of one setting: "This setting will prevent push
     notifications from being delivered"
     ([OnePlus](https://dontkillmyapp.com/oneplus)).
   - Samsung's exemptions can be undone: "Samsung may re-add them later
     after a firmware update" ([Samsung](https://dontkillmyapp.com/samsung)).
   - So push cannot be the only path for an urgent alert.
3. **A silent home cannot report itself, and no messaging channel
   notices a message that never came.** Telegram, WhatsApp and SMS act
   only when they are sent something. Whatever carries the alerts,
   something always on, outside the home, has to watch for the home's
   heartbeat and say when it stops. That something is a relay. **So
   every option below has a relay.** What differs is what the relay
   sends to.

## The options

| | Telegram bot (built) | WhatsApp Business Platform | SMS, and voice calls | Saans app over FCM / APNs |
|---|---|---|---|---|
| Can be loud | as loud as the family's Telegram settings; `disable_notification` only makes a message quieter | an ordinary WhatsApp notification | a text, or a ringing phone | its own channel; Do Not Disturb bypass if the user grants it (Android), critical alerts if Apple grants the entitlement (iOS); ADR-017 |
| Sender learns of delivery | no | yes: sent, delivered, read and failed webhooks | yes: SMS delivery reports; call status and "press 1" | from the app itself, not from FCM or APNs |
| Acknowledgement back | inline buttons; updates wait at most 24 h | quick-reply buttons | a keypress on a call; SMS replies not checked | in the app |
| Who can read it | Telegram | Meta: "Cloud API decrypts the message and forwards it to the business" | the SMS or voice provider and the telecom operators | only the phones, if sealed (Decision 3) |
| Onboarding | every family member needs Telegram | Meta business portfolio, approved templates, a 250-recipient daily cap until verified | DLT registration of the entity, the header and every template; A2P calls declared in advance | install an app (ADR-017) |
| Cost per message | free | India utility template ₹0.115 | ₹0.25 an SMS; ₹0.38 a minute for a call | free |

The sources for that table:

- **Telegram.** The Bot API is "free for both users and developers"
  ([bots](https://core.telegram.org/bots)). Updates "will not be kept
  longer than 24 hours" ([Bot API](https://core.telegram.org/bots/api)),
  so a gateway that is offline for a day loses the button presses made
  in that time.
- **WhatsApp.**
  - It has charged per message since July 2025 ([pricing](https://developers.facebook.com/docs/whatsapp/pricing)).
  - A utility template to an Indian number costs ₹0.115 under the rate
    card "effective October 1, 2026"
    ([rate cards](https://developers.facebook.com/documentation/business-messaging/whatsapp/pricing#rate-cards-and-volume-tiers)).
  - Utility messages sent inside a customer-service window stopped being
    free on the same date.
  - Outside a window, only approved templates can be sent
    ([templates](https://developers.facebook.com/documentation/business-messaging/whatsapp/templates/overview)).
  - Meta "acts as a data processor" and keeps messages for at most 30
    days ([data privacy](https://developers.facebook.com/documentation/business-messaging/whatsapp/data-privacy-and-security)).
- **SMS.**
  - TRAI requires the sender to register as a principal entity, then its
    header and its content templates, before it may send
    ([TRAI](https://trai.gov.in/advice-to-senders)).
  - A transactional SMS is only one "in response to Customer initiated
    transaction within thirty minutes"
    ([TCCCPR Second Amendment, 2025](https://www.trai.gov.in/sites/default/files/2025-02/Regulation_12022025.pdf)).
  - The price is ₹0.25 an SMS in the smallest pack, plus GST
    ([MSG91](https://msg91.com/in/pricing)).
- **Voice.**
  - The TCCCPR (Third Amendment) Regulations, 18 September 2026, require
    every sender to "declare to the Originating Access Provider (OAP), in
    advance, about the use of Application-to-Person (A2P) calls".
    Undeclared calls are "treated as Unsolicited commercial
    communication"
    ([TRAI](https://www.trai.gov.in/sites/default/files/2026-09/Regulation_18092026.pdf)).
  - A call costs ₹0.38 a minute in 30-second pulses
    ([Plivo](https://www.plivo.com/voice/pricing/in/)). That route needs
    a business registered in India; the international route costs $0.0496
    a minute ([Twilio](https://www.twilio.com/en-us/voice/pricing/in)) and
    shows a foreign caller id.
- **Push.** FCM costs nothing: "There's no cost to using Cloud
  Messaging" ([FCM](https://firebase.google.com/products/cloud-messaging)).

**Telegram** is built, free and has buttons. But it shows the family's
alerts to Telegram, tells the gateway nothing about delivery, and asks
every relative to use an app many of them do not. WhatsApp is the
Indian default: Reuters reports over 500 million users in India,
against more than 150 million for Telegram
([Reuters via The Star, June 2026](https://www.thestar.com.my/tech/tech-news/2026/06/23/india-039proactively-monitoring039-telegram-over-concerns-about-illegal-content-government-report-shows)).

**WhatsApp** is where families already are, and it reports delivery.
But:

- the content is decrypted by Meta, so it cannot be sealed from the
  gateway;
- every alert text is a template that Meta approves and can recategorise;
- an urgent alert looks like any other WhatsApp message;
- it can carry quick replies, but cannot be loud.

**SMS and voice** reach any phone, with no app and no data connection,
and report delivery. They are the one path that survives an app the
phone has stopped. But:

- the content is plain to the provider and the operators;
- every template and caller id is registered with TRAI's DLT system in
  advance;
- whether an alert about a parent counts as a "service" message to a
  "Customer" under TRAI's definition is a legal question, not answered
  here [unverified].

**A Saans app** is the only option in which the content can be sealed
from the gateway to the phone, the alert can be made loud on the phone's
own terms, and delivery and acknowledgement are under Saans's control.
It is also the only option that needs something installed. ADR-017
decides how.

## Decision

1. **Saans runs a relay, and the gateway only ever connects out to it.**
   - One HTTPS connection carries everything both ways, long-polled.
     This is the pattern the Telegram channel already uses
     (`alerts.rs`, `poll_telegram`).
   - The gateway never listens, and needs no port opened.
   - Every poll is the heartbeat. There is no separate heartbeat message
     with contents of its own.
   - The gateway authenticates with a key whose private half is in an
     environment variable, as the frame keys are (ADR-012).
   - It is hosted in an Indian region [proposal]. DigitalOcean Bangalore
     and AWS Lightsail Mumbai both qualify.

2. **The relay is a mailbox and a knocker. It is not a control plane.**
   - It holds, per home:
     - a random home id;
     - the gateway's public key;
     - the enrolled phones (each with a label, a public key and a push
       token);
     - the fallback phone numbers;
     - the escalation plan;
     - the time it last heard from the home.
   - It holds, per message, the sealed envelope, until it is acknowledged
     or 24 h have passed [proposal].
   - It keeps a delivery log for 30 days [proposal], the event log's
     default clock.
   - It holds nothing else: no room state, no history, no summaries in
     the clear.
   - It cannot read a gateway's data, change its configuration or raise
     an alert in its name.

3. **Alerts are sealed end to end.**
   - **Sealing.** The gateway seals every payload separately to each
     enrolled phone's public key, using an ephemeral-key public-key box:
     X25519 with an AEAD, as in libsodium's sealed box or HPKE (RFC
     9180). It also signs the payload with its own key. Phones sign
     their acknowledgements with theirs.
   - **What the relay can see:** the home id, the recipient phones, the
     message class (urgent, heads-up, summary or reply), sizes, times,
     and which phone acknowledged.
   - **What it cannot see:** the kind of alert, the room, the text or
     any number.
   - **Why both platforms allow it.** FCM "doesn't provide a built-in
     solution for end-to-end encryption", but it "works with FCM data
     messages", which the app decrypts before posting the notification
     itself ([FCM](https://firebase.google.com/docs/cloud-messaging/encryption)).
     iOS lets a Notification Service Extension "decrypt an encrypted
     data block"
     ([Apple](https://developer.apple.com/documentation/usernotifications/unnotificationserviceextension)).
   - **The fallback text is generic.** If the extension does not finish
     in time, iOS "displays the original contents"
     ([Apple](https://developer.apple.com/documentation/usernotifications/modifying-content-in-newly-delivered-notifications)),
     and FCM's newer message type has a plaintext fallback of its own.
     So the text outside the seal is always "Saans: urgent alert. Open
     the app."
   - **Gaps are visible.** Each sealed payload carries a per-home
     sequence number, so a phone can tell when a message is missing.
   - **Why a box and not ADR-012's HMAC.** Confidentiality is the point
     here, which HMAC does not give. And with a fresh sender key for
     every message there is no nonce that could repeat, which is the
     failure ADR-012 chose HMAC to avoid.

4. **What leaves the home.** These are the only messages, and the table
   says what each one holds.

   | Message | When | Inside the seal | The relay sees |
   |---|---|---|---|
   | Alert | an urgent or heads-up alert opens; this includes "a room cannot see" | alert id, level, kind, the room as the family labelled it, the time it opened, state, escalation step | class, recipients, time |
   | Alert update | acknowledged, resolved, movement seen since | alert id, new state, who, when | class, time |
   | Home now | an enrolled phone asks | home, away, not seen or unknown; the last room with movement; the time | class, which phone asked, time |
   | Daily summary | once a day, only if the monitored person allows it | the summary items the spec lists | class, time |
   | Poll | continuously | nothing: it is the poll | that the home is alive |

   - **Never:** raw CSI, FeatureFrames, the time series behind the
     charts, the event log or the audit log.
   - **Health details stay home on this path in the first version.**
     Heart rate never leaves, and alerts carry no breathing numbers.
     `include_health_details` is refused on a relay route.
   - A breathing trend inside the summary, if the monitored person opts
     in, is a later decision.

5. **Delivery is confirmed by phones and by people, not by
   platforms.** There are three receipts, and the relay records all
   three:
   1. the relay accepted the message;
   2. a phone decrypted it, and said so with a signed receipt;
   3. a person acknowledged it.

   The gateway's open → acknowledged → resolved stays the record of the
   alert. An acknowledgement from a phone reaches the gateway as an
   `ack` by that phone's label, and goes into the audit log like one
   made from the dashboard.

6. **The relay owns the knocking. The gateway owns the alert.**
   - The escalation plan is stored at the relay when phones are
     enrolled. The relay runs the clock for every open urgent alert, so
     escalation carries on even if the home goes dark straight after
     raising one. For an urgent alert [proposal, after the spec]:
     - **0 min:** push to the primary contacts, repeated every 2 min
       until someone acknowledges.
     - **2 min:** an SMS to any primary contact whose phone has not
       sent a receipt. The text is generic: "Saans: urgent alert from
       <home name>. Open the Saans app or call home."
     - **5 min:** push and SMS to every family contact.
     - **10 min:** a voice call to the nearby contact. The message is
       generic, and pressing 1 acknowledges.
   - **Heads-ups** are push only. The phone holds them through each
     person's quiet hours; the relay does not.
   - **Saans never calls emergency services.**
   - On a relay route, the gateway's own `ack_timeout_min` and
     `escalate_to` are refused at config load, so the alert has one
     clock, not two.

7. **Every failure is seen by someone.**

   | Failure | Who notices | What happens |
   |---|---|---|
   | The home loses power or internet | the relay, after 10 min without a poll [spec proposal] | a heads-up to the family, "No contact with the home since 14:05", by push and SMS. SMS too, because a stopped app would hide a push |
   | The gateway cannot reach the relay | the gateway | it keeps polling with backoff and holds alerts in `alerts.jsonl`. It shows "relay unreachable since" on the local dashboard and in `/api/health`. After 5 min [proposal], it sends urgent alerts directly by its own fallback, an SMS with the generic text, if one is configured |
   | The relay is down | the family's phones and the operator | the app checks in with the relay when opened, and at least daily, and says "Saans cannot reach its service; alerts may not arrive". An external uptime check pages whoever runs the relay. Stale never reads as quiet, as on the dashboard (`README.md`, "Data-source modes are never mixed") |
   | A push never arrives (the app was stopped) | the relay, from the missing receipt | an SMS at 2 min |
   | Nobody acknowledges | the relay | the plan above runs to its end. The alert stays open and keeps repeating to everyone, and the dashboard shows it unacknowledged |
   | The relay drops, delays or forges messages | the phones; the gateway | dropped or delayed messages show as gaps in the sequence. Forged alerts fail the gateway's signature, and forged acknowledgements fail the phone's |

8. **Telegram stays, as an interim channel for the demo only (#47),**
   with the people in the demo consenting to it.
   - It is not the channel for pilot families. Telegram reads the
     content, it reports no delivery, and no processor contract is on
     offer (see "DPDP Act 2023" below).
   - WhatsApp is not used in the first version.
   - SMS and voice are fallbacks, never the primary channel, and carry
     only generic text.

## Cost per home per month

These are the prices read on 2026-11-06, and a rough worst month.

- **The relay.** One small instance is $4 a month (DigitalOcean,
  Bangalore,
  [pricing](https://www.digitalocean.com/pricing/droplets)) or $3.50–5
  (Lightsail, Mumbai, [pricing](https://aws.amazon.com/lightsail/pricing/)).
  Across a 10-home pilot that is under $0.50 a home. A second instance,
  for when one fails, doubles it.
- **Push:** nothing per message. Apple's developer programme is "99 USD
  per membership year" ([Apple](https://developer.apple.com/programs/whats-included/)),
  for the whole service rather than per home.
- **Fallbacks, in a bad month for one home.** Assume two urgent alerts
  that each run to the end of the plan, and eight power cuts that each
  raise "no contact".
  - Each urgent alert costs about 6 SMS (₹1.50) and one call of a minute
    or two (₹0.38–0.76), so ₹1.88–2.26 by the domestic routes.
  - Each power cut costs about 2 SMS (₹0.50).
  - That comes to about **₹8–9 a home a month**, plus GST. By the
    international routes, which need no Indian entity, each call costs
    $0.05–0.10 instead.
  - A quiet month costs nothing beyond the relay.
- **One-off costs, not priced here:** DLT registration and any minimum
  SMS pack (₹1,250 at MSG91).

The variable cost is dominated by SMS sent because of power cuts. That
makes a UPS for the gateway and the router a cost question as well as a
safety one. The home kit is #42 and #44.

## DPDP Act 2023

ADR-015 says that what leaves the home makes Saans a Data Fiduciary.
This ADR decides what leaves, so it owes the following. None of this
is legal advice; the elder-care spec asks for a lawyer's reading. The
sections below were read from a mirror of the Act, not the Gazette:
[s.8](https://dpdpa.com/dpdpa2023/chapter-2/section8.html) and
[s.16](https://dpdpa.com/dpdpa2023/chapter-4/section16.html).

- **Processors only under a contract (s.8(2): "only under a valid
  contract").** The processors are the hosting provider, Google (FCM),
  Apple (APNs), and the SMS and voice provider. Each needs a data
  processing agreement before the first pilot.
  - Meta publishes processor terms for WhatsApp
    ([terms](https://www.whatsapp.com/legal/business-data-processing-terms)).
  - No processor contract for Telegram's Bot API was found
    [unverified]. That is the reason Decision 8 keeps Telegram to the
    demo.
- **Safeguards (s.8(5)).** Sealing is the safeguard that matters. Someone
  who takes the relay gets routing metadata, push tokens and phone
  numbers, not one alert's content.
- **Breach (s.8(6)).** The Board and every affected person must be told.
  The breach procedure the spec asks for covers the relay, and is
  written before the first pilot.
- **Erasure (s.8(7)).** When consent is withdrawn, the relay deletes the
  home's record, phones, numbers, messages and log. s.8(7)(b) also
  requires the fiduciary to "cause its Data Processor to erase" what it
  was given. The gateway's own files follow their own clocks (ADR-010).
- **Transfers abroad (s.16).** The government "may, by notification,
  restrict the transfer" to countries it names. FCM and APNs are global
  services. The relay is hosted in India anyway.
- **Whose data it is.** Phone numbers are the personal data of the
  people they belong to:
  - each family member enrols their own phone;
  - a nearby contact is added only after they confirm, by a test call
    or by replying, that they agree [proposal];
  - the monitored person sees every recipient (spec §5.5).

## How this keeps ADR-011

ADR-011 rules out a hosted plane because aggregated space state is its
own privacy object. The relay holds no space state. What it does hold
across homes is routing and timing metadata: which homes had an urgent
alert, and when. That is a much smaller version of the object ADR-011
warns about, and it is held because escalation cannot work without it.

It is limited in three ways:

- it is kept for 30 days, then reduced to counts;
- it carries nothing about rooms or kinds;
- a home that withdraws is deleted.

The relay cannot read or command a gateway. Every authorisation is still
made at the home: a phone is enrolled through the gateway, and removed
there. ADR-011's conditions for reopening a hosted plane are about
cross-site dashboards. This ADR does not touch them.

## Consequences

- **There is a service to run.** The relay's uptime becomes a safety
  property, for a project with one maintainer. It needs an external
  uptime check, a second instance before the pilot grows, and someone
  who is paged. That is a standing cost, and part of why the relay does
  so little.
- **The gateway grows:**
  - a `relay:<name>` channel;
  - phone enrolment, through a QR code on the local dashboard, approved
    by the monitored person;
  - a key in an environment variable;
  - the config rules in Decisions 4 and 6.
  
  The sealing and signing will need a cryptography crate. That is a new
  dependency, to be justified in the PR that adds it, as CONTRIBUTING
  requires.
- **The app has duties of its own (ADR-017):** keeping a private key in
  the platform's secure storage, decrypting before display, sending
  receipts and acknowledgements, and saying when the relay cannot be
  reached.
- **SMS and voice take paperwork and lead time:** DLT registration of the
  entity, header and templates, and the A2P call declaration. They
  probably need a registered Indian business. Start before the pilot.
- **Apple may refuse critical alerts.** The guidelines describe them as
  "Urgent information about health and safety ... typically come from
  governmental and public agencies or apps that help people manage their
  health or home"
  ([Apple HIG](https://developer.apple.com/design/human-interface-guidelines/managing-notifications)).
  Saans fits the words, and the grant is still Apple's decision. If it
  is refused, the plan in Decision 6 still ends in an SMS and a phone
  call.
- **"No contact" will fire during power cuts.** Without a UPS it will be
  the largest line in the heads-up false-alarm budget (spec §4.6). Kit
  decision: #42, #44.
- **Issue #40 is not finished by this ADR.** The stub relay and an
  end-to-end test are owed: gateway → relay → a test phone client →
  acknowledgement → gateway, with the relay killed halfway. `docs/NEXT.md`
  should carry this ADR as an item while it is decided and not built,
  which is the rule in `docs/adr/README.md`. That edit, and this ADR's
  row in the index, wait for #32 and #52 to merge.

## What this does not decide

- The app's stack (ADR-017, #41).
- The wire format. When the relay is built, its format goes in `spec/`,
  next to `spec/wire.md`.
- Which SMS and voice provider to use.
- Whether WhatsApp is added later for summaries or heads-ups.
- Care homes, whose staff have a dashboard and do not need a relay.

## What would reverse this

- **Pilot families will not install an app.** WhatsApp templates would
  then become the main channel, and sealing would be lost. That is a new
  ADR, not an edit to this one.
- **Phones stopping the app makes push unusable in practice.** If SMS
  ends up carrying most urgent alerts, the cost and privacy sections
  above need redoing.
- **Running the relay is more than one person can keep reliable.**
  Uptime is the product here, so a managed service would have to be
  weighed against Decision 2.
