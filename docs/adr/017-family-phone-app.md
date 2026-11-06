# ADR-017: The family phone app is native, Android first

**Status:** accepted, 2026-11-06. **Decided, not built.** No app exists
and nothing in the code changes with this ADR. It answers issue #41. It
builds on ADR-016 (#40), which decides how alerts reach the phone, and
on the elder-care spec and ADR-015 in
[PR #52](https://github.com/devreddy1015/areu_there/pull/52), which was
not merged when this was written.

## Context

### What the app has to do

ADR-016 decides that alerts are sealed end to end, from the gateway to
each enrolled phone. That gives the app duties beyond showing a screen:

1. **Receive a sealed payload and decrypt it before anything is
   displayed.** On Android that means an FCM *data* message. On iOS it
   means an APNs alert that a Notification Service Extension decrypts.
2. **Make an urgent alert loud,** including at night, and repeat it
   until someone acknowledges.
3. **Keep a private key** in the platform's secure storage.
4. **Send receipts and acknowledgements back,** signed with that key.
5. **Say when the relay cannot be reached,** so that silence never reads
   as a quiet day.

The rest of what the spec in PR #52 asks for is small. It is one alert
screen with three buttons ("I'm checking", "All fine", "False alarm"),
who else is acting, "home now", each room's sensing status, quiet hours,
and enrolment by scanning a QR code on the gateway. There are no charts
and no video.

### Who uses it

- **India is almost all Android.** In October 2026, StatCounter put
  mobile Android at 94.84% and iOS at 5.1%
  ([StatCounter](https://gs.statcounter.com/os-market-share/mobile/india)).
- **Family members abroad are much more likely to carry an iPhone.**
  iOS was 51.96% in the US
  ([StatCounter](https://gs.statcounter.com/os-market-share/mobile/united-states-of-america))
  and 50.14% in the UK
  ([StatCounter](https://gs.statcounter.com/os-market-share/mobile/united-kingdom)).
- **Indian Android phones stop background apps.**
  - dontkillmyapp ranks Xiaomi, OnePlus and Samsung second to fourth
    among the worst ([dontkillmyapp](https://dontkillmyapp.com/)).
  - Samsung "may re-add them later after a firmware update" when a user
    exempts an app ([Samsung](https://dontkillmyapp.com/samsung)).

### Who maintains it

Effectively one person. The repository is Rust, Python and C++, plus
plain JavaScript for the site, the dashboard and the browser tests. It
has no Kotlin, Swift or Dart. Issue #41 sets the bar: a new language
must earn its place.

### What makes an urgent alert reach a person, by platform

These facts were read for ADR-016 and decide most of this ADR.

**Android**

- **Decryption needs a data message, handled by the app.** FCM
  end-to-end encryption "works with FCM data messages, as standard
  notification payloads are handled by the operating system and cannot
  be decrypted by your app"
  ([FCM](https://firebase.google.com/docs/cloud-messaging/encryption)).
- **The app has seconds, not minutes.** The time it gets is "only
  expected to be long enough to immediately render a notification"
  ([FCM](https://firebase.google.com/docs/cloud-messaging/android/message-priority)).
- **A message that shows nothing is downgraded.** If messages "don't
  result in user-facing notifications, your messages may be
  deprioritized" (same page).
- **Loudness is a channel the user can change.** An IMPORTANCE_HIGH
  channel "Makes a sound and appears as a heads-up notification", and
  "the user can change their preferences for your app's channels at any
  time"
  ([Android](https://developer.android.com/develop/ui/views/notifications/channels)).
- **Do Not Disturb is crossed only with access the user grants in
  Settings**
  ([NotificationManager](https://developer.android.com/reference/android/app/NotificationManager)).
- **A full-screen alert needs permission.** On Android 14, "only apps
  that have calling or alarm functionalities will have this permission
  enabled by default"; other apps need the user to grant it
  ([Play Console help](https://support.google.com/googleplay/android-developer/answer/13392821)).
- **The app may ask to be exempt from battery optimisation.** Android's
  own table lists "Safety app. Apps that keep their users and their
  families safe." as an acceptable reason
  ([Android](https://developer.android.com/training/monitoring-device-state/doze-standby)).
  That is the platform's way through the stopping above, as far as it
  goes.

**iOS**

- **Decryption happens in a Notification Service Extension.** Apple's
  example is that "you could use the extension to decrypt an encrypted
  data block"
  ([Apple](https://developer.apple.com/documentation/usernotifications/unnotificationserviceextension)).
  - It has "only about 30 seconds to modify the payload"
    ([Apple](https://developer.apple.com/documentation/usernotifications/modifying-content-in-newly-delivered-notifications)).
  - If it fails, the system "displays the original contents", which is
    why ADR-016 keeps the outer text generic.
- **Loudness comes in two levels.**
  - *Time-sensitive* needs only a capability in Xcode, but "The user can
    turn off the ability for time sensitive notification interruptions"
    ([Apple](https://developer.apple.com/documentation/usernotifications/unnotificationinterruptionlevel/timesensitive)).
  - *Critical* "requires an approved entitlement" and "The system always
    presents this notification, even when Do Not Disturb is active"
    ([Apple](https://developer.apple.com/documentation/usernotifications/unnotificationinterruptionlevel/critical)).
- **Building needs a Mac.** The latest Xcode lists "macOS Tahoe 26.6 or
  later" as its supported system
  ([Apple](https://developer.apple.com/xcode/system-requirements/)).

**The web**

- **On iOS, Web Push works only for an installed web app.** It came "to
  Home Screen web apps" in iOS 16.4
  ([WebKit](https://webkit.org/blog/13878/web-push-for-web-apps-on-ios-and-ipados/)).
- **A web notification cannot be made loud.**
  - The Notifications API has a `silent` option, but nothing for sound,
    priority, Do Not Disturb or critical alerts
    ([MDN](https://developer.mozilla.org/en-US/docs/Web/API/ServiceWorkerRegistration/showNotification)).
  - `requireInteraction` is not supported by Safari.
  - On Android Oreo and later, the vibration pattern "has no effect".
- **Delivery can be held back.** "battery-preserving features like doze
  mode can delay the delivery of push based notifications"
  ([Chrome](https://developer.chrome.com/docs/web-platform/notification-triggers)).
- **A web page cannot request any of the Android permissions above.**

**Cryptography for the seal (RFC 9180, HPKE)**

- **Apple:** CryptoKit has HPKE from iOS 17.0
  ([Apple](https://developer.apple.com/tutorials/data/documentation/cryptokit/hpke.json)).
- **Google:** Tink "implements the Hybrid Public Key Encryption (HPKE)
  standard as specified in RFC 9180" ([Tink](https://developers.google.com/tink/hybrid)).
  - Its key types include X25519 with ChaCha20-Poly1305, and X25519 with
    AES-256-GCM.
  - Tink on Android "is fully supported starting from API level 24"
    ([Tink setup](https://developers.google.com/tink/setup/java)).
- **Dart:** no mature HPKE library was found.
  - The maintained libsodium binding, `sodium`, offers sealed boxes
    ([pub.dev](https://pub.dev/packages/sodium)).
  - The packages that advertise HPKE are young and little used
    ([pub.dev search](https://pub.dev/packages?q=hpke)).

## The options

### A progressive web app

- **For:** no new language. It would be the JavaScript the site already
  uses, and one codebase with no store.
- **Against:**
  - It cannot make an urgent alert loud on either platform, cannot
    cross Do Not Disturb, and cannot ask for a battery exemption.
  - On iOS it needs the family member to add it to the Home Screen
    before any notification arrives.
  - It fails the first thing the app is for.

### Flutter

- **For:**
  - One UI codebase for both platforms, in Dart.
  - On Android the whole urgent path can be Dart. FlutterFire runs a
    background message in its own isolate.
  - The `flutter_local_notifications` plugin can post the alert:
    - on a channel with `Importance.max`;
    - with `bypassDnd`, which "should attempt to bypass Do Not Disturb
      settings" after the app has asked for policy access;
    - with a full-screen intent;
    - with the iOS critical level ("Subject to specific approval from
      Apple")
    ([plugin API](https://pub.dev/documentation/flutter_local_notifications/latest/flutter_local_notifications/AndroidNotificationChannel-class.html),
    [README](https://pub.dev/packages/flutter_local_notifications)).
- **Against:**
  - **The iOS half of the urgent path is not Dart.** Firebase's own
    Flutter guide
    ([FlutterFire](https://firebase.google.com/docs/cloud-messaging/flutter/receive-messages))
    says that changing a notification before it is shown on iOS means
    "you must add an additional notification service extension". It is
    created in Xcode, in Swift or Objective-C. So Flutter adds two
    languages, Dart and Swift, the same count as native.
  - **On Android, a Flutter engine has to start in a background isolate
    before the alert can be decrypted and shown,** where "it is not
    possible to update application state or execute any UI impacting
    logic" (same guide). Whether that start fits the seconds Android
    gives, on a cheap phone, is not known. It would have to be measured.
  - **The seal would rest on third-party code on both platforms.** Dart
    has no mature HPKE, so it would be a libsodium sealed box through a
    community binding in Dart, and again in the Swift extension.
  - **Flutter has its own release cycle.** It ships a stable release
    roughly every quarter, with migration guides for breaking changes
    ([Flutter](https://docs.flutter.dev/release/breaking-changes)). That
    is on top of the yearly platform rules every stack follows.

### Native: Kotlin on Android, Swift on iOS

- **For:**
  - Every platform feature above is a direct call, documented by the
    platform's owner, with no plugin to lag behind it.
  - The seal uses each platform owner's own HPKE: Tink on Android,
    CryptoKit on iOS.
  - The Android urgent path is the platform's own: the FCM service, then
    Tink, then the notification manager, with no engine to start first.
- **Against:**
  - Two UI codebases once the iOS app exists, which is a real cost for
    one person.
  - Two toolchains, one of them on a Mac.

### Also considered

- **React Native.**
  - It would reuse JavaScript.
  - The iOS extension is still added in Xcode as native code
    ([Notifee](https://notifee.app/react-native/docs/ios/remote-notification-support)),
    and alarm-grade Android channels need native code too.
  - So it adds Swift, likely Kotlin, and a Node toolchain for both
    platforms.
- **Kotlin Multiplatform with Compose Multiplatform.**
  - It shares Kotlin, including the UI, across both platforms.
  - Kotlin lists iOS as "Stable"
    ([Kotlin](https://kotlinlang.org/docs/multiplatform/supported-platforms.html)).
  - The iOS extension is still a Swift target.
  - It is not needed for an Android-first app, and it is the first thing
    to weigh when the iOS app starts (Decision 2).

## Decision

1. **The app is native: Kotlin on Android, Swift on iOS.**
   - Native and Flutter each add two languages to the repository, so the
     count does not decide.
   - These do:
     - The seal on both phones comes from the platform owners' own HPKE,
       and matches the gateway's, instead of from community bindings.
     - The Android urgent path has no framework between the push and the
       notification.
     - The first phase is Android only (Decision 2), so Flutter's one
       real gain, a shared UI, would not arrive until later.
     - Upkeep is the platforms' yearly rules, without a framework's
       quarterly ones on top.
   - Kotlin earns its place on the urgent path: the FCM service that
     decrypts, the channel that crosses Do Not Disturb, the full-screen
     intent, the battery-exemption request and the Android Keystore.
   - Swift earns its place there too: the Notification Service
     Extension, critical alerts and the Keychain. It is needed under
     every option that can make an alert loud on an iPhone.
2. **Android first.**
   - Android is 94.84% of mobile use in India.
   - The iOS app comes before any pilot family that includes an iPhone
     user. Family abroad make that likely sooner rather than later.
   - When it starts, sharing the Android app's Kotlin UI through Compose
     Multiplatform is weighed first, because it adds no language. The
     extension stays Swift either way.
   - Until then, an iPhone user is reached by ADR-016's SMS and voice
     fallbacks. Those reach any phone, but they are not sealed, they are
     not loud on the phone's own terms, and they say only "open the app
     or call home".
3. **There is no web app for alerts.**
4. **The seal is HPKE (RFC 9180), in each platform's own library.** This
   narrows ADR-016, which allowed "a sealed box or HPKE".
   - The suite is X25519 key agreement, HKDF-SHA256, and an AEAD offered
     by both Tink and CryptoKit. The AEAD is fixed when the envelope is
     specified in `spec/` [unverified: CryptoKit's list of suites was not
     read].
   - The gateway, the Android app and the iOS app are three
     implementations of one small envelope. They are held to each other
     by golden vectors in `spec/golden/`, the ADR-002 pattern.
   - RFC 9180's own published test vectors are the reference for the
     primitive. The gateway generates the vectors for the Saans envelope
     around it, because the Python reference is standard-library only
     (ADR-002, ADR-012) and the standard library has no X25519.
   - **Not chosen: sharing the gateway's Rust in the app**, through
     generated Kotlin and Swift bindings. It would remove two
     implementations. In exchange, one person would maintain a
     cross-compiling build for every Android CPU type and for iOS, for
     code that both platform owners already ship and that golden vectors
     already hold together.
5. **The UI is the screens listed above, and no more.** Anything beyond
   them (charts, a history, a second home) is a new decision.
6. **Setup walks through every permission the alert depends on, and
   ends with a test alert sent end to end through the relay.**
   - **On Android:**
     - notifications;
     - Do Not Disturb access;
     - the full-screen permission;
     - the battery exemption;
     - the maker's own autostart or background setting, with the steps
       for that maker.
   - **On iOS:**
     - notifications;
     - time-sensitive notifications;
     - critical alerts, once Apple grants the entitlement.
   - **Force-stopping kills delivery.** If an app is force-stopped from
     the phone's settings, "it must be manually reopened for messages to
     start working" (FlutterFire guide, above). Setup says so in plain
     words.
   - **Afterwards:** each phone's last receipt is shown to the family
     (ADR-016, Decision 5). A phone that has stopped receiving shows up
     before an alert depends on it. A test alert can be sent at any
     time.

## Consequences

- **Two toolchains.** Android Studio and Gradle run on Linux. Xcode
  needs a Mac, or a hosted macOS build machine. That applies to every
  option except the web app.
- **Every year's platform release is work, whatever the stack.**
  - Since 31 August 2026, new apps and updates on Google Play "must
    target Android 16 (API level 36) or higher"
    ([Android](https://developer.android.com/google/play/requirements/target-sdk)).
  - App Store uploads "must be built with Xcode 26 or later using an SDK
    for iOS 26" ([Apple](https://developer.apple.com/news/upcoming-requirements/)).
- **The floor is Android 7.0 and iOS 17.** These come from Tink (API
  level 24) and CryptoKit HPKE (iOS 17). A phone below either is reached
  only by the SMS and voice fallbacks.
- **Apple decides whether urgent alerts can be critical.** If it
  declines, iOS has time-sensitive alerts, which the user can switch
  off, and the ADR-016 fallbacks.
- **The test client for issue #40's end-to-end test is not this app.**
  It is a small command-line client in the repository, so the relay can
  be tested in CI with no phone.
- **New golden vectors** for the envelope, in `spec/golden/`, before
  either app decrypts its first alert.
- **No new dependency enters the repository with this ADR.** Each app
  brings its own, Tink on Android first, and the PR that adds an app
  names them.

## What would reverse this

- **The pilot finds more iPhone users than expected.** Then iOS moves
  earlier. That reorders this decision; it does not reverse it.
- **Two native apps prove too much for one maintainer.** Compose
  Multiplatform is the first answer, because it keeps Kotlin. Flutter
  would be weighed again only if that fails, knowing the iOS extension
  stays Swift.
- **Families will not install an app at all.** That reverses ADR-016's
  choice of channel, and this ADR with it.
