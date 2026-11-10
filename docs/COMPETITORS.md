# Competitors and alternatives for elder-care sensing

**Status:** research, read 2026-11-06, for issue #48. Nothing in the code
depends on it. Prices and product status change quickly. Every figure
here is what a page said on that day, so re-check before quoting one
after that date.

## The question

This file answers the question a family asks. An older parent lives
alone in India, and someone elsewhere wants to know they are all right.
For each option the family can buy today, it covers:

- what it actually detects;
- what it costs;
- what it takes to install;
- what leaves the home;
- how good the evidence is that it works.

It also places Saans among them. Saans has not yet run on real hardware.

The customer and the signals are the ones in ADR-015 and the elder-care
spec. Both are in [PR #52](https://github.com/devreddy1015/areu_there/pull/52),
which was not merged when this was written:

- one person living alone, with alerts going to family;
- the lead safety signal is a **long lie**: present, but not moving for
  N minutes after a sudden movement. It is not fall detection.

## How to read this

- **Sources.** Every claim about a product links the page it was read
  from on 2026-11-06. **[unverified]** marks a claim that rests on a
  search-result snippet, a secondary source or a page that would not
  load, and says which.
- **Vendor wording is quoted, not upgraded.** "Claims fall detection"
  means the vendor says so. Where only the vendor has published a
  number, the evidence line says *vendor claim only*. That means
  unchecked, not false.
- **Evidence, strongest first:**
  1. a peer-reviewed study of the product in real use;
  2. a regulator's clearance or listing;
  3. a vendor figure with a stated method;
  4. a vendor claim with no method.
- **Saans has no real-world numbers.** No board has run its firmware
  ([`docs/EVIDENCE.md`](EVIDENCE.md) §1), so every Saans figure in this
  repository is **SYNTHETIC**. This file quotes none of them. It names
  the section instead, so nothing here can drift from
  [`docs/RESULTS.md`](RESULTS.md).
- **Method.** Four research passes, one per category, read the pages
  linked below. The claims the closing section depends on were then
  read a second time:
  - the NoBlink page;
  - the Cognitive Caregiver pages;
  - ADT's acquisition of Origin;
  - the XZent interview;
  - the BMJ long-lie study;
  - the SafelyYou study;
  - the Vayyar nursing-home pilot.

---

## 1. What every option here is trying to prevent

A fall is rarely the whole harm. The harm that grows with time is lying
on the floor, unable to get up, with nobody knowing.

Fleming and Brayne followed 110 people aged over 90 (BMJ 2008,
[PMC2590903](https://pmc.ncbi.nlm.nih.gov/articles/PMC2590903/)):

- "80% (53/66) were unable to get up after at least one fall".
- "30% (20/66) had lain on the floor for an hour or more".
- In the falls where the person was alone and could not get up, "80%
  (113/141) ... did not use their call alarm to summon help".
- In "97% of these 'long lies' (37/38) the person who fell alone did not
  use their alarm".

Every option below gets help in one of three ways:

- the person **presses** something: an SOS button;
- the person **wears and charges** something that decides for them: a
  pendant or watch with automatic fall detection;
- the person does **nothing**, and a sensor in the room watches: a
  camera, or a radar, thermal, WiFi or other ambient sensor.

Saans is in the third group, and so are its closest competitors.

## 2. At a glance

| Category | Example sold in India | Cheapest found in India | Must be worn | Forms an image | Claims falls | Best independent evidence found |
|---|---|---|---|---|---|---|
| Consumer camera | Qubo, Tapo, Xiaomi | about ₹1,400–2,800, plus cloud (§3.1) | no | yes | no; motion or person alerts | none |
| Elder-care camera | none found | none (US: $99.99 plus $45 a month, §3.2) | no | yes, processed or deleted | yes | SafelyYou: observational studies, authors tied to the company |
| Pendant or SOS service | NSS pendant; Emoha | ₹9,300 device; ₹1,999 a month service (§4.3) | yes | no | some | wearables in general: poor in real use (§4.4) |
| Smartwatch | Apple, Samsung, Google; AGEasy Empower | ₹4,999 (AGEasy Empower, §4.3); ₹29,900 (Apple Watch SE 3) | yes, charged daily | no | "hard falls" | none peer-reviewed for these watches |
| Radar | NoBlink (installed); Aqara FP2 (imported) | ₹499 for a bare module; FP2 ₹7,999–11,777 [price tracker] (§5) | no | no | yes, most | Vayyar: one lab test, one nursing-home pilot in 13 residents' rooms (§5.2) |
| Under-mattress | Dozee | $345 plus $25 a month, priced in US dollars (§6) | no; works only in bed | no | no | Dozee: FDA-cleared vital signs; company-authored studies |
| WiFi sensing | WiZ SpaceSense lights; XZent (pilot) | about ₹1,740 a light, two needed (§7) | no | no | no | none for any product sold |
| **Saans** | none | **unpriced; not a product** | no | no | **no**; a long lie instead | **none: simulation only** |

---

## 3. Cameras

### 3.1 Consumer cameras families already use

None of the cameras checked that are sold in India advertises fall or
long-lie detection. They sell live video, two-way talk and motion or
person alerts, and most families already have one.

| Product | What it claims | Price | Install | What leaves the home |
|---|---|---|---|---|
| **TP-Link Tapo C200 / C210** (India) | "smart motion detection technology" ([C200](https://www.tapo.com/in/product/smart-camera/tapo-c200/)); "built-in microphone and speaker" ([C210](https://www.tapo.com/in/product/smart-camera/tapo-c210/)) | about ₹1,399–1,599 [unverified: an expired deal listing on [desidime](https://www.desidime.com/deals/tp-link-tapo-c200-white-black-full-hd-pan-tilt-home-security-wi-fi-smart-camera); retailers refused automated reads]. The cloud plan page ([Tapo Care India](https://www.tapo.com/in/tapocare/)) lists no ₹ price | DIY, in the app | a microSD card at home; with Tapo Care, "event clips to the cloud" (same page) |
| **Qubo Smart Cam 360 3MP** (Hero Electronix, India) | "AI-Powered Person Detection"; marketed to "care for your parents" ([collection](https://quboworld.com/collections/indoor-cameras)) | "Rs. 2,789.00", MRP Rs. 3,990; cloud from "Rs. 1,490/yr" ([product](https://www.quboworld.com/products/smart-cam-360-3mp)) | DIY, in the app | a microSD card, or "end-to-end encrypted Indian cloud servers up to 28 days with a subscription plan" (collection page) |
| **Xiaomi 360 Home Security Camera 2K** (India) | "AI Human detection" | Rs. 3,299 at launch, January 2024 [unverified: secondary, [fonearena](https://www.fonearena.com/blog/414221/xiaomi-360-home-security-camera-2k-price-india-features.html); mi.com/in refused the read] | DIY | a microSD card or a NAS [unverified, same source] |
| **Ring Indoor Cam, 2nd gen** (US) | "Two-way audio"; person alerts with a paid plan ([plans](https://ring.com/plans)) | $49.99 ([product](https://ring.com/products/mini-indoor-security-camera-plug-in)); plan "$4.99/mo or $49.99/yr" | plug in, "5 to 10 minutes" | without a plan "you won't be able to review any videos or images you missed"; end-to-end encryption is "an optional security feature" ([Ring](https://ring.com/support/articles/myr75/using-video-end-to-end-encryption)) |
| **Wyze Cam v4** (US) | motion and sound; person detection with Cam Plus | $44.98 ([product](https://www.wyze.com/products/wyze-cam-v4)); Cam Plus "$2.99/mo" ([plans](https://www.wyze.com/pages/service-plans)) | DIY, 2.4 GHz WiFi only | a microSD card, or "14-day cloud event history" with the plan |

None of the five has any evidence for elder care.

**What has gone wrong with cameras before.** Each of these is a
documented incident, not a hypothetical.

- **Ring.**
  - The US FTC said Ring let "any employee or contractor ... access
    consumers' private videos".
  - Hackers reached the "account profiles of approximately 55,000 U.S.
    customers".
  - Ring settled for "$5.8 million"
    ([FTC, May 2023](https://www.ftc.gov/news-events/news/press-releases/2023/05/ftc-says-ring-employees-illegally-surveilled-customers-failed-stop-hackers-taking-control-users)).
- **Wyze.** "about 13,000 Wyze users received thumbnails from cameras
  that were not their own"
  ([Engadget, February 2024](https://www.engadget.com/wyze-camera-security-issue-showed-13000-users-other-owners-homes-140059551.html)).
- **Xiaomi.** A Google Nest Hub showed stills from other people's Xiaomi
  cameras, and Google disabled Xiaomi integrations. Xiaomi blamed "a
  cache update on December 26, 2019"
  ([The Register](https://www.theregister.com/2020/01/03/google_blocks_xiaomi/)).
- **Tapo.** A stack-based buffer overflow in the Tapo C200 v3's ONVIF
  parser is recorded as CVE-2025-8065
  ([CVE record](https://cveawg.mitre.org/api/cve/CVE-2025-8065)).

### 3.2 Cameras built for falls

None of these was found on sale in India.

**Kami Fall Detect camera (Kami Vision, US).** Launched January 2025
([press release](https://prnewswire.com/news-releases/kami-vision-brings-ai-powered-fall-detection-to-homeowners-and-caregivers-with-at-home-camera-302342493.html)).

- **Claims:** it "identifies falls with 99.5%* accuracy", and alerts go
  to a 24/7 professional monitoring service.
- **Price:** "introductory price of $99.99", plus "a monthly
  subscription fee of $45". kamihome.com did not load, so the current
  price was not checked.
- **Privacy:** video is "viewed remotely by a team of professional
  monitors", "only displayed in blurred snippets".
- **Evidence:**
  - The 99.5% comes from the company's own data. It was computed by
    "dividing the number of fall incidents captured via KamiCare by the
    number of fall incidents reported".
  - That figure has no false-alarm denominator, and a fall nobody
    reported is in neither count.
  - The regulatory status it claims is "FDA** registration", which is
    registration, not clearance.

**SafelyYou (US, memory-care facilities).**

- **Claims:** "AI-enabled camera systems that detect, in real time,
  falls of individuals with dementia"
  ([Berkeley CGHDDE](https://cghdde.berkeley.edu/projects/safelyyou)).
- **Price:** not published; sold to facilities.
- **Privacy:** "Non-fall video is automatically deleted within 60
  seconds and no audio is recorded" (same page).
- **Evidence, the strongest in this category:**
  - Bayen et al., JMIR 2021
    ([PMC8277400](https://pmc.ncbi.nlm.nih.gov/articles/PMC8277400/)),
    was an observational study with no randomised control group. It ran
    for 10 months in 6 California memory-care facilities, with 66
    residents and 436 falls.
  - Time on the ground fell by about half an hour, and "the proportion
    of fallers with TOG >1 hour fell from 31% (8/26; baseline) to zero
    events".
  - It found "over 80% of the falls happened in bedrooms, with
    two-thirds occurring overnight".
  - Several of its authors report advisory roles, employment or stock in
    SafelyYou.
  - A smaller 2019 pilot did have a control group. It reported "relative
    reductions of 75% (P = .001) in EMT visits and 80% (P = .003) in ED
    visits"
    ([AJMC](https://www.ajmc.com/view/realtime-video-detection-of-falls-in-dementia-care-facility-and-reduced-emergency-care)).
    Its authors also disclose a SafelyYou role.
  - The registered randomised trial shows "hasResults: false"
    ([NCT04478149](https://clinicaltrials.gov/api/v2/studies/NCT04478149)).

**Nobi smart lamp (Belgium; mainly care homes).**

- **Claims:** "Fall detection in the bedroom and bathroom, with nothing
  to wear" ([nobi.life](https://www.nobi.life/en)).
- **How it works:** it is a camera in a ceiling lamp. "Nobi's optical
  sensors take images of the room every second"
  ([privacy PDF](https://info.nobi.life/wp-content/uploads/2025/03/6.-Privacy_to-download.pdf)).
- **Privacy** (same PDF):
  - images are interpreted "locally, within the lamp";
  - they go to the cloud "only in the event of a fall";
  - they are "automatically deleted 14 days after the fall".
- **Price:** not on the site. In 2023 it was $2,490 plus $19 a month
  [unverified: secondary,
  [Digital Trends](https://www.digitaltrends.com/home/nobi-smart-lamp-provides-help-when-older-people-fall/)].
  The site's region selector has no India option.
- **Evidence:**
  - A vendor figure only: "Help after a fall in under 4 minutes on
    average" ([why-nobi](https://www.nobi.life/why-nobi)).
  - The site also reports an NHS order "for 500 additional units across
    50 care homes". That is a purchase, not an outcome.

**Teton.ai (Denmark; care homes and hospitals).**

- **Claims:** "uses a device equipped with a camera to monitor the room"
  ([FDA GUDID](https://accessgudid.nlm.nih.gov/devices/05745001037000)).
- **Privacy:** "Raw images never leave the room"
  ([Teton](https://www.teton.ai/blog/privacy)).
- **Evidence:**
  - A vendor figure, "up to an 83% decrease in fall-related incidents",
    with no method given (same page).
  - The FDA device database lists it as "Device Exempt from Premarket
    Submission: Yes". That is a listing, not a clearance.
- **Price:** not published. The vendor installs it.

### 3.3 What cameras do well and badly here

- **Well:**
  - They are cheap, sold everywhere in India, and the family can set
    one up in minutes.
  - Live video and two-way talk answer "is she all right?" directly,
    which nothing else in this file does.
- **Badly:**
  - The rooms that matter are the ones where a camera is least
    acceptable. In SafelyYou's study most falls were in bedrooms and
    overnight, and those were facilities where residents and families
    had agreed to cameras.
  - The vendors who take this seriously keep images away from people in
    software. They process on the device (Nobi, Teton), delete non-fall
    video (SafelyYou) or blur it (Kami). The incidents above are what
    happens when that software, or its operator, fails.
- **Evidence:** only SafelyYou has peer-reviewed outcome data, and that
  comes from observational studies with authors tied to the company.

---

## 4. Wearables and pendants

### 4.1 US medical alert systems

These are US-only services: each depends on a US mobile network and a US
monitoring centre. They are listed because they define what most of the
literature means by "an alert system for an older person".

| Service | Claims | Price | Wear and charge | Evidence |
|---|---|---|---|---|
| **Medical Guardian, Guardian Flex** | SOS, GPS, "24/7 U.S.-based monitoring center" ([home](https://www.medicalguardian.com/)); fall detection is an "optional add-on" ([Flex](https://www.medicalguardian.com/products/guardian-flex)) | "Annual Plan $39.95/mo"; device $149.95 list (Flex page). The fall add-on is about $10 a month [unverified: review sites only] | "Up to five days of power" | vendor claim only; it says "fall detection is very good but does not work 100% of the time" (home page) |
| **Bay Alarm Medical** | its automatic fall detection connects "without needing to push a button" ([pricing](https://www.bayalarmmedical.com/pricing/)); "optimized for fall detection accuracy when worn around the neck" ([All-In-One](https://www.bayalarmmedical.com/medical-alert-system/sos-all-in-one/)) | from "$27.95/mo" (landline home unit); All-In-One 2 "$34.95/mo"; about $10 a month more with fall detection (pricing page) | home buttons "last up to five years without charging"; the mobile unit lasts "up to 72 hours" | vendor claim only |
| **Lifeline** (formerly Philips; sold to Connect America in [2021](https://telecareaware.com/connect-america-acquires-philips-aging-and-caregiving-including-lifeline/)) | help is dispatched "even if you're unable to press your button" ([lifeline.com](https://www.lifeline.com/)) | "Starting at $39.95/mo", plus "$50 activation fee and a $49.95 programming fee" ([On the Go](https://www.lifeline.com/medical-alert-systems/on-the-go/)); fall detection is "an additional $15 per month" ([fall detection](https://www.lifeline.com/fall-detection/)) | "up to four days on a single charge" | vendor claim only; it "does not detect 100% of falls" |
| **Lively Mobile2** (Best Buy Health) | "Automatically calls for help if a fall is detected" ([product](https://shop.lively.com/products/lively-mobile2-all-in-one-medical-alert)) | "$89.99" device; "Basic: $24.99/month"; fall detection "$9.99/month" ([plans](https://shop.lively.com/pages/personal-emergency-response)) | "You must wear the Lively Mobile2 with the lanyard around your neck"; "Up to 3 days" on a charge | vendor claim only; it "does not detect 100% of falls" |
| **Life Alert** | a button press; no automatic fall detection on its device page ([devices](https://www.lifealert.com/medical-alert-devices)) | not published | "Never needs recharging" | none |

### 4.2 Smartwatches with fall detection

| Watch | Claims | Price | Catch |
|---|---|---|---|
| **Apple Watch SE 3** | detects hard falls; calls emergency services if the wearer is "immobile for about a minute"; on by default from age 55 ([Apple support](https://support.apple.com/en-us/108896)) | "From ₹29900.00" ([India](https://www.apple.com/in/shop/buy-watch/apple-watch-se)); "From $249" ([US](https://www.apple.com/shop/buy-watch/apple-watch-se)) | needs an iPhone; "Apple Watch cannot detect all falls"; "all-day, 18‑hour battery life" ([SE 3](https://www.apple.com/in/apple-watch-se-3/)) means it comes off the wrist at night to charge |
| **Samsung Galaxy Watch8** | sends an SOS message to contacts after 60 s without a response ([Samsung AU](https://www.samsung.com/au/support/mobile-devices/activate-hard-fall-detection/)) | "Rs. 32,999" Bluetooth, "Rs. 36,999" LTE ([fonearena](https://www.fonearena.com/blog/458602/samsung-galaxy-watch8-watch8-classic-price-india-features.html), press) | "Galaxy Watch cannot detect all falls"; needs a paired phone. Fall detection on Indian units [unverified: Samsung India pages returned 404] |
| **Google Pixel Watch 4** | "calls emergency services after 60 seconds"; India is on the supported list ([Google support](https://support.google.com/googlepixelwatch/answer/12663810?hl=en)) | "Rs 39,900 for the 41mm Wi-Fi version" ([fonearena](https://www.fonearena.com/blog/?p=462278), press) | the Indian model has no LTE, so "your paired phone must be nearby"; "may not detect all falls" |

Apple's ECG app has FDA De Novo authorisation
([DEN180044](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/denovo.cfm?id=DEN180044)).
No FDA authorisation for fall detection on any of the three watches was
found [unverified: the FDA databases were not searched exhaustively].
No peer-reviewed real-world validation of any of the three in older
adults was found either.

### 4.3 What is sold in India

India has no mature, professionally monitored alarm market with
published prices. What exists is concierge elder-care subscriptions,
cheap devices that alert family, and early startups.

- **Emoha** (Gurugram) is a concierge service.
  - It offers "24/7 Emergency Coordination", from "₹1,999/month*" to
    "₹10,999/month*" ([plans](https://www.emoha.com/plans)).
  - SOS works from the app, a panic button or a wearable
    ([emoha.com](https://www.emoha.com/)).
  - It does not claim automatic fall detection.
- **AGEasy Empower** is a fall-detection smartwatch
  ([1mg](https://www.1mg.com/otc/ageasy-empower-fall-detection-smartwatch-for-seniors-otc946484)).
  - It "Automatically alerts designated contacts in the event of a
    seniors fall".
  - It costs "₹4999", and the battery lasts "up to 7 days".
  - It alerts family, not a call centre. Vendor claim only.
- **NSS Security** (Bangalore) sells a medical alert pendant
  ([NSS](https://www.nsssecure.com/product/medical-alert-system/)).
  - "Worn around neck or wrist, the device can detect fall and notify
    loved ones".
  - It costs "₹ 9,300.00", and was out of stock when read.
  - Vendor claim only.
- **Others, briefly:**
  - Samarth offers "24X7 support for any health emergency" and lists no
    device or price ([Samarth](https://care.samarth.community/)).
  - DrRing launched an SOS wearable in May 2026, with no automatic fall
    detection stated
    ([BioSpectrum](https://www.biospectrumindia.com/news/20/27879/first-hospital-integrated-wearable-emergency-response-device-launches-in-india.html)).
  - Anvayaa's site did not resolve.
  - Ansfrid (Kochi) used wall-mounted bathroom sensors in 2023: "if the
    person does not move for 30 seconds, then it sends an alert to us"
    ([The Better India](https://thebetterindia.com/333453/kochi-startup-ansfrid-smartcare-emergency-alert-smart-watch-for-senior-citizens-technology/)).
    That is a stillness rule, like the long lie.

### 4.4 The evidence on wearables in real use

This is the strongest evidence in this file, and most of it counts
against the category.

**Automatic fall detection is much worse in real use than in the lab.**

- **Chaudhuri et al., J Am Geriatr Soc 2015**
  ([PMC4662041](https://pmc.ncbi.nlm.nih.gov/articles/PMC4662041/))
  followed 18 older adults at home wearing a commercial device.
  - Real-world sensitivity was 1 fall in 4, against a company-reported
    "sensitivity ranging from 94.1% to 94.4%".
  - "A total of 84 alarms indicating a fall were recorded, of which 83
    were false alarms."
- **Lipsitz et al., J Am Geriatr Soc 2016**
  ([Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:26783046%20AND%20SRC:MED&resultType=core&format=json))
  was in a nursing home.
  - "The device detected 17 of 89 total falls ... (sensitivity 19%)".
  - "111 (87%) were false positives".
- **Bagalà et al., PLoS One 2012**
  ([PMC3353905](https://pmc.ncbi.nlm.nih.gov/articles/PMC3353905/))
  ran 13 published algorithms on 29 real falls. Their sensitivity was
  "57.0%±27.3%".
- **Kangas et al., Gerontology 2015**
  ([Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:25138139%20AND%20SRC:MED&resultType=core&format=json))
  is the best real-world result found.
  - It "detected 12 out of 15 real-life falls".
  - It had "a false alarm rate of 0.049 alarms per usage hour", about
    one false alarm per day of wear.
- **Singh et al., JMIR mHealth 2024**
  ([PMC11399740](https://pmc.ncbi.nlm.nih.gov/articles/PMC11399740/)),
  a meta-analysis, grades the evidence for fall detection "(D)
  inconsistent or inconclusive".

**Devices are not worn when it matters, and buttons are not pressed.**

- **Fleming and Brayne** (§1): people who were alone and unable to get
  up mostly did not use the alarm. One participant had "already taken
  it off ready for bed".
- **Stokke, JMIR 2016**
  ([PMC4965612](https://pmc.ncbi.nlm.nih.gov/articles/PMC4965612/)), an
  integrative review:
  - "about 25% of the respondents never wore the pendant";
  - alarms were "highly accepted but rarely used".
- **De San Miguel and Lewin, Australasian J Ageing 2008**
  ([Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=TITLE:%22Personal%20emergency%20alarms:%20what%20impact%20do%20they%20have%20on%20older%20people%27s%20lives%22&resultType=core&format=json))
  surveyed 2,610 alarm users. "use is low in the shower/bath and in bed
  at night".
- **Chaudhuri 2015** again: "8 falls were reported that occurred while
  participants were not wearing their devices", and "half of these
  occurred with the device in the charger".

### 4.5 What wearables do well and badly here

- **Well:**
  - They work outside the home, over GPS and cellular.
  - They give two-way voice to a call centre or to family.
  - A button lets a conscious person ask for help at once, without
    waiting for any timer.
- **Badly:**
  - They help only when worn and charged. Non-wear clusters in bed, at
    night and in the shower, which is where and when a person living
    alone is likely to lie undiscovered.
  - Automatic detection loses most of its lab accuracy in real use and
    raises many false alarms. Every vendor checked disclaims it.

---

## 5. Radar and mmWave

Radar is the category closest to Saans: no image, nothing to wear, and
it works in the dark and in shower steam. It is also ahead of Saans on
the one thing Saans cannot do. A mmWave radar with an antenna array
measures range and angle, so it can see height. The Vayyar device below
confirms a fall by "a measurement of a height difference". A Saans node
has one antenna and cannot see height
([`docs/CAPABILITY_LIMITS.md`](CAPABILITY_LIMITS.md) §1).

### 5.1 In India

**NoBlink** (Bangalore) is the closest competitor found to what Saans
plans ([fall detection page](https://noblink.ai/fall-detection)). It
installs in Bangalore, Chennai, Mumbai, Pune and Gurugram.

- **Claims:**
  - "Camera-free fall detection for older adults living at home."
  - "Small radar sensors on the wall notice a fall or a long, unusual
    stillness and call family straight away."
  - It names "slow slides down to the floor", "a person who sat down
    and could not get up again" and "a far longer stay in the bathroom
    than normal".
- **Products:** a bathroom sensor, a room sensor and a bed monitor.
- **Install:** "Our own installer fits the sensors". Each sensor "goes
  high on the wall, plugs into a nearby power point and joins the home
  Wi-Fi".
- **Alerts:**
  - "Automatic voice calls go to the family members, neighbours or
    caregivers you have chosen."
  - "If the first person does not answer, NoBlink escalates to the
    next."
  - There is also an app.
- **Privacy:** "No camera". The page does not say what leaves the home.
- **Price:** not published.
- **Evidence:** none published.
  - "No fall detection system catches every single fall, and we would
    rather tell you that upfront."
  - It calls itself "a safety and wellness monitoring service. It is
    not a medical device".

**Kubo Care** (Bengaluru)
([YourStory, October 2026](https://yourstory.com/2026/10/tech30-kubo-care-mmwave-radar-senior-fall-detection-bengaluru))
[unverified: the research pass read this page once; a second automated
read was refused (HTTP 403), so open it in a browser before quoting].

- **Hardware:** 60 GHz mmWave radar, "mounted on a wall or ceiling".
- **Claims:**
  - It "can detect sudden falls in less than a second while
    distinguishing them from deliberate movements".
  - It turns point clouds into "real-time skeletal models to assess
    posture and gait". That is a vendor claim. Note that a multi-antenna
    60 GHz radar is a different instrument from the single-antenna WiFi
    radio that [`docs/CAPABILITY_LIMITS.md`](CAPABILITY_LIMITS.md) §1 is
    about.
  - A "patented method that reduces false alarms by more than 99%", with
    no study.
- **Stage:** "currently pre-revenue and conducting pilots in India,
  Japan, and the US".
- **Price:** it "plans to charge senior care operators between $50 and
  $70 per device per month". A direct-to-consumer launch is planned.
- **Privacy:** it uses cloud applications, and has a family app.

**Hi-Link HLK-LD2410 and LD2450 modules (24 GHz).** These are what
hobbyists and integrators build with.

- **Claims:**
  - The LD2410 "can identify human bodies in motion and stationary
    states", out to 5 m, with "distance resolution ... 0.75m"
    ([Hi-Link](https://www.hlktech.net/index.php?id=988)).
  - The LD2450 measures "the distance, angle and speed" of a moving
    target ([Hi-Link](https://www.hlktech.net/index.php?id=1157)).
  - Neither claims falls or breathing.
- **Price in India:**
  - An LD2410 is "Rs. 499.00"
    ([Indian Hobby Center](https://www.indianhobbycenter.com/products/hi-link-hlk-ld2410-24ghz-human-presence-body-induction-distance-detection-radar-sensor-module-support-gpio-and-uart-ports);
    stock unclear).
  - An LD2450 sold with an **ESP32-S3** and Home Assistant firmware is
    ₹1,951, in stock
    ([Fab.to.Lab](https://www.fabtolab.com/mmwave-presence-sensor-ld2450-home-assistant)).
- **Privacy:** fully local if the builder wants it.
- **Evidence:** datasheet claims only.

### 5.2 Elsewhere

**Vayyar Care** (Israel; US consumer version and care homes).

- **Claims:**
  - "Touchless fall detection. Exclusively with Alexa Together"; sold
    through Amazon in the US ([Vayyar](https://vayyar.com/care-docs/b2c/)).
  - It detects "both major and minor falls in all lighting conditions
    and even through steam".
  - Range: "13 feet to the front and 6 feet 6 inches either side".
  - "stairwells are not supported at this time".
  - The care-home version sends "immediate alerts when a resident
    experiences a fall" ([Vayyar Care](https://vayyar.com/care/)).
  - The older Walabot Home page now redirects there.
- **Price:** not on the vendor page. No India availability found.
- **Evidence:** the best for any radar here, and small.
  - A lab and hospital test reached "detection accuracy of 92.50 % and a
    sensitivity of 92.50 %", but only with "two Vayyar Home Care devices
    in one room"
    ([Hrubý and Černý, 2023](https://ebooks.iospress.nl/volumearticle/63827)).
  - A nursing-home pilot in Istanbul put the device in 13 residents'
    rooms for 3 months
    ([Can et al., Psychogeriatrics 2024](https://doi.org/10.1111/psyg.13126)).
    - One resident with the device fell. The staff were alerted and
      responded promptly.
    - "The false alarm rate per day was 0.22."
    - Fall alarms without a fall "were generated by bend-over movement,
      either to pick up an object from the floor or during prayer".

**Aqara Presence Sensor FP2** (60 GHz; a consumer product).

- **Claims** ([Aqara](https://www.aqara.com/us/product/presence-sensor-fp2/)):
  - "The FP2 has the capability to detect and alert you when a family
    member falls down".
  - But "In fall detection mode, most features of the FP2 including zone
    positioning and presence detection become unavailable", and fall
    detection needs a ceiling mount.
  - A December 2023 firmware added sleep tracking "as well as breathing
    and heart rate"
    ([Notebookcheck](https://www.notebookcheck.net/Aqara-FP2-presence-sensor-gets-sleep-tracking-and-new-AI-features.782979.0.html)).
  - It is "not a medical device".
- **Price:**
  - US: $82.99 ([Aqara store](https://us.aqara.com/products/presence-sensor-fp2)).
  - India: no official listing. An imported Amazon.in listing ranges
    from ₹7,999 to ₹11,777
    ([price tracker](https://price-history.in/product/aqara-presence-sensor-fp2-bracket-set-OdsjAqG5);
    secondary).
- **Privacy:** "Automations of FP2 are executed without sending to the
  cloud", but only with other Aqara or Apple Home devices.
- **Evidence:** vendor claims only.
- **Sibling:** the FP1E ($49.99, needs a Zigbee hub) detects "a human
  moving or still" and claims no falls
  ([Aqara](https://www.aqara.com/en/product/presence-sensor-fp1e/)).

**Cherish Serenity / Home Hub** (US).

- **Claims:**
  - "We detect a wide variety of falls, ranging from hard falls to
    slumps or when people fall off a bed"
    ([Cherish](https://www.cherishhealth.com/homehub)).
  - It detects posture, and works "within a 40-foot range through walls"
    ([TechCrunch, January 2024](https://techcrunch.com/2024/01/09/cherish-health-serenity)).
- **Price:** "$300 plus $39 per month subscription" (TechCrunch). US
  only.
- **Privacy:** "Cherish AI™ runs at the edge"
  ([AT&T](https://about.att.com/story/2023/cherish-serenity.html)). Its
  sensor list includes microphones.
- **Evidence:** none found.

**Xandar Kardian XK300** (radar; vital signs, not falls).

- **Regulatory:** FDA 510(k)
  [K202464](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K202464).
  It measures heart and breathing rate "in a general care hospital
  environment including nursing homes".
- **What it is not:** it "is not an alarm system for acute or
  life-threatening situations" ([xkcorp](https://xkcorp.com/)).
- **Evidence:** an independent study of 50 clinic patients
  ([Park et al., Sci Rep 2021](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8655004/fullTextXML)).
  Heart-rate limits of agreement were "- 12.78 and 15.04 beats/min".
- **Price:** no device price found.

**Bedside sleep radars.**

- **Somnofy:** an independent study of 35 adults aged 65–83 reported a
  breathing-rate "mean absolute error ≤1.6 cycles per minute", pooled
  across three devices
  ([Ravindran et al., JMIR mHealth 2024](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:39190477&resultType=core&format=json)).
- **Google Nest Hub (2nd gen) Sleep Sensing:** "not intended to
  diagnose, cure, mitigate, prevent, or treat any disease or condition"
  ([Google](https://support.google.com/googlenest/answer/10357288?hl=en)).
- Both work only at the bedside.

### 5.3 What radar does well and badly here

- **Well:**
  - Radar sees presence, including a person sitting still.
  - It can see height, which is how it separates a fall from sitting
    down.
  - It forms no image.
  - The modules are cheap: ₹2,000 buys a radar and an ESP32-S3 together.
- **Badly:**
  - The evidence on falls is thin. The best test used two devices in one
    room, and the nursing-home pilot saw a single fall among the
    residents who had a device.
  - Bending to the floor and prayer caused false fall alarms. In Indian
    homes, where sitting and praying on the floor are common, that
    matters.
- **India:**
  - Little is sold. NoBlink installs in five cities; Kubo Care is in
    pilots; the Aqara FP2 is an import.
  - Most products depend on a cloud, and NoBlink does not say what
    leaves the home.

---

## 6. Other ambient sensors

**Dozee** (Bengaluru; a sensor under the mattress).

- **The home product:**
  - "Shravan" / Dozee One offers "Whatsapp Alerts"
    ([Dozee](https://www.dozeehealth.ai/shravan)).
  - "Introductory offer for Dozee One $425 $345", with a subscription of
    "+$25/mo" or "+$50/mo".
  - It is priced in US dollars, and the page says "Delivery starts dec
    1, 2024", so it may be stale.
- **Requirements and data path:**
  - "Dozee requires trained medical professionals to interpret reports
    and act on alerts" ([dozeehealth.ai](https://www.dozeehealth.ai/)).
  - "The data is sent to the secure Cloud"
    ([FAQ](https://www.dozeehealth.ai/faq)).
- **Evidence:**
  - FDA 510(k)
    [K221555](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K221555)
    ("Dozee VS").
  - In a 706-patient hospital study with company authors, "The
    sensitivity of the Dozee-EWS varied between 67% and 94%"
    ([Frontiers, 2024](https://frontiersin.org/articles/10.3389/fmedt.2024.1436034/full)).
- **The limit:** it works only while the person is in bed. It cannot see
  a fall, or a long lie on the bathroom floor.

**Other bed sensors.**

- **Withings Sleep Analyzer:**
  - US "$199.95"
    ([Withings](https://www.withings.com/us/en/sleep-analyzer)).
  - The prescription version is FDA-cleared as "an aid for diagnosis of
    obstructive sleep apnea"
    ([K231667](https://www.accessdata.fda.gov/cdrh_docs/pdf23/K231667.pdf)).
- **Emfit Care:**
  - Sold for "Monitoring the presence, absence, and return to bed of
    elderly individuals", and "not classified as medical devices"
    ([Emfit Care](https://emfit.care/en/)).
  - It "does not replace an emergency phone!"
- None of these was found on sale in India.

**Butlr Care** (US; thermal, no image).

- **Claims:**
  - It uses "low-resolution thermal sensors"
    ([MIT News](https://news.mit.edu/2025/butlr-creates-smart-buildings-with-privacy-first-sensors-0211)).
  - At launch it flagged "situations where no motion is observed to
    detect potential falls"
    ([Butlr, 2023](https://www.butlr.com/news/butlr-care-press-release)).
    That is the same idea as the long lie.
- **Privacy:** "purely thermal information with no visual or personally
  identifiable data" ([Butlr Care](https://www.butlr.com/solutions/care)).
- **Install:** "wired sensors" on the ceiling. "A typical studio
  apartment requires two sensors".
- **Price:** not published; a "60-day pilot plan" is offered.
- **Evidence:** no accuracy figures found.

**Activity monitoring with PIR, door and plug sensors (UK).**

- **Howz** "uses discreet home sensors and machine learning to build a
  round-the-clock picture of daily routine"
  ([Howz](https://www.howz.com/)). It reports a "42% Reduction in risk
  of care home admission", from research it co-published with an NHS
  trust. That report was not read.
- **Canary Care** offers notifications "when you're concerned a bathroom
  visit is taking longer than it should", with "no cameras, microphones
  or wearables" ([Canary Care](https://www.canarycare.co.uk/)).
- Both are UK-only. They are the established model for the routine
  summary (#39).

**DIY: Home Assistant with Zigbee motion sensors.**

- **Price:**
  - A SONOFF SNZB-03P is "INR 3269" through an import reseller
    ([Ubuy India](https://www.ubuy.co.in/producttr/NH8I1ME5G-sonoff-snzb-03p-zigbee-motion)).
  - It is $18.95 in the US
    ([ameridroid](https://ameridroid.com/collections/vendors/products/sonoff-zigbee-motion-detector-snzb-03p)).
- **Privacy:** "Home Assistant keeps your data local"
  ([Home Assistant](https://www.home-assistant.io/)).
- **The limit:** a PIR sensor "only detects movement" (Fab.to.Lab page,
  §5.1), so it misses a person lying still.

---

## 7. WiFi sensing

The products a household can actually buy detect **motion**, and say
plainly that they do not identify people or locate them. The elder-care
claims (falls, breathing, sleep) come from companies that sell through
ISPs or security firms, or that are not shipping. None of them publishes
a validation.

### 7.1 On sale, motion only

**Comcast Xfinity WiFi Motion** (US).

- **Claims:**
  - It is "designed to detect motion, not identify specific people, and
    does not track the exact location of movement", and is "included
    with Xfinity Internet"
    ([Xfinity](https://www.xfinity.com/hub/smart-home/wifi-motion)).
  - Comcast's own FAQ says it "is not a home security service and is not
    professionally monitored"
    ([FAQ](https://www.xfinity.com/support/articles/wifi-motion-faqs)).
- **Privacy:**
  - "Motion event history is stored in Comcast cloud systems for seven
    days" (hub page).
  - Comcast "may be required to disclose information generated by WiFi
    Motion in response to a valid subpoena" (FAQ).
  - It is "opt-in and disabled by default"
    ([BleepingComputer, 2026](https://bleepingcomputer.com/news/security/comcast-turns-your-xfinity-wifi-into-a-home-motion-detector)).

**Verizon Fios Home Awareness** (US).

- **Claims:** launched in 2022 on "select Fios router models", "with no
  additional cost", using technology "patented by Origin Wireless"
  ([press release, syndicated](https://www.stocktitan.net/news/VZ/verizon-launches-new-tech-to-monitor-activity-on-home-wi-qzflosl4fj7u.html)).
- **Status:** unclear [unverified: Verizon's support page for it no
  longer carries its content].

**Plume HomePass Sense** (sold through ISPs).

- **Claims:** it "turns subscriber-owned devices into sensors to detect
  expected and unexpected movement"
  ([Plume](https://www.plume.com/legal/product-descriptions)).
- **Evidence:** vendor claim only.
- **India:** no Indian ISP was found offering it.

**Signify WiZ SpaceSense** (lights; sold in India under the Philips
brand).

- **Claims:**
  - The lights "determine if there is an object moving in the room"
    ([Signify](https://www.signify.com/global/our-company/news/press-releases/2022/20220916-signify-delivers-new-way-to-automate-your-lights-using-wi-fi-sensing-technology)).
  - "It does not detect people's exact location nor does it recognise
    faces"
    ([Signify UK](https://www.signify.com/en-gb/our-company/news/press-releases/2022/20220916-surprisingly-thoughtful-lighting-for-everyone-signify-brings-to-light-motion-detection-technology-spacesense)).
- **Privacy:** "All the detection data is processed locally without
  going to the Internet or Cloud service" (same page).
- **Install:** "two WiZ lights in the same room, with at least 2m/6.5ft
  separating them"
  ([WiZ](https://www.wizconnected.com/en-us/explore-wiz/spacesense)).
- **Price in India:** "Rs. 1,740" for a downlight with SpaceSense
  ([Philips India](https://in.shop.lighting.philips.com/products/philips-smart-wifi-neo-led-downlight)),
  so a working pair is about ₹3,500.
- **Elder care:** none claimed.

**Linksys Aware** has been discontinued.

- End of life was "04/01/2024"
  ([Linksys](https://www.linksys.com/pages/linksys-product-end-of-life)).
- It cost "$2.99 per month or $24.99 per year"
  ([Engadget, 2019](https://www.engadget.com/2019/10/08/linksys-motion-sensing-velop/)).

### 7.2 Elder-care claims, not on general sale

**Origin Wireless / Origin AI**, now owned by ADT.

- **Status:**
  - ADT bought it on "February 24, 2026": "The purchase price for the
    transaction was $170 million in cash"
    ([ADT](https://news.adt.com/corporate-news/adt-acquires-origin-ai-to-power-ai-sensing-and-ambient-intelligence-for-the-home)).
  - ADT names "applications supporting aging in place" as a possible
    use, and "expects to begin commercializing new offerings" in 2027.
  - Verisure has licensed "certain AI sensing use cases, including for
    age-technology applications"
    ([ADT](https://news.adt.com/corporate-news/adt-origin-ai-and-verisure-sign-long-term-commercial-agreement-to-scale-ai-driven-security)).
- **Earlier claims:** Origin's 2021 elder-care page claimed "the highest
  rate of detecting falls and the lowest false alarm rate in the
  industry", with no figures
  ([Origin](https://originwirelessai.com/a-non-wearable-approach-to-elderly-monitoring-the-value-and-benefits)).
- **The consumer product:** the Hex Home site now redirects to a
  domain-sale page.
- **Evidence:** the best academic evidence in this category is the
  SMARS paper, from the group of Origin's founder (IEEE Trans. Mobile
  Computing, 2021,
  [DOI 10.1109/TMC.2019.2939791](https://doi.org/10.1109/TMC.2019.2939791)).
  - It reports "median absolute error of 0.47 breaths per minute" over
    "6 homes, resulting in 32 nights of data in total"
    ([abstract](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TMC.2019.2939791?fields=title,abstract,authors,year,venue)).
  - It is peer-reviewed, but not independent.

**Cognitive Systems** (Waterloo, Canada): WiFi Motion, and "Caregiver by
Cognitive".

- **WiFi Motion:**
  - Licensed to ISPs.
  - The company claims "Human vitals sensing and sleep analytics"
    ([cognitivesystems.com](https://www.cognitivesystems.com/)).
  - It claims deployment "in more than 20 million homes"
    ([Forbes, 2026](https://www.forbes.com/councils/forbestechcouncil/2026/08/31/beyond-connectivity-how-wi-fi-is-becoming-a-foundation-for-spatial-intelligence/)).
- **Caregiver** ([Caregiver](https://www.cognitivesystems.com/caregiver)):
  - Uses "CarePods" and takes "Less than 10 minutes to set up".
  - "When there is no motion for an extended period, you are notified so
    you can check in if needed."
  - It does not mention falls or breathing.
- **Status and price:** the product site's call to action is "Join the
  Waitlist", and no price is published
  ([caregiverbycognitive.com](https://caregiverbycognitive.com/)).
- **Evidence:** vendor claim only.

**Aerial** (Montreal) **and nami** (Singapore).

- Aerial targets "the elderly and remote caregivers"
  ([aerial.ai](https://www.aerial.ai/)) and publishes no accuracy data.
- nami took a "47% equity stake in Aerial.ai" in June 2025, and planned
  ageing-in-place pilots in Spain, Benelux, the USA, Japan and Singapore
  ([TechNode Global](https://technode.global/?p=101121)).
- nami's "Personal Awareness System" offers presence detection through
  Alarm.com
  ([PR Newswire Asia](https://en.prnasia.com/releases/apac/nami-introduces-agile-security-and-personal-awareness-systems-compatible-with-alarm-com-ahead-of-mwc-2026-523290.shtml)).

**Gamgee Wi-Fi Fall Protection** (Amsterdam).

- **Claims:** crowdfunded in 2024, it claimed to "accurately detect
  falls" by "analyzing disruptions in Wi-Fi signals", with no test data
  ([TechBuzz Ireland](https://techbuzzireland.com/2024/07/16/gamgee-unveils-gamgee-wi-fi-fall-protection-for-elderly-safety/)).
- **Status:** gamgee.com now redirects to a personal profile.

**XZent Solutions, PulsEye** (Pune). This is the one Indian WiFi-CSI
elder-care effort found
([Indian Startup Times, 2026-08-22](https://www.indianstartuptimes.com/interviews/xzent-solutions-how-wi-fi-signals-and-ambient-intelligence-are-enabling-privacy-preserving-health-monitoring-without-cameras-or-wearables/)).

- **Claims:** PulsEye uses "Wi-Fi Channel State Information (CSI)" for
  "Contactless respiration and wellness trend tracking" and for
  "Presence, room occupancy, and mobility monitoring".
- **Privacy:** "only lightweight, encrypted event alerts are
  transmitted" from "local microcontrollers". That describes an
  architecture similar to Saans's (ADR-001).
- **Evidence:** no accuracy figures and no price.
- **Stage:** the company calls PulsEye its "long-term ambient sensing
  engine". The product it is rolling out commercially is BedExit, which
  "uses kinetic load sensors installed under bed legs".

### 7.3 Standards and open source

- **IEEE 802.11bf** (WLAN sensing) is an "Active Standard", published
  2025-09-26 ([IEEE](https://standards.ieee.org/ieee/802.11bf/11574/)).
  It standardises how WiFi devices sense, and says nothing about how
  well a product detects a person.
- **Espressif esp-csi** (Apache-2.0) is the chip vendor's CSI toolkit
  for ESP32-family chips
  ([GitHub](https://github.com/espressif/esp-csi)). It mentions sensing
  "breathing and chewing" and publishes no accuracy figures.
- **ESPectre** (GPLv3) is a "Wi-Fi motion sensing platform for ESP32",
  with Home Assistant, MQTT and Matter front ends
  ([GitHub](https://github.com/francescopace/espectre)).
  - It is plain about its limits: "It does not identify people, count
    them, prove that a room is empty".
  - It does not claim to detect breathing.

### 7.4 Where this market is going

- WiFi motion is being absorbed in two directions:
  - into ISP routers as a free feature (Comcast, Verizon);
  - into security companies (ADT buying Origin; Verisure's licence).
- Standalone consumer kits have mostly gone. Linksys Aware ended, the
  Hex Home and Gamgee sites are gone, and Cognitive's caregiver product
  is a waitlist.
- No Indian ISP was found offering WiFi motion.
  - Searches for Airtel and Jio found nothing.
  - An ACT Fibernet post on "WiFi sensing"
    ([ACT](https://www.actcorp.in/blog/what-is-wifi-sensing)) is about
    network tuning, not people.
  - If one did offer it, it would be free with the router, and motion
    only.

---

## 8. Where Saans would be better, and where worse

Read every "better" below as *by design, unmeasured*. Saans has no
real-world number for any of it, and several products above have years
of deployments behind them.

### Where it could be better

1. **Nothing to wear, charge or press.** The case against pendants and
   watches (§4.4) is the strongest evidence in this file. But radar,
   thermal, camera and WiFi products share this, so it does not set
   Saans apart from its nearest competitors.
2. **It cannot form an image, by construction.** The node has one
   antenna, so it has no angular resolution
   ([`docs/CAPABILITY_LIMITS.md`](CAPABILITY_LIMITS.md) §1). That
   limitation is also a guarantee: the bathroom and the bedroom are
   possible where a camera is not (§3.3). Radar and thermal sensors
   offer the same.
3. **What leaves the home is small and fixed.**
   - Raw CSI stays on the node (ADR-001).
   - The FeatureFrame stays on the home network.
   - By default, an alert carries only its kind and the room (README,
     "Alerts").

   Compare Comcast keeping motion history in its cloud, Kami sending
   video to remote monitors, and Dozee and Kubo Care processing in the
   cloud. WiZ, the Aqara FP2's automations and XZent also say they
   process locally. So this is a difference from some competitors, not
   from all of them.
4. **Its limits are published, and its detectors must report both
   error rates.** Every output may abstain, and every detector reports
   its miss rate next to its false-alarm rate (ADR-005). No commercial
   product in this scan publishes a false-alarm rate for itself, and
   Kami's 99.5% has no false-alarm denominator. The wearable studies in
   §4.4 show what such figures hide. This is a process advantage, and
   it lasts only until Saans has numbers of its own to be judged by.
5. **Its code is open,** so a family, an auditor or a regulator can read
   what it does. Of everything above, only the hobbyist projects in §7.3
   publish their code. None of the elder-care products does.

### Where it is worse

1. **It has no real-world evidence at all.**
   - SafelyYou has observational outcome studies.
   - Vayyar has a lab test and a nursing-home pilot.
   - Origin has a peer-reviewed six-home breathing study.
   - Dozee and Xandar Kardian have FDA clearances for vital signs.
   - NoBlink installs in five cities, by its own site (§5.1), and the
     US alarm services have decades of paying customers.
   - Saans has a simulator. That does not change until the first bench
     session and the first real recording (#45, #46).
2. **It does not detect falls.**
   - In simulation, the possible-collapse detector cannot tell a fall
     from a sit-down, and it ships off (ADR-015, PR #52;
     [`docs/RESULTS.md`](RESULTS.md), "S3 health detectors").
   - Radar can see height, which is how it tells the two apart; Saans
     cannot (§5).
   - The long lie alerts only after N minutes of stillness, so by design
     it is slower than a button press, and slower than products that
     claim to react to the fall itself.
   - In simulation the long lie (#38, PR #60) catches most falls but
     cannot tell them from ordinary stillness. It alerts on nearly
     every person who sits down hard and stays still, on most readers
     and on most people who get into bed quickly, and it misses about
     half of slow slides to the floor
     ([`docs/RESULTS.md`](RESULTS.md), "Long lie: miss rate and false
     alarms together, by N"). So the long lie ships off as well, and it
     is not yet a family alert (#62).
   - Still unmeasured are the movements that fooled the Vayyar pilot:
     bending to the floor, and prayer. A sudden movement followed by
     stillness on the floor is exactly what a long lie looks for. #62
     should include them.
3. **It covers only the sensed rooms of one home.** A watch or a mobile
   pendant goes to the market and the temple with the person. A room
   with no nodes is a room Saans cannot see.
4. **Only family answers an alert.**
   - US alarm services have 24/7 call centres, and Emoha offers 24/7
     emergency coordination.
   - NoBlink calls a chain of people by voice.
   - Saans has a Telegram and webhook alert sink tested only against
     local stand-ins ([`docs/EVIDENCE.md`](EVIDENCE.md) §2.6). The relay
     (#40) and the phone app (#41) do not exist.
5. **Installing it is harder than installing a camera.**
   - It needs two links per room where breathing matters (ADR-006), an
     empty room during calibration (README, "Operating requirements"),
     and a gateway.
   - A Tapo is set up from an app in minutes.
   - NoBlink sends its own installer, which says something about this
     whole category.
6. **The hardware is not a cost advantage.**
   - A devkit is roughly ₹600–900 ([`docs/HARDWARE.md`](HARDWARE.md),
     itself marked unverified), and a home needs several, plus a
     gateway.
   - A 24 GHz radar sold together with an ESP32-S3 costs ₹1,951 in India
     (§5.1). It detects a still person and measures range and angle,
     which a Saans node cannot.
   - Nobody has priced a Saans kit (#44, #42).
7. **It cannot tell people apart, count them, or say where in a room
   someone is** ([`docs/CAPABILITY_LIMITS.md`](CAPABILITY_LIMITS.md)).
   A visitor or a pet looks the same as the person it is meant to watch.
8. **In a flat, a neighbour may read as "present".** A link with no
   range or bearing cannot tell which side of a wall a person is on
   ([`docs/CAPABILITY_LIMITS.md`](CAPABILITY_LIMITS.md) §2d). In Indian
   apartments that is a product risk, and it is unmeasured.
9. **Others in India are already doing the job.**
   - NoBlink sells camera-free stillness and fall alerts, with
     installers in five cities.
   - Kubo Care is piloting radar fall detection.
   - Dozee sends WhatsApp alerts from a bed sensor with FDA-cleared vital
     signs.
   - XZent describes a similar on-device WiFi-CSI architecture,
     at an earlier stage.

   Against NoBlink, Saans's differences today are openness, published
   limits and a privacy boundary written into the code. None of these is
   a reason for a family to choose it before it has worked in a real
   home.

### What would change this section

- Real recordings (#45, #46).
- The long-lie miss rate, and false alerts per home per week (#38).
- A priced kit, and a setup a family can do themselves (#44, #42).
- A relay and an app a family can use (#40, #41).

Until then the honest pitch is narrow: no image, nothing to wear, a
small and fixed privacy boundary, and claims bounded by published
limits. All of it is still to be shown on hardware.

## 9. What this scan did not do

- No product was bought or tested. Every capability above is what a
  vendor or a study says.
- No family, care home or installer was interviewed.
- Amazon, Flipkart and several vendor sites refused automated reads, so
  some Indian street prices are **[unverified]** or missing.
- It covers India and the US, plus European products where they bear on
  the design. It does not cover China, Japan or Korea, where much of the
  hardware is made and some of these categories are larger.
- It is a snapshot. Re-check prices and product status before quoting
  any of them.
