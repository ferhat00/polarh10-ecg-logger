# Polar Flow sync (AccessLink)

This document is the design grounding for the opt-in Polar Flow feature: what
the AccessLink API gives, what it deliberately does not touch, how the data
relates to what the H10 pipeline measures, and how to set it up. Everything
here is context for self-tracking, **never diagnosis** — nothing from Flow
reaches the screening rules, and §6 explains why that is a design decision
rather than an omission.

---

## 1. Why this exists, and what it is not

The question that led here was whether raw data can be logged off a **Polar
Loop Gen 2**. It can, but not usefully for this app:

- Polar's BLE SDK documents Loop alongside Polar 360 with raw PPG (22 Hz,
  24-bit, two photodiodes off the central green LED), ACC (50 Hz, ±8g, up to
  400 Hz in SDK mode), PPI, and skin temperature — as an online stream and as
  offline recording into the band's 15 MB.
- But the SDK is **Android/iOS only**. Polar's answer to a developer driving
  the PMD service directly from React Native was that support exists only for
  the native SDKs, and touching the PMD service on a 360/Loop disconnected the
  device outright.
- The Loop bonds to **exactly one peer**. Once paired with the Polar Flow app
  it is invisible to every other scanner, and pairing overwrite is refused by
  design as an anti-spoofing measure. Logging raw data means factory-resetting
  the band away from Flow — it is genuinely either/or.
- There is **no ECG**. It is a wrist PPG band. The R-peak detection, beat
  morphology, ectopy triple-gate, and rhythm strips this app is built on have
  nothing to consume.

AccessLink is the other route: what Flow has already derived from the band,
over plain HTTPS, in Python, with Flow left working. It is device-agnostic —
it returns whatever the linked Flow account holds, so it covers the Loop, any
Polar watch, and H10 sessions recorded through Polar Beat.

## 2. What the API gives

Verified against the OpenAPI spec at
`https://www.polar.com/accesslink-api/swagger.yaml` (v3).

| Endpoint | Returns |
| --- | --- |
| `GET /v3/users/sleep` | the last 28 nights, one request |
| `GET /v3/users/sleep/{date}` | `sleep_start_time`, `sleep_end_time`, `device_id`, `light_sleep`, `deep_sleep`, `rem_sleep`, `unrecognized_sleep_stage` (seconds), `sleep_score`, `continuity`, `continuity_class`, `total_interruption_duration`, `sleep_charge`, `sleep_cycles` |
| `GET /v3/users/sleep/available` | which dates hold data — cheap gap discovery |
| `GET /v3/users/nightly-recharge` | the last 28 nights, one request |
| `GET /v3/users/nightly-recharge/{date}` | `heart_rate_avg`, `beat_to_beat_avg` (ms), `heart_rate_variability_avg`, `breathing_rate_avg`, `nightly_recharge_status` (1–6), `ans_charge` (−10…+10), `ans_charge_status` (1–5), `hrv_samples` and `breathing_samples` (5-minute maps) |
| `GET /v3/users/continuous-heart-rate?from=&to=` | per-day `heart_rate_samples[]` of `{heart_rate, sample_time}` |
| `GET /v3/users/biosensing/skintemperature?from=&to=` | `sleep_time_skin_temperature_celsius`, `deviation_from_baseline_celsius` — allowlisted but not yet consumed; whether Loop populates it is unverified |

`heart_rate_variability_avg` is documented by Polar in as many words as **the
Root Mean Square of Successive Differences (RMSSD)**, in milliseconds,
averaged over a four-hour period starting 30 minutes after sleep onset.

**There is no beat-to-beat interval series in this API.** `beat_to_beat_avg`
is a single nightly mean. Anything needing an RR series still needs the H10.

## 3. The transaction endpoints, and why they are unreachable

AccessLink has two families of endpoints:

- **Non-transactional** — everything in §2. Plain GETs, re-readable as often
  as you like.
- **Transactional** — `exercise-transactions` and `activity-transactions`.
  These follow create → read → **commit**, and committing **deletes the data
  server-side**. One careless call permanently destroys the wearer's ability
  to re-fetch it.

`app/polar/client.py` therefore holds an explicit allowlist
(`ALLOWED_PATH_PREFIXES`) and **raises** `DisallowedEndpointError` on anything
outside it. Raising rather than noting is deliberate: reaching a transaction
endpoint is a programming error, not a runtime condition to shrug off. A test
asserts the refusal for each transaction path.

This is why the feature does not offer exercise or activity import. Those live
only behind the destructive family, and the trade — permanently consuming the
data to read it once — is not one this tool should make on the user's behalf.

## 4. Comparability: Flow's RMSSD is not this app's RMSSD

Both are called RMSSD and both are in milliseconds. They are different
measurements:

| | Session RMSSD | Flow overnight RMSSD |
| --- | --- | --- |
| Signal | chest ECG, R-peaks at ~130 Hz | wrist PPG, pulse-to-pulse |
| Window | minutes, controlled posture | ~4 h of sleep |
| Timing source | ventricular depolarisation | peripheral pulse arrival |
| Artifact handling | this app's Kubios-style correction, reported | Polar's, opaque |

Pulse arrival adds pulse-transit-time variability that R-peaks do not carry,
and Polar's PPI algorithm filters beats on its own quality criteria. The two
may diverge without either being wrong.

The app therefore **never merges them**. The trends page draws overnight RMSSD
as a separate aligned figure beneath the session timeline — never a dual axis,
following the same rule the environment figures already obey — with a caption
saying not to read the two as one series. In the trigger statistics they are
separate outcomes (`ln_rmssd` and `flow_rmssd`), never pooled.

## 5. What it is good for

The payoff is coverage. Session HRV exists only for nights you strapped on the
H10; Flow's exists for **every** night. That gives the trigger models a
denominator that does not depend on remembering to record, and it turns the
subjective `sleep_quality_1_5` field and the `poor-sleep` tag into something
checkable against a measurement.

The subjective field is **not** auto-filled from the objective one.
`CONTEXT_METRICS.md` §2.11 keeps them separate because their divergence is
itself informative. Both are displayed; neither is derived from the other.

## 6. Screening interaction: none, deliberately

Nothing from Flow enters `app/screening`. Every threshold there is rule-based
on measured ECG and cited in `screening/thresholds.py`. Sleep score, Nightly
Recharge status, and ANS charge are proprietary composites whose internals are
not published — putting one behind a flag that carries a medical disclaimer
would mean asserting something unauditable. They are context, displayed as
context.

## 7. Setup

1. **Register an API client.** Sign in at
   `https://admin.polaraccesslink.com` with the Polar Flow account and create
   a client. Set the authorization redirect URL to
   `http://localhost:5000/polar/callback`.

   > The redirect URL must match **byte for byte**. `localhost` and
   > `127.0.0.1` are different registrations even though they reach the same
   > server — and Flask's own start-up message prints `127.0.0.1:5000`. If
   > you browse to the app on `127.0.0.1`, either register that spelling too
   > or set `ECGLOG_POLAR_REDIRECT_URI` to whichever one you registered.

2. **Opt in.** The feature is off by default; with `ECGLOG_POLAR_ENABLED`
   unset the routes are not registered and no request to Polar is ever made.

   ```bash
   export ECGLOG_POLAR_ENABLED=1
   export ECGLOG_POLAR_CLIENT_ID=<client id>
   export ECGLOG_POLAR_CLIENT_SECRET=<client secret>
   # optional: ECGLOG_POLAR_REDIRECT_URI, ECGLOG_POLAR_TIMEOUT_S (default 10)
   ```

3. **Link a person.** Open the person page, press *Link Polar Flow*,
   authorise in Polar Flow, and you land back on the person page. Linking
   registers the account with the client; a `409 already registered` is the
   normal outcome of re-linking and is treated as success.

4. **Sync.** Press *Sync last 28 nights*, or:

   ```bash
   flask --app wsgi polar-sync
   flask --app wsgi polar-sync --person <slug> --from 2026-01-01 --to 2026-08-21
   ```

## 8. Request budgeting

The per-user rate limit is **500 requests per 15 minutes** and 5000 per 24 h
(both scale with registered user count). A naive year-long backfill at one
request per date per endpoint is ~1100 requests and will 429.

The sync therefore:

- uses the **list** endpoints for the last 28 days — two requests, regardless
  of how many nights come back;
- for older dates, asks `/v3/users/sleep/available` first and requests only the
  dates Polar says it holds;
- stops cleanly on 429, keeps everything that arrived before the wall, and
  reports the date it stopped at rather than silently truncating.

Historical depth beyond the documented 28-day list window is **not documented**
by Polar. By-date fetches reach further; the CLI reports where it ran dry
rather than asserting a number.

## 9. Credentials at rest

This is the one place the app stores a secret that is not its own.

- **Client id and secret** identify the *application* and live in environment
  variables only — never the database, never the repository.
- **The access token** identifies the *wearer's Polar account* and is stored in
  the local SQLite file, in plaintext, on the `polar_account` row. It has to
  survive restarts and it is per-person, so there is nowhere else for it to go
  under a single-local-machine threat model.

AccessLink issues **no refresh token**, and its access tokens do not expire
unless revoked. Treat one as a long-lived read credential for the Flow
account:

- *Unlink* on the person page deletes the stored token here and keeps the
  nights already synced.
- Revoking at Polar's end is separate: `https://account.polar.com`.
- A sync that gets a 401 deletes the stored token automatically rather than
  retrying a dead credential on every future run.

## 10. Limitations

- **28-day list window**; older data is reachable but at one request per date,
  and to an undocumented depth.
- **No raw samples.** Sleep stages, scores, and nightly averages are Polar's
  own derived output. If you want raw PPG or ACC off the band, that is the
  native-SDK route in §1, with the costs listed there.
- **No exercise or activity import**, by the reasoning in §3.
- **Wrist, not chest.** Every number here is measured by a device on the
  non-dominant wrist, and §4 applies to all of it.
- **Sleep skin temperature is allowlisted but unconsumed.** Whether Loop Gen 2
  populates the Elixir biosensing endpoints is unverified; probe it before
  building on it.
