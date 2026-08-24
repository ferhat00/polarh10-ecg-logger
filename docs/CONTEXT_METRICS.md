# Research notes: contextual metrics for HRV/ECG recordings

This document is the literature review behind the contextual-metrics features:
the expanded trigger-tag vocabulary, the structured per-session context fields,
the opt-in weather/air-quality capture, and the sensor-derived context (posture,
ECG-derived respiration). It answers three questions:

1. Which environmental exposures have documented effects on HRV, resting heart
   rate, or sleep — and which of them can be fetched automatically from a date
   and location instead of typed in?
2. Which lifestyle exposures are worth tagging, and what effect sizes does the
   literature support?
3. What additional context can the Polar H10 itself derive, beyond the HR/RR/ECG
   channels the app already uses?

Everything here is screening/self-experiment context, **never diagnosis**. The
statistical caveats of `docs/RESEARCH.md` §3–4 (burden variability, circadian
confounding, session-level tag limitations, association-not-causation) apply to
every factor below; they are not repeated per item.

Capture-path legend: **Auto** = fetched from the opt-in Open-Meteo lookup
(date + configured home location), **Sensor** = derived from the H10 recording
itself, **Tag** = binary trigger tag, **Field** = structured per-session column,
**Note** = free-text context note only.

---

## 1. Environmental exposures

### 1.1 Air quality — the strongest environmental literature — *Auto*

- **Gold et al. 2000** (Circulation 101:1267, doi:10.1161/01.CIR.101.11.1267):
  the landmark elderly-panel study — ambient PM2.5 and ozone were associated
  with reduced SDNN and r-MSSD in repeated 25-minute protocols.
- **Brook et al. 2010** (Circulation 121:2331, doi:10.1161/CIR.0b013e3181dbece1):
  the AHA scientific statement formalising particulate air pollution →
  autonomic dysfunction as one of the mechanistic pathways to cardiovascular
  events.
- **Wang et al. 2020** (Chemosphere 261:127635,
  doi:10.1016/j.chemosphere.2020.127635): meta-analysis in older adults —
  roughly −0.4 % SDNN and −1.2 % RMSSD per 10 µg/m³ short-term PM2.5, larger
  for long-term exposure. (This meta-analysis is often mis-cited as "Niu et
  al."; the Crossref record resolves to Wang.)

Small percent-level effects — visible in trends, not in any single session. The
app records PM2.5/PM10/O₃/NO₂/European AQI window-averaged over the recording
and offers descriptive comparison only (see §4 guard rails in the dashboard).

### 1.2 Ambient temperature and humidity — *Auto*

- Experimental heat exposure lowers HRV primarily through parasympathetic
  withdrawal and raises heart rate (review: IJERPH 2021, PMID 34073134
  [verify]); repeated-measures panel work associates short-term temperature
  *variability* itself with HRV perturbation (Environ Res 2020 [verify]).
- Cold exposure findings are protocol-dependent (acute pressor/sympathetic
  response vs. transient vagal activation in facial/whole-body cooling); no
  single direction is asserted in the UI.
- Humidity is captured as a modifier, not a standalone exposure: the app stores
  relative humidity and Open-Meteo's apparent temperature (which folds in
  humidity and wind) rather than making independent humidity claims.

### 1.3 Season and daylight — *Auto*

- **Kristal-Boneh et al. 2000** (J Cardiovasc Risk 7:141,
  doi:10.1177/204748730000700209): 24-h HRV is lower in winter than in summer
  in repeated measurements of the same workers.
- Muscle sympathetic nerve activity peaks in winter (Cui et al., seasonal MSNA
  studies [verify]). Daylight duration for the recording date is stored so
  seasonal context is visible in trends without the user tagging anything.

### 1.4 Barometric pressure — *Auto, logged without claims*

Evidence is weak and inconsistent: a small altitude study associated falling
pressure with falling sleeping HR (PMID 29282538 [verify]); an ambulatory-BP
study linked pressure with BP variability (Am J Hypertens 2002 [verify]).
Pressure is stored because it arrives free with the same API call — the app
deliberately offers **no inference** about it.

### 1.5 Nighttime noise — *Tag (`noisy-sleep-env`)*

- **Schmidt et al. 2013** (Eur Heart J 34:3508, doi:10.1093/eurheartj/eht269):
  nighttime aircraft noise impaired endothelial function and raised stress-
  hormone levels in field exposure; noise events during sleep raise heart rate
  and fragment sleep (IJERPH 2019, doi:10.3390/ijerph16020269 [verify]).

No free API can know a bedroom's noise, so this stays a manual tag for nights
the wearer knows were noisy (construction, storms, snoring partner, street
noise).

### 1.6 Not captured: altitude, pollen

Altitude/hypoxia effects are real (acute ↑RHR, ↓vagal HRV during
acclimatisation) but a home-location app cannot know travel altitude; wearers
who record at altitude should say so in the context note. Pollen has weak and
directionally inconsistent HRV evidence and Open-Meteo's pollen coverage is
Europe-only — excluded rather than half-supported.

## 2. Lifestyle exposures

### 2.1 Alcohol — *Field (`alcohol_drinks_24h`) + Tag*

The best-documented lifestyle exposure, with a clean dose-response:

- WHOOP cohort (~21,000 users, >5 million nights; PLOS Digit Health,
  doi:10.1371/journal.pdig.0001284): each additional drink ≈ **+2.4–2.8 bpm
  sleeping heart rate and −3.3–3.8 ms overnight RMSSD**, effects larger in
  women and younger users.
- **Pietilä et al. 2018** (JMIR Ment Health 5:e23, doi:10.2196/mental.9519):
  ~4,000 real-world users — dose-dependent suppression of vagal HRV in the
  first hours of sleep even at moderate doses.
- **Altini & Plews 2021** (Sensors 21:7932, doi:10.3390/s21237932): across ~9
  million HRV4Training measurements, alcohol is among the largest acute
  stressors visible at population scale.
- I-STOP-AFib (Marcus et al. 2022 — already reviewed in `RESEARCH.md` §4.1)
  adds the next-morning lag caveat: alcohol effects often appear the following
  day, not the same hour.

Because dose matters, alcohol gets a structured drinks-count field, not just a
binary tag.

### 2.2 Caffeine — *Tag (existing)*

Kept as a binary tag; deliberately **no** dose/timing field. The CRAVE trial
(RESEARCH.md §4.1) motivates the tag for *ectopy*; for HRV the evidence is
surprisingly weak — a 2013 systematic review found at most modest vagal-HF
increases (Koenig et al., J Caffeine Res 2013 [verify]) and a 2024 meta-analysis
found no significant effect of dose or timing on post-exercise HRV recovery
(PMID 38494935 [verify]). Late intake matters mainly through sleep disruption,
which the sleep metrics already capture.

### 2.3 Nicotine — *Tag (`nicotine`)*

Acute nicotine reduces HF-HRV in non-users (PMID 21350044 [verify]); chronic
smokers and e-cigarette users show sympathetic-dominant profiles
(meta-analysis [verify]). Direction: ↓HRV, ↑RHR.

### 2.4 Training load — *Tag (existing `exercise-earlier`) + activity profiles*

- **Plews et al. 2013** (Sports Med 43:773, doi:10.1007/s40279-013-0071-8):
  the canonical review of HRV in elite-athlete monitoring — hard sessions
  acutely suppress next-morning vagal HRV; chronic aerobic load raises the
  baseline; interpretation needs rolling averages, not single days.
- Buchheit 2014 (Front Physiol 5:73 [verify]) for the broader
  monitoring framework.

### 2.5 Illness and vaccination — *Tags (`illness`, `vaccination`)*

- **Mishra et al. 2020** (Nat Biomed Eng 4:1208, doi:10.1038/s41551-020-00640-6):
  smartwatch RHR/steps/sleep detected COVID-19 infection in ~81 % of cases,
  frequently days before symptom onset — elevated resting HR is a real,
  personal-baseline-relative infection signal.
- **Quer et al. 2022** (npj Digit Med 5:49, doi:10.1038/s41746-022-00591-z):
  COVID vaccination measurably raises resting HR, peaking ~day 2 and
  normalising by ~day 6 — a several-day window in which HRV/RHR readings are
  expected to deviate and should not be over-read. Hence a dedicated tag.

### 2.6 Stress and breathwork — *Tags (existing `stress`, new `breathwork`)*

- Psychological stress lowers HRV (Kim et al. 2018, Psychiatry Investig 15:235,
  doi:10.30773/pi.2017.08.17).
- **Laborde et al. 2022** (Neurosci Biobehav Rev 138:104711,
  doi:10.1016/j.neubiorev.2022.104711): meta-analysis of voluntary slow
  breathing — acute HRV increases are robust; lasting resting-HRV gains are
  smaller and mainly in RMSSD.
- The `breathwork` tag exists chiefly as an **honesty flag**: paced breathing
  near ~6 breaths/min inflates RMSSD and HF power for mechanical
  (respiratory-sinus-arrhythmia) reasons, so a session recorded during
  breathwork must not be compared against free-breathing baselines. The EDR
  module (§3.2) detects sustained slow breathing automatically and adds a
  caution even when the tag was forgotten.

### 2.7 Meals — *Tags (existing `large-meal`, new `late-meal`)*

Late mealtimes shift the circadian phase of cardiac autonomic markers
(Physiol Behav 2013 [verify]) and eating close to bedtime associates with
worse sleep in part of the literature (Stress Health 2021 [verify]; nocturnal
awakening association [verify]). Evidence is modest; the tag is cheap.

### 2.8 Sauna and cold exposure — *Tags (`sauna`, `cold-exposure`)*

Sauna acutely suppresses HRV during exposure with a parasympathetic rebound in
recovery (Complement Ther Med 2019 [verify]); an RCT found regular post-exercise
sauna did **not** raise resting HRV [verify]. Cold-water immersion meta-analysis
shows enhanced post-exposure parasympathetic activity (J Therm Biol 2024
[verify]). Tags capture the acute-phase context so an unusual reading has an
explanation on record.

### 2.9 Menstrual cycle — *Tag (`menstruation`)*

- **Schmalenberger et al. 2019** (J Clin Med 8:1946, doi:10.3390/jcm8111946):
  meta-analysis across 37 studies — vagal HRV falls from the follicular to the
  luteal phase, Hedges g ≈ −0.39; follow-up work implicates progesterone
  (Schmalenberger et al. 2020, doi:10.3390/jcm9030617).

A bleeding-days tag makes no phase claims; full phase tracking (cycle-day
input or cycle-app import) is deferred — see §5.

### 2.10 Travel and jet lag — *Tag (`travel-jetlag`)*

Circadian misalignment reduces nocturnal vagal activity — higher social jetlag
associates with lower workday pNN50 [verify] — and RESEARCH.md §6 already notes
the time-zone caveat for the hour-of-day covariate. The tag marks sessions where
the internal clock and the recording timestamp disagree.

### 2.11 Sleep quality (subjective) — *Field (`sleep_quality_1_5`)*

The objective sleep metrics (TST, efficiency, WASO…) come from the sleep
engines; the subjective 1–5 rating is kept **separate** because
subjective-objective divergence is itself informative and the existing
`poor-sleep` tag conflates the two. Sleep-regularity/social-jetlag metrics
computable from staged nights are future work.

The opt-in Polar Flow sync ([`POLAR_FLOW.md`](POLAR_FLOW.md)) adds a third
source: the wrist device's own sleep score and Nightly Recharge for the night
each session followed, available even for the nights with no ECG recording.
It is displayed beside this field and **never fills it** — that would collapse
exactly the divergence this separation exists to preserve.

### 2.12 Dehydration, posture-as-tag — *existing tags*

Already in the vocabulary (RESEARCH.md §4.1). The `lying-down` tag is now
complemented by the measured `body_position` field (§3.1), which supersedes it
for new recordings; the tag remains for backward compatibility.

### 2.13 Medications — *Note*

Beta-blockers raise HRV; tricyclic antidepressants lower it strongly; SSRIs are
roughly neutral (quantitative review [verify]). A structured medication model is
out of scope for a session-level tagger — medication context belongs in the
person-level annotation log, and absolute HRV values of a medicated wearer
should never be compared against population numbers. Documented here so the
limitation is on record.

## 3. Sensor-derived context from the Polar H10

### 3.1 Body position from the accelerometer — *Sensor (`body_position`)*

Body position is arguably the **largest within-subject HRV confounder**: supine
recordings show markedly higher vagal indices than sitting or standing
(orthostatic-testing review: **Schneider et al. 2024**, Eur J Appl Physiol,
doi:10.1007/s00421-024-05601-4), so uncontrolled posture invalidates day-to-day
comparison. Chest-worn accelerometers classify lying postures with high
accuracy in research settings (~88–99 % for supine/prone/left/right
[verify: arXiv:2006.10931; IEEE EMBC 2022]).

The app classifies trunk orientation per 30 s epoch from the H10's 200 Hz
accelerometer (gravity vector via low-pass filter), reports time-in-posture,
and auto-fills the session's body-position field when a lying posture clearly
dominates. Upright sitting cannot be distinguished from standing by a chest
accelerometer, and a flipped strap swaps left/right — both stated in the UI
disclaimer.

### 3.2 ECG-derived respiration — *Sensor (`resp_rate_*`)*

- **Schaffarczyk et al. 2022** (Sensors 22:7156, doi:10.3390/s22197156):
  ECG-derived respiratory frequency from *this exact hardware* correlated
  r = 0.85 with gas-exchange measurement across an exercise ramp, with wide
  limits of agreement at high intensity — accuracy is best at rest.
- Charlton et al. 2016 (Physiol Meas 37:610 [verify]) for the two-channel
  (rate + amplitude modulation) estimation-and-fusion approach the
  implementation follows.
- The H10's underlying signal quality (Gilgen-Ammann 2019, RESEARCH.md §1)
  supports beat-accurate modulation extraction.

Respiratory rate is reported as an **estimate** (median and p5–p95 across
quality-gated windows), never as measured airflow. Windows during high HR are
dropped per Schaffarczyk's intensity caveat. Sustained rates in the paced-
breathing band (~4.5–7.5 brpm) trigger the RMSSD-inflation caution of §2.6.
A drifting nightly respiratory-rate baseline is one of the stronger
illness-detection signals (Mishra 2020, §2.5) — trend-level use is future work.

### 3.3 Already implemented elsewhere

DFA α1 (aerobic-threshold literature: Rogers et al. 2021, Front Physiol,
doi:10.3389/fphys.2020.596567 — note the DOI carries a 2020 article ID),
orthostatic ΔHR (activity profile), accelerometer actigraphy for sleep
(docs/SLEEP.md), and rhythm-feasibility grounding (Skála 2022, RESEARCH.md §1).

## 4. The opt-in environment lookup

The app's default remains **fully offline** — no network call is ever made
unless the wearer explicitly opts in (`ECGLOG_WEATHER_ENABLED=1` plus a home
latitude/longitude). When enabled, one lookup per processed session fetches
from [Open-Meteo](https://open-meteo.com/) (free, no API key, historical
archive + recent forecast + air-quality endpoints):

hourly temperature, apparent temperature, relative humidity, surface pressure;
daily daylight duration; hourly PM2.5, PM10, O₃, NO₂, European AQI — each
window-averaged over the recording span and stored on the session row. Missing
data (regional air-quality gaps, network failure) is a first-class outcome:
fields stay empty, processing never fails, and the report says what was and
wasn't fetched. A backfill command (`flask env-backfill`) fills history after
opting in.

Dashboard guard rails: environment variables are offered as **descriptive
context only** (scatter + rank correlation + tertile summary, minimum 12
sessions with data) — no regression covariate, because the effect sizes in §1
are percent-level and a personal dataset of dozens of sessions cannot support
them honestly.

## 5. Deliberately excluded (and why)

| Idea | Why it is excluded |
|---|---|
| QT/QTc, QRS width, PR, axis | Banned by the app's wording/safety rules: 130 Hz sampling cannot support credible interval measurement (Kligfield 2007; RESEARCH.md §2.1). Applies to any "QT tracking" feature request. |
| Seismocardiography from the chest ACC | No published Polar-H10-specific SCG validation; 200 Hz strap-mounted sensing is unproven for cardiac-mechanics timing. Research-grade only. |
| Core-temperature estimation (ECTemp) | Buller et al. 2013 (Physiol Meas 34:781, doi:10.1088/0967-3334/34/7/781) estimates core temp from HR via Kalman filter, validated in athletes [verify] — but it needs population calibration and exercise context; out of scope for a screening app. |
| Pressure/pollen **inference** | Evidence too weak/inconsistent (§1.4, §1.6). Pressure is logged without claims; pollen not fetched. |
| Caffeine dose/timing field | CRAVE's exposure is binary coffee-day assignment; HRV dose evidence is weak (§2.2). The binary tag plus context note suffice. |
| Menstrual **phase** tracking | Phase claims need cycle-day arithmetic or app import plus per-person cycle length; a mislabeled phase is worse than a bleeding-days tag. Deferred. |
| Noise/altitude auto-capture | No API can know a bedroom or a trip (§1.5–1.6). Manual tag / context note. |

## 6. Bibliography

Environment
- Gold DR, et al. *Ambient pollution and heart rate variability.* Circulation 2000;101:1267–1273. doi:10.1161/01.CIR.101.11.1267
- Brook RD, et al. *Particulate matter air pollution and cardiovascular disease: an update to the scientific statement from the AHA.* Circulation 2010;121:2331–2378. doi:10.1161/CIR.0b013e3181dbece1
- Wang X, et al. *Short-term and long-term exposure to PM2.5 and heart rate variability in older adults: a systematic review and meta-analysis.* Chemosphere 2020;261:127635. doi:10.1016/j.chemosphere.2020.127635
- Kristal-Boneh E, et al. *Summer-winter differences in 24 h variability of heart rate.* J Cardiovasc Risk 2000;7:141–146. doi:10.1177/204748730000700209
- Schmidt FP, et al. *Effect of nighttime aircraft noise exposure on endothelial function and stress hormone release in healthy adults.* Eur Heart J 2013;34:3508–3514. doi:10.1093/eurheartj/eht269
- Heat exposure and HRV review, IJERPH 2021, PMID 34073134 [verify]; temperature variability panel, Environ Res 2020 [verify]; sleeping HR vs pressure at altitude, PMID 29282538 [verify]; aircraft-noise sleep events, doi:10.3390/ijerph16020269 [verify]

Lifestyle
- *Association of alcohol consumption with sleep and cardiovascular measures* (WHOOP cohort). PLOS Digit Health. doi:10.1371/journal.pdig.0001284
- Pietilä J, et al. *Acute effect of alcohol intake on cardiovascular autonomic regulation during the first hours of sleep.* JMIR Ment Health 2018;5(1):e23. doi:10.2196/mental.9519
- Altini M, Plews D. *What is behind changes in resting heart rate and heart rate variability? A large-scale analysis of longitudinal measurements acquired in free-living.* Sensors 2021;21:7932. doi:10.3390/s21237932
- Plews DJ, et al. *Training adaptation and heart rate variability in elite endurance athletes: opening the door to effective monitoring.* Sports Med 2013;43:773–781. doi:10.1007/s40279-013-0071-8
- Mishra T, et al. *Pre-symptomatic detection of COVID-19 from smartwatch data.* Nat Biomed Eng 2020;4:1208–1220. doi:10.1038/s41551-020-00640-6
- Quer G, et al. *Inter-individual variation in objective measure of reactogenicity following COVID-19 vaccination via smartwatches and fitness bands.* npj Digit Med 2022;5:49. doi:10.1038/s41746-022-00591-z
- Kim HG, et al. *Stress and heart rate variability: a meta-analysis and review of the literature.* Psychiatry Investig 2018;15:235–245. doi:10.30773/pi.2017.08.17
- Laborde S, et al. *Effects of voluntary slow breathing on heart rate and heart rate variability: a systematic review and meta-analysis.* Neurosci Biobehav Rev 2022;138:104711. doi:10.1016/j.neubiorev.2022.104711
- Schmalenberger KM, et al. *Menstrual cycle changes in vagally-mediated heart rate variability are driven by progesterone: evidence from a meta-analysis.* J Clin Med 2019;8:1946. doi:10.3390/jcm8111946 — and *…within-person analysis* 2020;9:617. doi:10.3390/jcm9030617
- Koenig J, et al., J Caffeine Res 2013 [verify]; caffeine post-exercise HRV meta-analysis 2024, PMID 38494935 [verify]; nicotine HF-HRV, PMID 21350044 [verify]; Buchheit M, Front Physiol 2014;5:73 [verify]; social jetlag & pNN50 [verify]; meal-timing circadian shift, Physiol Behav 2013 [verify]; sauna acute HRV, Complement Ther Med 2019 [verify]; CWI meta-analysis, J Therm Biol 2024 [verify]; antidepressants & HRV quantitative review [verify]

Sensor-derived
- Schneider C, et al. *Heart rate monitoring in team sports — the orthostatic test as a monitoring tool: a systematic review.* Eur J Appl Physiol 2024. doi:10.1007/s00421-024-05601-4
- Schaffarczyk M, Rogers B, Reer R, Gronwald T. *Validation of a non-linear index of heart rate variability and ECG-derived respiration from the Polar H10 during incremental exercise.* Sensors 2022;22:7156. doi:10.3390/s22197156
- Charlton PH, et al. *An assessment of algorithms to estimate respiratory rate from the electrocardiogram and photoplethysmogram.* Physiol Meas 2016;37:610 [verify]
- Rogers B, et al. *A new detection method defining the aerobic threshold for endurance exercise and training prescription based on fractal correlation properties of heart rate variability.* Front Physiol 2021;11:596567. doi:10.3389/fphys.2020.596567
- Buller MJ, et al. *Estimation of human core temperature from sequential heart rate observations.* Physiol Meas 2013;34:781. doi:10.1088/0967-3334/34/7/781
- Chest-accelerometer lying-posture classification, arXiv:2006.10931 [verify]; sleep-posture EMBC 2022, PMID 36086102 [verify]
- (Device validation refs — Skála 2022, Gilgen-Ammann 2019, Schaffarczyk 2022 Sensors 22(17):6536 — in `docs/RESEARCH.md` §7.)
