# Research notes: ectopic-beat statistics from a Polar H10 chest strap

This document is the literature review behind the trigger-tag and ectopy-statistics
feature. It answers four questions:

1. Is Polar H10 raw ECG good enough to count ectopic beats at home?
2. What detection approach does the evidence support at 130 Hz on a single lead?
3. How variable is ectopic burden, and what does that mean for interpreting counts?
4. What experimental and statistical design can actually identify *triggers* of
   rare, benign ectopy — the thing wrist-based spot-check wearables cannot do?

Everything here is screening/self-experiment context, **never diagnosis**. A single
chest-strap lead cannot determine the origin of an ectopic beat, and none of the
numbers this app produces are clinical measurements.

> Citation caveat: entries marked **[verify]** were assembled from abstracts or
> search summaries of paywalled sources; double-check the quantitative figures
> against the PDF before repeating them anywhere formal.

---

## 1. Is Polar H10 ECG good enough?

**Yes, for beats and rhythm; carefully, for morphology.** The key results:

- **Skála et al. 2022** (Cor et Vasa 64(4):411–422, doi:10.33678/cor.2022.083) is the
  anchor study: 161 subjects (54 cardiology inpatients, 53 outpatients, 54 controls),
  1–2 h of raw H10 ECG each, **1,153,229 beats** read by cardiologists. Over 97 % of
  beats were rated easy to interpret; only **2.16 %** of signal was noise. Atrial
  premature beats (0.46 %) and ventricular premature beats (0.49 %) were identified
  from the strips, and agreement with in-hospital telemetry was 98.1 %. Cautions:
  paced rhythm and atrial flutter.
- **Gilgen-Ammann et al. 2019** (Eur J Appl Physiol 119:1525, doi:10.1007/s00421-019-04142-5):
  RR signal quality **99.6 % for the H10 vs 94.6 % for a clinical Holter**, and the
  strap held 99.4 % at high exercise intensity while the Holter dropped to 89.8 %.
  The strap is not the weak link.
- **Schaffarczyk et al. 2022** (Sensors 22(17):6536): H10 RR agreement with 12-lead is
  essentially perfect at rest (r = 1.00, bias 0.2 ms). Their stated limitation — that
  RR-only recording "cannot identify or correct" artefact-vs-ectopy ambiguity — is
  precisely the gap raw-ECG logging fills.
- **Choi et al. 2024** (Sensors 24(19):6394): during treadmill stress tests, a
  single-lead wearable matched multi-lead reference with **ICC = 1.0 for ventricular
  and supraventricular ectopic-beat counts and burden**. This is the strongest direct
  evidence that single-lead ectopy *counting* — this app's task — is viable.
- **Lindsey et al. 2025** (Sensors 25(16):5186): across walking, circuit training and
  an obstacle course, the H10 had the **lowest signal-rejection rate** of the straps
  tested and was comparable to a wired reference; rejection still reached ~40 % in
  the worst activity. Motion is the dominant data-loss mechanism.
- Long-term feasibility (**Saggu et al. 2024**, Indian Pacing Electrophysiol J
  24(5):282 [verify]): 12-week chest-strap wear produced diagnostic-quality ECG for
  ~76.5 % of the time, lost ~18.7 % to motion — and arrhythmia yield rose from 24 %
  at 24 h to 64 % over 12 weeks. Notably, *all* device-detected pauses/tachycardias
  in that study were motion artifacts: quality gating matters more than classifier
  sophistication. **Joutsen et al. 2024** (Sci Rep 14:8882): dry electrodes need
  ~10 min to settle after donning; conductive-fabric electrodes are the weakest class
  under motion after 24 h — wet the strap.

**The cautionary tale for RR-only monitoring — Gajda et al. 2018** (Scand J Med Sci
Sports 28:496, doi:10.1111/sms.12917): among endurance athletes re-tested with
simultaneous Holter after their heart-rate monitors flagged "arrhythmias", **~45 % of
alerts were pure artifact and 15 % of true isolated ectopics were missed**. An
RR-only chest strap cries wolf roughly half the time. This is why this app's ectopy
gate requires morphology and motion evidence, not just RR prematurity, and why the
2024 sports-cardiology consensus (**Gajda et al. 2024**, Sports Med 54:1) frames such
tools as *triage and documentation, never standalone diagnosis* — the stance this
app enforces in its wording rules.

**The niche:** no published study reports sensitivity/specificity for PVC/PAC
detection from Polar H10 raw ECG against a Holter reference, and no peer-reviewed
citizen-science chest-strap ectopy-statistics project exists. Skála 2022 shows
humans can read premature beats off H10 strips; Choi 2024 shows single-lead counting
matches multi-lead. The combination — automated home ectopy statistics from H10 raw
ECG — is genuinely unoccupied territory.

## 2. Detection at 130 Hz on one lead

### 2.1 What the sampling rate allows

The H10 samples internally at 1000 Hz but streams ECG at a fixed **130 Hz** (Polar
BLE SDK documentation). One sample every 7.7 ms means:

- **QRS detection is unaffected.** Habib et al. 2020 (arXiv:2007.02052) found ≤0.6 %
  accuracy difference between 100 Hz and 250 Hz for QRS detection. Kraft & Rumm 2026
  (Sensors 26(2):513) run their entire beat classifier **at 125 Hz** and reach
  **PVC-class sensitivity 0.978 / precision 0.956** on MIT-BIH — the decisive
  evidence that ventricular-type morphology discrimination survives this rate.
- **Interval/width measurement is not credible.** Diagnostic-ECG standards call for
  ≥500 Hz (Kligfield et al. 2007, Circulation 115:1306). A 100 ms QRS is ~13
  samples; width thresholds land inside the quantisation error. This is why the app
  computes template-distance and timing descriptors, never QRS width/QT/PR (see the
  pipeline docstring), and why events carry a *compensatory-pause ratio* (pure
  timing) rather than any width feature.
- **Atrial-origin classification is out of reach.** Even a state-of-the-art CNN
  achieves only 0.36–0.80 sensitivity for the atrial-premature class on single-lead
  data, collapsing further in noise (Kraft & Rumm 2026). Combined with the wording
  rules, this app reports **"ectopic beats" without origin** — full stop.

### 2.2 The detection stack this app already had, versus the literature

| Stage | This app | Literature support |
|---|---|---|
| R-peaks | neurokit2 default detector, biosppy fallback | Kristof et al. 2024 (PLOS Digit Health, 18 detectors × 6 databases, doi:10.1371/journal.pdig.0000538): NeuroKit ranked best overall on single-lead telehealth data; also found F1 collapses to ≤0.84 on low-quality signal → a hard quality gate is essential |
| Quality gate | 5 s windows: SQI, clipping, wander, gaps → excluded segments | Same benchmark's headline recommendation; Saggu 2024's all-false-positives-were-motion finding |
| Beat classification | Lipponen & Tarvainen 2019 (J Med Eng Technol 43:173) two-pass via neurokit2 | The Kubios algorithm itself: dRR-pattern classification, **Se 96.96 % / Sp 99.94 %** for real atrial+ventricular ectopics — the best-documented RR-domain classifier |
| Morphology | Median template, per-beat Pearson r, threshold 0.90 | Conventional template-matching cut; personalised template matching is the canonical single-subject approach (Krasteva & Jekova 2007, Ann Biomed Eng 35:2065) |
| Prematurity | preceding RR ≤ −20 % vs local median of 10 | The standard "premature < 80 % of reference RR" rule family [verify: exact operating points vary by study] |
| Confirmation | triple gate: correction-classified ectopic ∧ premature ∧ ¬motion | Matches the multi-evidence designs that keep specificity high; RR-only evidence alone is wrong ~half the time in motion (Gajda 2018) |

Two literature findings shaped the *new* event layer:

- **Rule-based ceilings are fine for this purpose.** A pure rules detector reaches
  Se ~90 %/Sp 99.6 % (Manikandan et al. 2015, Healthc Technol Lett 2(6)); a
  commercial implantable-monitor algorithm runs at Se ~74 %/Sp 99.6+ % and still
  "accurately represented overall burden" (Heart Rhythm O2 2023 [verify]). For
  trigger inference, **stable sensitivity across conditions matters far more than
  high sensitivity** — a detector that misses 25 % of events *uniformly* still
  estimates rate ratios correctly.
- **Pattern grouping uses the classical grading vocabulary**: couplet = 2
  consecutive ectopics, run/salvo ≥ 3 (Lown & Wolf grading tradition, Circulation
  1971;44:130–142), bigeminy/trigeminy as alternation patterns, and the
  **compensatory-pause ratio** (RRpre + RRpost vs 2× local reference; a full pause
  is *typical of* ventricular origin, an incomplete pause of atrial origin — kept as
  a descriptor, never a classification, for the reasons in §2.1).

## 3. Burden variability — the central statistical threat

Ectopic burden fluctuates enormously within a person, which is exactly why "I had
coffee yesterday and more ectopics today" is uninterpretable without design:

- **Morganroth et al. 1978** (Circulation 58:408): comparing two single 24-h
  recordings requires a **>83 % change** in ectopic frequency before it exceeds
  spontaneous variation. Day-to-day variation 23 %, hour-to-hour 48 %.
- **Hamon et al. 2015** (Heart Rhythm 12:2372 [verify]): hourly PVC-burden
  coefficient of variation is **~60 %** in people without cardiomyopathy.
- **Ahn et al. 2024** (J Med Internet Res, doi:10.2196/46098): across 6-hour windows,
  max/min burden varied **12-fold** (IQR 4–58); daily max/min 1.68-fold. Their
  single-lead patch matched a Holter with R² = 0.995 — the variability is
  biological, not instrumental.
- **Mullis et al. 2019** (Heart Rhythm 16:1570 [verify]): over 14 days, the same
  patient's daily burden ranged from a median-min of 4.5 % to a median-max of
  16.2 % around a 9.0 % mean — crossing management-relevant category boundaries on
  sampling luck alone.
- **Måneheim et al. 2024** (Europace 26:euae198): day-to-day ICC 0.76 (atrial) /
  0.87 (ventricular); **≥10–11 days** of monitoring are needed to estimate a
  14-day frequency within ±20 %. Krumerman et al. 2020 (Heart Rhythm): day 1
  explains ~60 % of 14-day burden; day 14, ~88 %.

Consequences implemented in the app:

1. Counts are modelled with a **negative-binomial** distribution (Poisson intervals
   would be absurdly narrow given ~60 % CoV).
2. Every rate uses **analysed time as the exposure denominator** (motion-excluded
   time doesn't count, and it is *not* missing at random with respect to activity).
3. The dashboard shows **confidence intervals, never verdicts**, and states the
   Morganroth context so a two-session comparison is never over-read.
4. Minimum-data guards refuse to fit models on session counts the variability
   literature says are uninterpretable.

## 4. Trigger epidemiology and experiment design

### 4.1 What's actually known about triggers

- **Caffeine.** The **CRAVE trial** (Marcus et al. 2023, NEJM 388:1092,
  doi:10.1056/NEJMoa2204737) is the template: 100 adults, 14 days, coffee assigned
  in randomized 2-day blocks by daily text, continuous Zio-patch ECG. Result:
  **PVCs +51 % on coffee days (RR 1.51, CI 1.18–1.94); PACs null (RR 1.09, CI
  0.98–1.20)**. Also: coffee days had +1,058 steps and −36 min sleep — a mediation
  ambiguity a home experiment inherits (was it the caffeine, or the extra activity
  and shorter sleep?). Contrast **Dixit et al. 2016** (JAHA 5:e002503): chronic
  caffeine consumption showed *no* between-person association with ectopy in 1,388
  people. **Within-person randomized designs find effects that observational
  between-person designs miss** — the core argument for this app's design.
- **Alcohol.** I-STOP-AFib (Marcus et al. 2022, JAMA Cardiol 7:167) found alcohol
  OR 1.77 for next-day AF in N-of-1 testing; the HOLIDAY infusion study suggests
  the mechanism runs through substrate/withdrawal, i.e. **effects may appear the
  next morning, not the same hour**. Session tags for alcohol should be interpreted
  with that lag in mind.
- **Sleep.** A CRAVE secondary analysis found **no** relationship between objective
  sleep and next-day ectopy in healthy volunteers [verify: PMID 42172865]; an
  inpatient natural experiment (Rosen et al. 2016, SLEEP 39:927) found sleep
  disruption associated with ~33 % more hourly ventricular ectopy carrying into the
  next day. Expect weak, lagged, individually variable effects.
- **Exercise & posture.** Exercise-phase and recovery-phase ectopy are
  physiologically distinct (catecholaminergic vs vagal-reactivation; JACC 2021
  78:2267) — which is why the activity profile is a covariate here, and why
  "lying down" is in the tag vocabulary (classic bradycardia/vagal facilitation of
  ectopy when supine). The sign of the HR–ectopy relationship is *unstable
  day-to-day within a person* (consistent in only ~37 % of patients across days
  [verify: PMC12758478]).
- **Stress.** Anger is the best-documented emotional precipitant of ventricular
  arrhythmia (Lampert et al. 2002, Circulation 106:1800) — captured here as a tag.
- **Circadian rhythm.** Group-level ectopy peaks late morning/afternoon, but only
  ~47–62 % of individuals show a significant *individual* circadian rhythm
  (Am J Cardiol 1990, doi:10.1016/0002-9149(90)90512-Y). **Any caffeine analysis
  that ignores time of day rediscovers the circadian rhythm instead** — hence the
  hour-of-day covariate in the model.

The built-in tag vocabulary (caffeine, alcohol, poor sleep, stress, large meal,
dehydration, lying down, illness, exercise earlier) is the I-STOP-AFib self-selected
trigger menu plus CRAVE's exposure.

### 4.2 Why a wrist watch can't do this (the premise)

Consumer wrist devices run **spot-check architectures**: periodic tachogram sampling
at rest (Apple Heart Study, Perez et al. 2019, NEJM 381:1909 — 0.52 % notification
rate, PPV 0.84 *for AF*). Burden estimation needs a continuous denominator of
analysed beats, and infrequent ectopy is bursty on exactly the timescales spot
checks miss. When wrist devices do notice ectopy, they tend to misreport it as
AF-like irregularity (Circ Arrhythm Electrophysiol 2022,
doi:10.1161/CIRCEP.121.010063). Continuous chest-strap ECG plus per-session
statistics is the correct instrument for the job.

### 4.3 The statistical design implemented here

Given the user's choice of **session-level tags** (rather than timestamped
exposures), the unit of analysis is the recording session:

- **Model:** one negative-binomial regression per tag —
  `ectopy_count ~ tag + sin(hour) + cos(hour) [+ activity]`, with
  `offset = log(analysed hours)`. NB2 with MLE-estimated dispersion; where the NB
  fit fails to converge, a Poisson GLM with robust (HC1) errors is the fallback and
  is labelled as such. Rate ratio = exp(β_tag) with Wald 95 % CI.
- **Why per-tag models:** with N-of-1 sample sizes, a joint model over all tags is
  routinely rank-deficient. The multiple-comparison cost is disclosed in the UI.
- **Why hour-of-day:** §4.1's circadian confound. A sin/cos pair spends 2 degrees
  of freedom, affordable sooner than binned hours.
- **Guards** (each cited in `app/triggers/stats.py`): ≥5 tagged and ≥5 untagged
  sessions, ≥10 total ectopics, ≥10 min analysed per session — below these, the
  Morganroth/Hamon variability numbers say an estimate is noise, and the UI says
  "insufficient data" instead of printing one.
- **Interpretation stance,** shown on the dashboard: tags are exposures *reported
  by the wearer*, not randomized assignments, so results are associations. For a
  step up in rigor, the literature's recommendation is randomizing exposure in
  blocks (CRAVE's 2-day blocks; N-of-1 methodology review: median 6 cycles over
  ~77 days, Hawksworth et al. 2024, Trials 25; reporting per CENT 2015,
  BMJ 350:h1738) — the app's session tags support exactly this usage pattern:
  assign yourself coffee/no-coffee blocks, tag honestly, let the weeks accumulate.

## 5. Design decisions mapped to sources

| App decision | Grounding |
|---|---|
| Keep raw-ECG triple gate; never RR-only flags | Gajda 2018 (45 % artifact alerts); Schaffarczyk 2022 (RR-only can't separate artifact/ectopy) |
| Say "ectopic beats", never origin, never diagnosis | Kraft & Rumm 2026 (atrial class Se 0.36–0.80 on single lead); Kligfield 2007 (130 Hz can't do widths); Gajda 2024 consensus (triage, not diagnosis) |
| Event grouping: singles/couplets/runs/bi-trigeminy | Lown & Wolf grading tradition |
| Compensatory-pause ratio as descriptor only | Classical full-vs-incomplete pause physiology; single-lead origin-calling unsupported |
| Rates over analysed time, not wall time | Saggu 2024 (~19 % motion loss; not missing-at-random); Kristof 2024 (quality gate) |
| Negative binomial + CIs, no significance verdicts | Hamon 2015 (CoV ~60 %); Ahn 2024 (12× 6-h swings); Hawksworth 2024 (report CIs) |
| Hour-of-day covariate | Circadian ectopy rhythm (Am J Cardiol 1990; Hamon 2015) |
| Minimum-data guards | Morganroth 1978; Måneheim 2024 (≥10–11 days for ±20 % burden) |
| Session tags from a fixed vocabulary | I-STOP-AFib trigger menu; CRAVE |
| Multi-session trending as the product | Krumerman 2020 (day 1 = 60 % of truth); Turakhia 2013 (yield grows for days) |

## 6. Honest limitations

- Sensitivity of the triple-gated detector on this specific hardware is
  **unvalidated against a clinical reference** — no such study exists for the H10.
  Counts are internally consistent (the same gate every session), which is what the
  rate-ratio design needs, but absolute burden numbers are not clinical measurements.
- Session-level tags cannot resolve within-day timing (e.g. same-hour caffeine
  effects vs next-morning alcohol effects); lags are approximated by how the wearer
  chooses to tag.
- The hour-of-day covariate uses the recording timestamp as stored (UTC-derived);
  wearers who record across time zones should read the trend pages accordingly.
- Ectopy statistics are association-generating, not causal, unless the wearer
  self-randomizes exposures in blocks — the dashboard says so.

## 7. Bibliography

Device validation & feasibility
- Skála T, et al. *Feasibility of evaluation of Polar H10 chest-belt ECG in patients with a broad range of heart conditions.* Cor et Vasa 2022;64(4):411–422. doi:10.33678/cor.2022.083
- Schaffarczyk M, Rogers B, Reer R, Gronwald T. *Validity of the Polar H10 sensor for HRV analysis…* Sensors 2022;22(17):6536. doi:10.3390/s22176536
- Gilgen-Ammann R, Schweizer T, Wyss T. *RR interval signal quality of a heart rate monitor and an ECG Holter at rest and during exercise.* Eur J Appl Physiol 2019;119:1525–1532. doi:10.1007/s00421-019-04142-5
- Vermunicht P, et al. *Validation of Polar H10 chest strap… in cardiac patients.* Europace 2023;25(Suppl 1):euad122.550. doi:10.1093/europace/euad122.550 [verify]
- Lindsey B, et al. *Activity type effects signal quality in electrocardiogram devices.* Sensors 2025;25(16):5186. doi:10.3390/s25165186
- Bläsing D, et al. *ECG performance in simultaneous recordings of five wearable devices…* PLOS ONE 2022;17(9):e0274994. doi:10.1371/journal.pone.0274994
- Saggu DK, et al. *Feasibility of using chest strap and dry electrode system for longer term cardiac arrhythmia monitoring.* Indian Pacing Electrophysiol J 2024;24(5):282–290. [verify]
- Joutsen A, et al. *ECG signal quality in intermittent long-term dry electrode recordings with controlled motion artifacts.* Sci Rep 2024;14:8882. doi:10.1038/s41598-024-56595-0
- Miyazaki Y, et al. *AF after ischemic stroke detected by chest strap-style 7-day Holter monitoring.* J Atheroscler Thromb 2021;28:544–554. doi:10.5551/jat.58420 [verify]
- Rogers B, et al. *The Movesense Medical sensor chest belt… validation.* Sensors 2022;22(5):2032. doi:10.3390/s22052032
- Polar Electro. *Polar H10 heart rate sensor white paper* (130 Hz BLE ECG stream; 1000 Hz internal RR).

Ectopy detection on single-lead ECG
- Hartikainen S, et al. *Effectiveness of the chest strap electrocardiogram to detect atrial fibrillation.* Am J Cardiol 2019;123:1643–1648. doi:10.1016/j.amjcard.2019.02.028
- Santala OE, et al. *Automatic mobile health arrhythmia monitoring for the detection of atrial fibrillation.* JMIR mHealth Uhealth 2021;9(10):e29933. doi:10.2196/29933
- Choi H-I, et al. *Efficacy of wearable single-lead ECG monitoring during exercise stress testing.* Sensors 2024;24(19):6394. doi:10.3390/s24196394
- Orini M, et al. *Premature atrial and ventricular contractions detected on wearable-format electrocardiograms and prediction of cardiovascular events.* Eur Heart J Digit Health 2023;4(2):112–118. doi:10.1093/ehjdh/ztad007
- Kraft D, Rumm P. *Automated detection of normal, atrial, and ventricular premature beats from single-lead ECG using CNNs.* Sensors 2026;26(2):513. doi:10.3390/s26020513
- Cuesta P, et al. *Detection of premature ventricular contractions using the RR-interval signal.* Technol Health Care 2014;22(4):651–656. doi:10.3233/THC-140818
- Krasteva V, Jekova I (Christov et al.). *QRS template matching for recognition of ventricular ectopic beats.* Ann Biomed Eng 2007;35(12):2065–2076. doi:10.1007/s10439-007-9368-9
- Manikandan MS, et al. *Robust detection of premature ventricular contractions using sparse signal decomposition and temporal features.* Healthc Technol Lett 2015;2(6). doi:10.1049/htl.2015.0006
- Cai Z, et al. *Robust PVC identification by fusing expert system and deep learning.* Biosensors 2022;12(4):185. doi:10.3390/bios12040185 [verify]
- *Evaluation of a novel PVC and PAC detection algorithm in an implantable cardiac monitor.* Heart Rhythm O2 2023. PMID 37744934 [verify]
- Lipponen JA, Tarvainen MP. *A robust algorithm for heart rate variability time series artefact correction using novel beat classification.* J Med Eng Technol 2019;43(3):173–181. doi:10.1080/03091902.2019.1640306
- Gajda R, Biernacka EK, Drygas W. *Are heart rate monitors valuable tools for diagnosing arrhythmias in endurance athletes?* Scand J Med Sci Sports 2018;28(2):496–516. doi:10.1111/sms.12917
- Gajda R, et al. *Sports heart monitors as reliable diagnostic tools… expert consensus statement.* Sports Med 2024;54(1):1–21. doi:10.1007/s40279-023-01948-4
- Kristof F, et al. *QRS detection in single-lead, telehealth electrocardiogram signals: benchmarking open-source algorithms.* PLOS Digit Health 2024. doi:10.1371/journal.pdig.0000538
- Porr B, Howell L. *R-peak detector stress test with a new noisy ECG database…* bioRxiv 2019:722397.
- Pan J, Tompkins WJ. *A real-time QRS detection algorithm.* IEEE Trans Biomed Eng 1985;32(3):230–236. — Hamilton PS. *Open source ECG analysis.* Comput Cardiol 2002;29:101–104.
- Zhao Z, Zhang Y. *SQI quality evaluation mechanism of single-lead ECG signal…* Front Physiol 2018. (note: ectopic segments must be excluded from SQI computation or true beats get discarded)
- Kligfield P, et al. *Recommendations for the standardization and interpretation of the electrocardiogram, Part I.* Circulation 2007;115:1306–1324. doi:10.1161/CIRCULATIONAHA.106.180200
- Habib A, Karmakar C, Yearwood J. *Choosing a sampling frequency for ECG QRS detection using convolutional networks.* arXiv:2007.02052 (2020).
- de Chazal P, O'Dwyer M, Reilly RB. *Automatic classification of heartbeats using ECG morphology and heartbeat interval features.* IEEE Trans Biomed Eng 2004;51(7):1196–1206. (AAMI EC57 inter-patient protocol)
- Tan S, et al. *Icentia11K: an unsupervised representation learning dataset for arrhythmia subtype discovery.* PhysioNet / arXiv:1910.09570. — Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* IEEE EMB Mag 2001;20(3):45–50.
- Lown B, Wolf M. *Approaches to sudden death from coronary heart disease.* Circulation 1971;44:130–142. (grading vocabulary: couplets, salvos)

Burden variability
- Morganroth J, et al. *Limitations of routine long-term electrocardiographic monitoring to assess ventricular ectopic frequency.* Circulation 1978;58(3):408–414. doi:10.1161/01.CIR.58.3.408
- Hamon D, et al. *Effect of circadian variability in frequency of premature ventricular complexes on left ventricular function.* Heart Rhythm 2015;12(11):2372–2379. doi:10.1016/j.hrthm.2015.08.005 [verify]
- Mullis AH, et al. *Fluctuations in premature ventricular contraction burden can affect medical assessment and management.* Heart Rhythm 2019;16(10):1570–1574. doi:10.1016/j.hrthm.2019.04.027 [verify]
- Ahn HJ, et al. *Three-day monitoring of adhesive single-lead ECG patch for premature ventricular complex.* J Med Internet Res 2024. doi:10.2196/46098
- *PVC variability and impact on meeting expert consensus cutoffs…* medRxiv 2024.06.10.24308734. [preprint]
- Krumerman A, et al. *Determining the optimal duration for premature ventricular contraction monitoring.* Heart Rhythm 2020;17(12):2119–2125. doi:10.1016/j.hrthm.2020.07.015 [verify]
- *Premature ventricular complexes: assessing burden density in a large national cohort…* Heart Rhythm 2024;21(8):1289–1295. doi:10.1016/j.hrthm.2024.03.007 [verify]
- Måneheim A, et al. *Diagnostic reliability of monitoring for premature atrial and ventricular complexes.* Europace 2024;26(8):euae198. doi:10.1093/europace/euae198
- Karlsson M, et al. *Automatic filtering of outliers in RR intervals…* Biomed Eng OnLine 2012;11:2. — Salo MA, et al. *Ectopic beats in heart rate variability analysis.* Ann Noninvasive Electrocardiol 2001. PMID 11174857

Triggers & statistical design
- Marcus GM, et al. *Acute effects of coffee consumption on health among ambulatory adults* (CRAVE). N Engl J Med 2023;388(12):1092–1100. doi:10.1056/NEJMoa2204737
- Dixit S, et al. *Consumption of caffeinated products and cardiac ectopy.* J Am Heart Assoc 2016;5:e002503. doi:10.1161/JAHA.115.002503
- Voskoboinik A, Kalman JM, Kistler PM. *Caffeine and arrhythmias: time to grind the data.* JACC Clin Electrophysiol 2018;4(4):425–432.
- Marcus GM, et al. *Individualized studies of triggers of paroxysmal atrial fibrillation* (I-STOP-AFib). JAMA Cardiol 2022;7(2):167–174. doi:10.1001/jamacardio.2021.5010
- HOLIDAY trial protocol (alcohol infusion, NCT01996943).
- *Nightly sleep as a predictor of next-day arrhythmias in ambulatory adults* (CRAVE secondary). J Electrocardiol 2025. PMID 42172865 [verify]
- Rosen Y, et al. *Sleep disruption is associated with increased ventricular ectopy and cardiac arrest in hospitalized adults.* SLEEP 2016;39(4):927–935. doi:10.5665/sleep.5656 [verify]
- *Exercise-induced ventricular ectopy and cardiovascular mortality in asymptomatic individuals.* JACC 2021;78(23):2267–2277. doi:10.1016/j.jacc.2021.09.1366 [verify]
- Lampert R, et al. *Emotional and physical precipitants of ventricular arrhythmia.* Circulation 2002;106(14):1800–1805.
- *Reproducibility in circadian rhythm of ventricular premature complexes.* Am J Cardiol 1990. doi:10.1016/0002-9149(90)90512-Y [verify]
- Maclure M. *The case-crossover design.* Am J Epidemiol 1991;133:144–153.
- Tobias A, Kim Y, Madaniyazi L. *Time-stratified case-crossover studies for aggregated data in environmental epidemiology: a tutorial.* Int J Epidemiol 2024;53(2):dyae020. doi:10.1093/ije/dyae020
- Hawksworth O, et al. *A methodological review of randomised n-of-1 trials.* Trials 2024;25. doi:10.1186/s13063-024-08100-1
- Shamseer L, et al. *CONSORT extension for reporting N-of-1 trials (CENT) 2015.* BMJ 2015;350:h1738.
- *Negative binomial mixed effects location-scale models for intensive longitudinal count-type data from wearable devices.* Biometrics 2025;81(3):ujaf099. [verify]
- Gardner W, Mulvey EP, Shaw EC. *Regression analyses of counts and rates.* Psychol Bull 1995;118:392–404.
- Perez MV, et al. *Large-scale assessment of a smartwatch to identify atrial fibrillation* (Apple Heart Study). N Engl J Med 2019;381:1909–1917. doi:10.1056/NEJMoa1901183
- *Arrhythmias other than atrial fibrillation in those with an irregular pulse detected with a smartwatch.* Circ Arrhythm Electrophysiol 2022. doi:10.1161/CIRCEP.121.010063 [verify]
- Turakhia MP, et al. *Diagnostic utility of a novel leadless arrhythmia monitoring device.* Am J Cardiol 2013;112(4):520–524. doi:10.1016/j.amjcard.2013.04.017

Open-source / ecosystem
- Broman K. *detectPVC* (R, Polar H10 PVC detection; Zenodo doi:10.5281/zenodo.11174768) and *AndroidPolarPVC2* (Zenodo doi:10.5281/zenodo.11626183); blog posts on personal PVC logging (12–28 % day-to-day burden variation observed).
- Makowski D, et al. *NeuroKit2: a Python toolbox for neurophysiological signal processing.* Behav Res Methods 2021;53:1689–1696. — wfdb-python, BioSPPy, sleepecg, systole, hrv-analysis, py-ecg-detectors.
- fsmeraldi/bleakheart (async Python BLE for Polar ECG); Polar Sensor Logger; sokolmarek/hrv-correction (Lipponen & Tarvainen implementation).
- Kardi Ai × Polar (MDR class IIa certified arrhythmia screening on H10; accuracy trial NCT07018648 unreported at time of writing).
