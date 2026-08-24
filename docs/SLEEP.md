# Sleep-stage analysis from an H10 chest strap

This document is the design and literature grounding for the sleep feature:
what heart-beat-based staging can and cannot deliver, which engines this app
runs, how to enable each one, and the honest accuracy numbers. Everything
here is a screening/self-tracking context, **never diagnosis** — the report
says so on every night.

> Citation caveat (same convention as `RESEARCH.md`): entries marked
> **[verify]** were assembled from abstracts or search summaries; check the
> quantitative figures against the PDF before repeating them anywhere formal.

---

## 1. Can heartbeats measure sleep stages at all?

Yes, imperfectly, with a known ceiling. Sleep stages are defined by EEG, but
each stage has an autonomic signature readable from beat-to-beat intervals:
slow-wave (deep) sleep is the night's vagal maximum — lowest heart rate,
highest respiratory sinus arrhythmia, LF/HF minimum — while REM shows
sympathetic predominance (LF/HF maximum, elevated irregular HR) and wake
shows movement plus HR near waking levels (Bušek et al. 2005, Physiol Res
54(4):369–376; Vanoli et al. 1995, Circulation 91(7):1918–1922).

The ceiling: cardio(-actigraphy)-only staging plateaus around **κ≈0.6 /
~77 % epoch agreement** for 4-class staging even with deep learning trained
on hundreds of nights (Radha et al. 2019, Sci Rep 9:14149,
doi:10.1038/s41598-019-49703-y). The best result validated **on this exact
strap** is Sleep² (below) at 80.3 %. Human inter-scorer agreement on PSG
itself is ~82–83 % — heart-beat staging approaches but does not reach it,
and per-stage errors are not uniform: Wake and N1 are the weak classes.

## 2. The Sleep² reference point — and why it isn't an engine here

**Sleep² / NUKKUAA™** (University of Salzburg spin-off) is the
multi-resolution CNN of **Topalidis et al. 2023** (Sensors 23(5):2390,
doi:10.3390/s23052390): 4-class Wake/Light/Deep/REM from interbeat
intervals, trained on 8,898 PSG nights, validated against PSG with a
**Polar H10** at **80.3 % accuracy (κ≈0.69)** — the number this feature
treats as the honest ceiling for the hardware.

The model, weights, and training data are **proprietary**. This app cannot
run or reimplement it (reimplementation would need PSG corpora at that
scale plus retraining). What it adopts instead:

* the **4-class Wake/Light/Deep/REM vocabulary** as the canonical display
  convention,
* the citation as the **accuracy ceiling**, printed in the report next to
  every engine's own accuracy note.

## 3. The engines

Every engine scores the same canonical grid — 30 s epochs from the ECG
recording start — and returns a hypnogram with a mandatory accuracy note.
All computed hypnograms are shown; the **primary** engine — best available
(external > sleepecg > heuristic) unless you chose one for that night
(§3.4) — fills the queryable per-night columns (TST, efficiency, SOL, WASO,
stage minutes, awakenings). When several engines ran, the report shows
epoch-by-epoch agreement (Cohen's κ); low agreement means the night's
numbers are a range, not a value.

| Engine | Stages | Setup | Accuracy grounding |
| --- | --- | --- | --- |
| Built-in heuristic | Wake/Light/Deep/REM | none — always available | transparent cited rules + Viterbi smoothing; expect ~65–75 % (below both trained models) |
| SleepECG GRU (`wrn-gru-mesa-weighted`) | Wake/REM/NREM | `pip install -r requirements-sleep.txt` | trained on 1,971 MESA nights, evaluated on SHHS (Brunner et al. 2023, JOSS 8(86):5411, doi:10.21105/joss.05411) |
| External deep net (adammj/ecg-sleep-staging) | Wake/N1/N2/N3/REM | separate AGPL clone + env vars (below) | reported median κ≈0.725; per-stage medians Wake 0.87, N1 0.33, N2 0.68, N3 0.63, REM 0.83 **[verify]** |
| *(reference only)* Sleep² / Topalidis 2023 | Wake/Light/Deep/REM | proprietary — cannot run | 80.3 %, κ≈0.69 on the H10 |

### 3.1 Built-in heuristic (always on)

Per-epoch features from the pipeline's corrected RR series: mean HR, 5-min
windowed RMSSD and LF/HF band power (band-passed 4 Hz tachogram, computed
per contiguous run — never across excluded gaps), HR relative to the
night's sustained minimum, and movement (accelerometer counts when
attached, else an ECG baseline-wander proxy). Cited rule scores are
smoothed by a Viterbi pass over a sticky transition prior (self-transition
~0.9/epoch; structure per the stage-bout literature, Kishi et al. 2008, Am
J Physiol Regul Integr Comp Physiol 294(6):R1980 **[verify]**).
Deterministic; every threshold is a named constant in
`app/sleep/engines/heuristic.py`.

### 3.2 SleepECG (optional extras)

```bash
.venv/Scripts/python -m pip install -r requirements-sleep.txt
```

The classifiers ship **inside the sleepecg wheel** — nothing is downloaded
at analysis time, preserving the app's offline stance. TensorFlow is the
inference backend; `tensorflow-cpu` is sufficient (staging takes seconds on
CPU). On native Windows the CPU wheel is Intel's `tensorflow-intel`,
installed automatically — WSL2 is only needed for GPU, which this doesn't
use. Inputs: heartbeat times, the recording's local clock time (circadian
features), and — when set on the person profile — age and sex, the
covariates the classifier was trained with. Missing covariates are
tolerated (NaN features) at a small accuracy cost; the report notes it.

### 3.3 External 5-class engine (optional, AGPL, separate install)

[`adammj/ecg-sleep-staging`](https://github.com/adammj/ecg-sleep-staging)
is **AGPL-3.0**; this repository is MIT. The engine therefore runs as a
**separate program in its own clone and environment** — this app writes an
`input.h5` to the tool's documented interface, invokes it as a subprocess,
and reads back `results.h5`. Nothing from the AGPL codebase is imported,
vendored, or copied.

Setup:

```bash
git clone https://github.com/adammj/ecg-sleep-staging /path/to/clone
# create ITS environment per ITS README (PyTorch etc.), then point this app at it:
export ECGLOG_SLEEP_EXTERNAL_DIR=/path/to/clone/your_own_data
export ECGLOG_SLEEP_EXTERNAL_PYTHON=/path/to/clone-venv/bin/python
# optional: ECGLOG_SLEEP_EXTERNAL_TIMEOUT_S (default 1800)
```

The adapter (`app/sleep/engines/external_ecg_staging.py`) implements the
tool's published input contract: 0.5 Hz high-pass, mains notch, resample to
256 Hz after filtering, median → 0, per-heartbeat 90th-percentile amplitude
scaled to ±0.5, clamp [−1, 1], epochs of 7,680 samples, demographics
`[sex, age/100]`, and a ±12 h midnight offset
(cardiosomnography.com/requirements). The results-file dataset name and the
exact `midnight_offset` convention are pinned by adapter tests against a
stub scorer but should be **verified against the real tool on first use**
— any mismatch surfaces as an engine note naming what was found, never as
a failed session.

### 3.4 Choosing an engine and re-analysing a night

The ranking above is a default, not a lock-in. Every logged night keeps its
raw ECG (and ACC) file, so it can be re-staged at any time — which matters
most after installing the optional extras, since nights recorded before
that are otherwise stuck on the heuristic forever.

* **One night:** the *Sleep algorithm* card on the session page. Pick an
  engine and press *Re-analyse this night*.
* **A history:** the **Sleep** page (`/sleep`) lists every staged night.
  Tick the ones you want, pick one algorithm, re-analyse them together.
  They are queued and processed **one at a time** — the external engine
  runs a separate deep network per night, so a long history is hours of
  work, not minutes.

Three things worth knowing:

1. **Every available engine still runs.** The choice decides which one
   fills the night's numbers and leads the report; the others stay in the
   agreement table. Dropping them to save time would remove the only
   signal that two algorithms disagree about the same night.
2. **The choice sticks to the session** and is re-applied every time that
   night is re-analysed, from anywhere.
3. **Switching to a coarser engine blanks stage minutes, by design.**
   SleepECG scores Wake/NREM/REM, so choosing it leaves *light* and *deep*
   empty rather than inventing a split it cannot see — the same
   vocabulary-honesty rule as §5. Switch back and they return.

Engines that cannot run here are shown in the dropdown, greyed out, with
the exact reason (a missing package, an unset environment variable). An
unavailable engine is a fact to report, not an error — and if the chosen
engine fails or disappears between choosing and running, the night is
staged by the next-best engine and the report says so.

## 4. Accelerometer (optional second upload)

Polar Sensor Logger can export the H10's accelerometer stream alongside the
ECG. Attach it at upload and the app computes per-epoch **activity counts**
(vector magnitude, 0.25–3 Hz band-pass, rectified and integrated per 30 s —
the raw-accelerometry convention of te Lindert & Van Someren 2013, Sleep
36(5):781–789, doi:10.5665/sleep.2648). Movement feeds three things:

1. the heuristic engine's wake evidence,
2. a **wake override** on every engine's hypnogram: ≥ 2 consecutive epochs
   above a per-night robust threshold (median + 5·MAD) are re-scored to
   Wake — still-body wakefulness remains invisible, which is exactly the
   published weakness of cardio-only staging,
3. the report's movement trace.

Without the ACC file, movement falls back to ECG baseline wander — a coarse
proxy the report flags as such.

## 5. Definitions used for the summary numbers

Per AASM conventions, each an epoch count × 0.5 min: **TIB** = the scored
grid (recording start ≈ lights-off — start the recording when you intend to
sleep); **sleep onset** = first epoch of any sleep stage; **TST** = sleep
epochs; **efficiency** = TST/TIB (normative reference Ohayon et al. 2017,
Sleep Health 3(1):6–19); **WASO** = wake strictly between onset and last
sleep; **awakenings** = wake runs ≥ 1 epoch in that span (sub-30 s arousals
are invisible at this resolution); **REM latency** = onset → first REM.
Stage minutes are vocabulary-honest: a 3-class engine reports NREM and
never invents a light/deep split.

## 6. Screening interaction

Nocturnal HR normally dips 10–30 % below waking rest; sinus rates in the
40s are unremarkable in healthy sleepers and in the 30s in trained athletes
(Kusumoto et al. 2018 ACC/AHA/HRS guideline, Circulation 2019;140:e382;
Sharma et al. 2017, JACC 69:1057). Sleep sessions therefore screen
bradycardia at **40 bpm (35 athlete)** instead of the resting thresholds —
otherwise every healthy night would flag. Sustained tachycardia >100 bpm
remains flagged. Whole-night frequency-domain and nonlinear HRV are
*cautioned* (stages violate stationarity by design); the per-stage table in
the report is the readable version.

## 7. Limitations (also printed in every report)

* Stages are estimated from heartbeats, not brain activity; even the
  H10-validated ceiling is ~80 % epoch agreement.
* Cannot detect sleep apnea, periodic limb movements, or distinguish quiet
  wakefulness from sleep misperception (the insomnia failure mode).
* Wake and N1 are systematically the least reliable classes for every
  heart-beat-based method.
* Recording span stands in for time in bed; starting the strap long before
  lights-off inflates TIB and deflates efficiency.
* Changing the engine changes the *estimate*, not the night. Two engines
  disagreeing on the same recording is the κ≈0.6–0.7 ceiling in action, not
  a bug — the agreement table is where to look before trusting a number.
* Nothing here is a diagnosis or a substitute for polysomnography.
