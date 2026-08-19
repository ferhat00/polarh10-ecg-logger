"""The built-in trigger-tag vocabulary.

The core list is the self-selected trigger menu of the I-STOP-AFib N-of-1
trial (Marcus et al., JAMA Cardiol 2022;7:167-174 — caffeine, alcohol,
reduced sleep, exercise, lying down, dehydration, large meals) plus the CRAVE
trial's exposure (Marcus et al., N Engl J Med 2023;388:1092-1100), with
stress and illness added as the commonly reported remainder. See
docs/RESEARCH.md.

The second block extends it with evidence-backed HRV/RHR exposures — the
literature grounding for each is in docs/CONTEXT_METRICS.md: nicotine
(acute HF-HRV reduction), sauna and cold exposure (acute autonomic phase
effects), breathwork (paced ~6/min breathing mechanically inflates RMSSD —
this tag is an honesty flag as much as an exposure), late meal (circadian
phase shift), travel/jet lag (clock-timestamp misalignment), noisy sleep
environment (Schmidt 2013, Eur Heart J), menstruation (Schmalenberger 2019
meta-analysis; bleeding-days marker, deliberately not a phase claim), and
vaccination (transient RHR elevation for days, Quer 2022).

Tags describe exposures in the hours before or during a recording. They are
session-level: the statistics compare sessions with a tag against sessions
without it, so tag honestly and consistently.
"""

from __future__ import annotations

#: (slug, display name), in display order.
BUILTIN_TRIGGER_TAGS: tuple[tuple[str, str], ...] = (
    ("caffeine", "Caffeine"),
    ("alcohol", "Alcohol"),
    ("poor-sleep", "Poor sleep"),
    ("stress", "Stress"),
    ("large-meal", "Large meal"),
    ("dehydration", "Dehydration"),
    ("lying-down", "Lying down"),
    ("illness", "Illness"),
    ("exercise-earlier", "Exercise earlier"),
    ("nicotine", "Nicotine"),
    ("sauna", "Sauna"),
    ("cold-exposure", "Cold exposure"),
    ("breathwork", "Breathwork / paced breathing"),
    ("late-meal", "Late meal"),
    ("travel-jetlag", "Travel / jet lag"),
    ("noisy-sleep-env", "Noisy sleep environment"),
    ("menstruation", "Menstruation"),
    ("vaccination", "Vaccination"),
)
