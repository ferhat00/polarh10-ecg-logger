"""The built-in trigger-tag vocabulary.

The list is the self-selected trigger menu of the I-STOP-AFib N-of-1 trial
(Marcus et al., JAMA Cardiol 2022;7:167-174 — caffeine, alcohol, reduced
sleep, exercise, lying down, dehydration, large meals) plus the CRAVE trial's
exposure (Marcus et al., N Engl J Med 2023;388:1092-1100), with stress and
illness added as the commonly reported remainder. See docs/RESEARCH.md.

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
)
