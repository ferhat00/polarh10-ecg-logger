"""The screening-flag object and the language rules it enforces.

Non-negotiables enforced here, at construction time, so no code path or
template change can violate them:

* Every flag carries a non-empty disclaimer — it is part of the flag object,
  not a page footer.
* Flag descriptions must use screening language ("consistent with",
  "screening threshold exceeded", "pattern worth noting") and may never use
  diagnostic phrasing. Banned phrasings raise ``ValueError`` at construction.
* "No flags raised" renders via :data:`NO_FLAGS_STATEMENT`, which says a
  threshold was not crossed — never that the person is fine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: The disclaimer attached to every flag. Part of the flag object by design.
DISCLAIMER = (
    "Screening observation from a single-lead chest strap — not a diagnosis, and this "
    "tool cannot make one. If a pattern is persistent, symptomatic, or worrying, take "
    "the recording to a clinician."
)

#: How "no flags" must be rendered. Never "you're fine".
NO_FLAGS_STATEMENT = (
    "No rule-based screening threshold was crossed in this session. That is not "
    "clinical clearance: screening rules on single-lead data have limited "
    "sensitivity, and this tool cannot rule anything in or out."
)

#: Diagnostic phrasings that must never appear in a flag description.
#: Checked case-insensitively at construction time.
BANNED_PHRASES = (
    "you have",
    "indicates",
    "indicative of",
    "abnormal",
    "pvc",
    "premature ventricular",
    "premature atrial",
    "diagnos",
)

#: Allowed severities, mildest first.
SEVERITIES = ("info", "review")


@dataclass(frozen=True)
class ScreeningFlag:
    """A rule-based screening flag.

    ``evidence`` holds the measurements that produced the flag; ``threshold``
    holds the rule definition that was crossed. Both are JSON-serialisable
    for storage on the ``flag`` table.
    """

    kind: str
    severity: str
    description: str
    evidence: dict[str, Any]
    threshold: dict[str, Any]
    disclaimer: str = field(default=DISCLAIMER)

    def __post_init__(self) -> None:
        if not self.disclaimer or not self.disclaimer.strip():
            raise ValueError("A screening flag cannot exist without a disclaimer.")
        if self.severity not in SEVERITIES:
            raise ValueError(f"Unknown severity {self.severity!r}; use one of {SEVERITIES}.")
        lowered = self.description.lower()
        for phrase in BANNED_PHRASES:
            if phrase in lowered:
                raise ValueError(
                    f"Diagnostic phrasing {phrase!r} is not allowed in a flag "
                    "description — use screening language ('consistent with', "
                    "'screening threshold exceeded', 'pattern worth noting')."
                )
        if not self.evidence:
            raise ValueError("A flag must carry the evidence that produced it.")
        if not self.threshold:
            raise ValueError("A flag must carry the threshold definition it crossed.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "description": self.description,
            "evidence": self.evidence,
            "threshold": self.threshold,
            "disclaimer": self.disclaimer,
        }
