"""Loader exceptions.

A wrong unit or epoch assumption corrupts everything downstream, so the loader
never guesses: when a file does not match the verified Polar H10 export shape
it raises :class:`AmbiguousFormatError` carrying the specific questions that
need answering. The UI turns those questions into a mapping form; the answers
come back as ``FormatOverrides``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


class LoaderError(Exception):
    """Base class for all loader failures (unusable file, too short, etc.)."""


@dataclass(frozen=True)
class FormatQuestion:
    """One thing the loader could not determine and refuses to guess.

    Attributes
    ----------
    key:
        Machine key identifying which ``FormatOverrides`` field answers this
        question (e.g. ``"delimiter"``, ``"decimal"``, ``"columns"``,
        ``"ecg_unit"``, ``"epoch"``, ``"time_unit"``).
    question:
        Plain-language question to show the user.
    observed:
        What the loader actually saw in the file (evidence for the user).
    options:
        Suggested answers, when the choice is enumerable.
    """

    key: str
    question: str
    observed: str
    options: tuple[str, ...] = field(default=())
    #: Structured values behind ``observed`` (e.g. the raw column names for a
    #: ``columns`` question) so the mapping form can build per-item inputs.
    observed_values: tuple[str, ...] = field(default=())


class AmbiguousFormatError(LoaderError):
    """The file doesn't match the known export shape; user input is needed.

    Carries structured :class:`FormatQuestion` items so the upload UI can show
    a mapping form instead of guessing.
    """

    def __init__(self, message: str, questions: Sequence[FormatQuestion]) -> None:
        super().__init__(message)
        self.questions: list[FormatQuestion] = list(questions)

    def as_dict(self) -> dict:
        """JSON-serialisable form for the status endpoint / mapping form."""
        return {
            "message": str(self),
            "questions": [
                {
                    "key": q.key,
                    "question": q.question,
                    "observed": q.observed,
                    "options": list(q.options),
                    "observed_values": list(q.observed_values),
                }
                for q in self.questions
            ],
        }
