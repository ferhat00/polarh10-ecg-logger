"""Upsert fetched Flow nights into the database.

Idempotent by ``(person_id, date)``: re-running a sync over the same window
updates rows rather than adding them, so the CLI and the person-page button
can be pressed as often as the wearer likes.

Failures are reported, never raised. A sync that hits a rate limit or a
revoked token still writes whatever it got before the wall, and says where it
stopped — a half-synced window is more useful than an exception, and silence
about the wall would be worse than either.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from sqlalchemy import select

from app.extensions import db
from app.models import FlowNight, Person, PolarAccount
from app.polar import client as polar_client

#: Default window for a routine sync: the documented list-endpoint window,
#: which costs two requests regardless of how many nights come back.
DEFAULT_WINDOW_DAYS = polar_client.LIST_WINDOW_DAYS

#: Columns copied straight across from the fetched record. Kept as a tuple so
#: adding a field to both sides is one edit, not two.
_COPIED_FIELDS: tuple[str, ...] = (
    "source_device_id",
    "sleep_start",
    "sleep_end",
    "light_sleep_s",
    "deep_sleep_s",
    "rem_sleep_s",
    "unrecognized_sleep_s",
    "total_interruption_s",
    "sleep_score",
    "sleep_charge",
    "continuity",
    "continuity_class",
    "sleep_cycles",
    "hr_avg_bpm",
    "beat_to_beat_avg_ms",
    "hrv_rmssd_ms",
    "breathing_rate_avg",
    "nightly_recharge_status",
    "ans_charge",
    "ans_charge_status",
    "hrv_samples",
    "breathing_samples",
    "hr_samples",
)


@dataclass
class SyncResult:
    """What one sync did, and honestly what it could not do."""

    created: int = 0
    updated: int = 0
    empty: int = 0
    notes: list[str] = field(default_factory=list)
    auth_failed: bool = False
    rate_limited: bool = False

    @property
    def touched(self) -> int:
        return self.created + self.updated

    def summary(self) -> str:
        """One line for the CLI and the person page."""
        if self.auth_failed:
            head = "re-authorisation needed"
        elif self.touched == 0:
            head = "no nights returned"
        else:
            head = f"{self.created} new, {self.updated} updated"
        if self.rate_limited:
            head += " (stopped: rate limited)"
        return head


def sync_person(
    person: Person,
    account: PolarAccount,
    start: dt.date,
    end: dt.date,
    with_heart_rate: bool = True,
    today: dt.date | None = None,
    timeout_s: float = 10.0,
    opener=None,
) -> SyncResult:
    """Fetch ``[start, end]`` for one person and upsert the nights."""
    result = SyncResult()
    fetched = polar_client.fetch_nights(
        account.access_token,
        start,
        end,
        today=today,
        timeout_s=timeout_s,
        opener=opener,
    )
    if with_heart_rate:
        polar_client.fetch_continuous_hr(
            account.access_token,
            start,
            end,
            fetched,
            timeout_s=timeout_s,
            opener=opener,
        )

    result.notes.extend(fetched.notes)
    result.auth_failed = fetched.auth_failed
    result.rate_limited = fetched.rate_limited

    existing = {
        night.date: night
        for night in db.session.scalars(
            select(FlowNight).where(
                FlowNight.person_id == person.id,
                FlowNight.date >= start,
                FlowNight.date <= end,
            )
        )
    }

    for date in sorted(fetched.nights):
        record = fetched.nights[date]
        # A date can come back from /sleep/available with nothing behind it;
        # storing an all-null row would fake coverage we do not have.
        if not _has_content(record):
            result.empty += 1
            continue
        night = existing.get(date)
        if night is None:
            night = FlowNight(person_id=person.id, date=date)
            db.session.add(night)
            result.created += 1
        else:
            result.updated += 1
        for name in _COPIED_FIELDS:
            setattr(night, name, getattr(record, name))
        night.fetched_at = dt.datetime.now(dt.UTC)

    account.last_sync_at = dt.datetime.now(dt.UTC)
    account.last_sync_note = result.summary()[:500]
    if result.auth_failed:
        # The token is dead; keeping it would only produce the same 401 on
        # every future sync. Drop it and let the person re-link.
        db.session.delete(account)
    db.session.commit()
    return result


def default_window(today: dt.date | None = None) -> tuple[dt.date, dt.date]:
    """The routine-sync window: the last ``DEFAULT_WINDOW_DAYS`` days."""
    today = today or dt.datetime.now(dt.UTC).date()
    return today - dt.timedelta(days=DEFAULT_WINDOW_DAYS), today


def _has_content(record: polar_client.NightRecord) -> bool:
    """True when the record carries anything beyond its date."""
    return any(
        getattr(record, name) is not None for name in _COPIED_FIELDS
    )
