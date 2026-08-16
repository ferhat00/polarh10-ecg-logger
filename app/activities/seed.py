"""Seed the eight built-in activity types into the activity_type table."""

from __future__ import annotations

from app.activities.registry import all_profiles
from app.extensions import db
from app.models import ActivityType


def ensure_builtin_activity_types() -> int:
    """Insert missing built-in rows; idempotent. Returns rows added."""
    existing = {
        row.profile_key
        for row in db.session.query(ActivityType).filter_by(is_builtin=True)
    }
    added = 0
    for key, profile in all_profiles().items():
        if key not in existing:
            db.session.add(
                ActivityType(
                    name=profile.display_name,
                    profile_key=key,
                    is_builtin=True,
                )
            )
            added += 1
    if added:
        db.session.commit()
    return added
