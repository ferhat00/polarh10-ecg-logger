"""Seed the built-in trigger tags into the trigger_tag table."""

from __future__ import annotations

from app.extensions import db
from app.models import TriggerTag
from app.triggers.vocabulary import BUILTIN_TRIGGER_TAGS


def ensure_builtin_trigger_tags() -> int:
    """Insert missing built-in rows; idempotent. Returns rows added."""
    existing = {
        row.slug for row in db.session.query(TriggerTag).filter_by(is_builtin=True)
    }
    added = 0
    for slug, name in BUILTIN_TRIGGER_TAGS:
        if slug not in existing:
            db.session.add(TriggerTag(name=name, slug=slug, is_builtin=True))
            added += 1
    if added:
        db.session.commit()
    return added
