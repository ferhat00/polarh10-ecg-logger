"""Seed the built-in trigger tags into the trigger_tag table."""

from __future__ import annotations

from app.extensions import db
from app.models import TriggerTag
from app.triggers.vocabulary import BUILTIN_TRIGGER_TAGS


def ensure_builtin_trigger_tags() -> int:
    """Insert missing built-in rows; idempotent. Returns rows added/promoted.

    A user may already have created a *custom* tag whose slug matches a tag
    that later became built-in (e.g. they typed "nicotine" into the free-text
    field before the vocabulary grew). Inserting would violate the slug
    UNIQUE constraint, so such a row is promoted in place — keeping the
    user's display name and, crucially, all existing session links.
    """
    existing = {row.slug: row for row in db.session.query(TriggerTag)}
    changed = 0
    for slug, name in BUILTIN_TRIGGER_TAGS:
        row = existing.get(slug)
        if row is None:
            db.session.add(TriggerTag(name=name, slug=slug, is_builtin=True))
            changed += 1
        elif not row.is_builtin:
            row.is_builtin = True
            changed += 1
    if changed:
        db.session.commit()
    return changed
