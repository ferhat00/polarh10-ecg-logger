"""Activity-profile registry.

Coded profiles register themselves via :func:`register`; the eight built-ins
live one-per-file in ``app/activities/profiles/``. A DB ``activity_type`` row
maps a display name to a profile key and may carry overrides (expected HR
range, comparison key) for UI-added activities — :func:`resolve_profile`
applies them.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Any

from app.activities.base import ActivityProfile

_REGISTRY: dict[str, ActivityProfile] = {}
_LOADED = False


def register(cls: type[ActivityProfile]) -> type[ActivityProfile]:
    """Class decorator: instantiate and register a profile by its key."""
    instance = cls()
    if not instance.key:
        raise ValueError(f"{cls.__name__} has no key.")
    if instance.key in _REGISTRY:
        raise ValueError(f"Duplicate activity profile key {instance.key!r}.")
    if not instance.comparison_key:
        instance.comparison_key = instance.key
    _REGISTRY[instance.key] = instance
    return cls


def _load() -> None:
    """Import every module in app.activities.profiles exactly once."""
    global _LOADED
    if _LOADED:
        return
    from app.activities import profiles as pkg

    for mod in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"{pkg.__name__}.{mod.name}")
    _LOADED = True


def get_profile(key: str) -> ActivityProfile:
    _load()
    if key not in _REGISTRY:
        raise KeyError(
            f"No activity profile {key!r}. Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]


def all_profiles() -> dict[str, ActivityProfile]:
    _load()
    return dict(_REGISTRY)


@dataclass
class ResolvedActivity:
    """A profile with any DB-row overrides applied."""

    profile: ActivityProfile
    display_name: str
    expected_hr_range: tuple[int, int]
    comparison_key: str


def resolve_profile(activity_type: Any) -> ResolvedActivity:
    """Resolve an ``activity_type`` row (or any duck-typed object with
    ``profile_key`` / ``name`` / override fields) to its effective profile."""
    profile = get_profile(activity_type.profile_key)
    hr_min = getattr(activity_type, "expected_hr_min", None) or profile.expected_hr_range[0]
    hr_max = getattr(activity_type, "expected_hr_max", None) or profile.expected_hr_range[1]
    return ResolvedActivity(
        profile=profile,
        display_name=getattr(activity_type, "name", profile.display_name),
        expected_hr_range=(hr_min, hr_max),
        comparison_key=getattr(activity_type, "comparison_key", None)
        or profile.comparison_key,
    )
