"""Activity profiles: what an activity makes valid, suspect, or meaningless.

The user's activity selection changes which metrics are computed, which are
suppressed, and how results are read — because posture and motion dominate
HRV far more than fitness does. Adding a coded activity is one new file in
``app/activities/profiles/``; adding a user activity is one ``activity_type``
row referencing an existing profile.
"""

from app.activities.registry import all_profiles, get_profile, resolve_profile

__all__ = ["all_profiles", "get_profile", "resolve_profile"]
