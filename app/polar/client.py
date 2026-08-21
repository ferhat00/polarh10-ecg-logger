"""Polar AccessLink client for the opt-in Flow sync.

Design constraints mirror ``app/environment/client.py`` — that module is the
template for "opt-in outbound call" in this codebase:

* **Opt-in only** — callers gate on config; this module never reads config.
* **stdlib only** — ``urllib`` keeps the dependency surface unchanged; the
  ``opener`` parameter is the injection seam so tests never touch the network.
* **Never raises on network trouble** — a failed or partial fetch returns what
  it got with human-readable notes. A missing night is a normal outcome.
* **Minimal data out** — a bearer token and a date range, nothing else.

One constraint is specific to this API and is enforced structurally rather
than by convention. AccessLink has two families of endpoints:

* *Non-transactional* (sleep, nightly recharge, continuous heart rate,
  SleepWise, biosensing) — plain GETs, re-readable as often as you like.
* *Transactional* (``exercise-transactions``, ``activity-transactions``) —
  create -> read -> **commit**, where committing **deletes the data
  server-side**. One careless call permanently destroys the wearer's ability
  to re-fetch that data.

This client therefore refuses any path outside :data:`ALLOWED_PATH_PREFIXES`,
and does so by raising: reaching a transaction endpoint is a programming
error, not a runtime condition to be noted and shrugged off.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

API_BASE = "https://www.polaraccesslink.com"
AUTHORIZATION_URL = "https://flow.polar.com/oauth2/authorization"
TOKEN_URL = "https://polarremote.com/v2/oauth2/token"

#: Registration is the one non-data path this client may reach.
REGISTER_PATH = "/v3/users"

#: Every data path this client may request. Anything else — above all the
#: transaction families — raises. See the module docstring.
ALLOWED_PATH_PREFIXES: tuple[str, ...] = (
    "/v3/users/sleep",
    "/v3/users/nightly-recharge",
    "/v3/users/continuous-heart-rate",
    "/v3/users/biosensing/skintemperature",
)

#: The list endpoints cover this window and cost one request each; anything
#: older needs by-date calls. Polar documents the window, not the depth
#: available behind it.
LIST_WINDOW_DAYS = 28


class DisallowedEndpointError(RuntimeError):
    """Raised when a caller asks for a path outside the allowlist."""


@dataclass
class NightRecord:
    """One night of Flow data, merged from the sleep and recharge endpoints."""

    date: dt.date
    source_device_id: str | None = None

    sleep_start: dt.datetime | None = None
    sleep_end: dt.datetime | None = None
    light_sleep_s: int | None = None
    deep_sleep_s: int | None = None
    rem_sleep_s: int | None = None
    unrecognized_sleep_s: int | None = None
    total_interruption_s: int | None = None
    sleep_score: int | None = None
    sleep_charge: int | None = None
    continuity: float | None = None
    continuity_class: int | None = None
    sleep_cycles: int | None = None

    hr_avg_bpm: int | None = None
    beat_to_beat_avg_ms: int | None = None
    hrv_rmssd_ms: int | None = None
    breathing_rate_avg: float | None = None
    nightly_recharge_status: int | None = None
    ans_charge: float | None = None
    ans_charge_status: int | None = None

    hrv_samples: dict | None = None
    breathing_samples: dict | None = None
    hr_samples: list | None = None


@dataclass
class FetchResult:
    """What one fetch produced, plus why anything is missing.

    ``auth_failed`` and ``rate_limited`` are separate from the note text
    because callers act on them: the first should clear the stored token, the
    second should stop rather than keep hammering.
    """

    nights: dict[dt.date, NightRecord] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    auth_failed: bool = False
    rate_limited: bool = False

    def night(self, date: dt.date) -> NightRecord:
        record = self.nights.get(date)
        if record is None:
            record = NightRecord(date=date)
            self.nights[date] = record
        return record


@dataclass
class TokenGrant:
    """A successful token exchange."""

    access_token: str
    polar_user_id: str


# --------------------------------------------------------------------------
# Authorisation
# --------------------------------------------------------------------------


def authorization_url(client_id: str, redirect_uri: str, state: str) -> str:
    """Where to send the wearer's browser to authorise this client."""
    return AUTHORIZATION_URL + "?" + urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": "accesslink.read_all",
            "state": state,
        }
    )


def exchange_code(
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    timeout_s: float = 10.0,
    opener=None,
) -> tuple[TokenGrant | None, list[str]]:
    """Trade an authorisation code for an access token.

    AccessLink issues no refresh token: the access token stays valid until it
    is revoked, which is why the caller stores it rather than re-deriving it.
    """
    notes: list[str] = []
    body = urllib.parse.urlencode(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
        }
    ).encode()
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    request = urllib.request.Request(
        TOKEN_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
    )
    payload, status, sent_notes = _send(request, timeout_s, opener, "token exchange")
    notes.extend(sent_notes)
    if not isinstance(payload, dict):
        return None, notes
    token = payload.get("access_token")
    user_id = payload.get("x_user_id")
    if not token or user_id is None:
        notes.append(f"token exchange returned no token (HTTP {status}).")
        return None, notes
    return TokenGrant(access_token=str(token), polar_user_id=str(user_id)), notes


def register_user(
    token: str,
    member_id: str,
    timeout_s: float = 10.0,
    opener=None,
) -> tuple[bool, list[str]]:
    """Register the wearer with this client. Returns ``(linked, notes)``.

    A **409 means the user is already registered** to this client — the normal
    outcome of re-linking, and a success for our purposes, not an error.
    """
    notes: list[str] = []
    request = _api_request(
        REGISTER_PATH,
        token,
        method="POST",
        data=json.dumps({"member-id": member_id}).encode(),
        content_type="application/json",
    )
    payload, status, sent_notes = _send(
        request, timeout_s, opener, "user registration", ok_statuses=(409,)
    )
    if status == 409:
        return True, notes
    notes.extend(sent_notes)
    if status is None or status >= 400:
        return False, notes
    if payload is None and status not in (200, 204):
        return False, notes
    return True, notes


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def fetch_nights(
    token: str,
    start: dt.date,
    end: dt.date,
    today: dt.date | None = None,
    timeout_s: float = 10.0,
    opener=None,
) -> FetchResult:
    """Sleep + Nightly Recharge for ``[start, end]``, merged by date.

    Dates inside the documented 28-day list window cost two requests in total;
    older dates are fetched one at a time and only for dates Polar says it
    holds (``/v3/users/sleep/available``), because the per-user rate limit is
    500 requests per 15 minutes and a naive year-long backfill blows through
    it.
    """
    result = FetchResult()
    today = today or dt.datetime.now(dt.UTC).date()
    if start > end:
        result.notes.append("empty date range — nothing fetched")
        return result

    list_floor = today - dt.timedelta(days=LIST_WINDOW_DAYS)

    # --- the cheap path: two list calls cover the last 28 days ------------
    if end >= list_floor:
        for path, key, apply in (
            ("/v3/users/sleep", "nights", _apply_sleep),
            ("/v3/users/nightly-recharge", "recharges", _apply_recharge),
        ):
            body = _get_json(path, token, timeout_s, opener, result, what=key)
            if not isinstance(body, dict):
                continue
            for item in body.get(key) or []:
                date = _parse_date(item.get("date"))
                if date is None or not (start <= date <= end):
                    continue
                apply(result.night(date), item)

    # --- older dates: only those Polar says it holds ----------------------
    if start < list_floor and not (result.auth_failed or result.rate_limited):
        older_end = min(end, list_floor - dt.timedelta(days=1))
        available = _available_dates(token, timeout_s, opener, result)
        wanted = [d for d in available if start <= d <= older_end]
        if available and not wanted:
            result.notes.append(
                f"no Flow data on record before {list_floor.isoformat()} "
                "within the requested range"
            )
        for date in wanted:
            if result.auth_failed or result.rate_limited:
                result.notes.append(
                    f"stopped at {date.isoformat()} — see the note above"
                )
                break
            iso = date.isoformat()
            sleep = _get_json(
                f"/v3/users/sleep/{iso}",
                token,
                timeout_s,
                opener,
                result,
                what=f"sleep {iso}",
                quiet_404=True,
            )
            if isinstance(sleep, dict):
                _apply_sleep(result.night(date), sleep)
            recharge = _get_json(
                f"/v3/users/nightly-recharge/{iso}",
                token,
                timeout_s,
                opener,
                result,
                what=f"nightly recharge {iso}",
                quiet_404=True,
            )
            if isinstance(recharge, dict):
                _apply_recharge(result.night(date), recharge)

    return result


def fetch_continuous_hr(
    token: str,
    start: dt.date,
    end: dt.date,
    into: FetchResult,
    timeout_s: float = 10.0,
    opener=None,
) -> None:
    """Add 24/7 heart-rate samples to the nights already in ``into``.

    Merged into the same records rather than returned separately: Polar keys
    these by the same result date, and a day of heart rate with no sleep
    behind it is still a row worth having.
    """
    if start > end or into.auth_failed or into.rate_limited:
        return
    query = urllib.parse.urlencode({"from": start.isoformat(), "to": end.isoformat()})
    body = _get_json(
        f"/v3/users/continuous-heart-rate?{query}",
        token,
        timeout_s,
        opener,
        into,
        what="continuous heart rate",
        quiet_404=True,
    )
    if body is None:
        return
    # The spec documents a single object here, but the range form returns a
    # collection in practice — accept either rather than guess.
    if isinstance(body, list):
        days: list = body
    elif "heart_rate_samples" in body:
        days = [body]
    else:
        days = next((v for v in body.values() if isinstance(v, list)), [])
    for day in days:
        if not isinstance(day, dict):
            continue
        date = _parse_date(day.get("date"))
        samples = day.get("heart_rate_samples")
        if date is None or not samples or not (start <= date <= end):
            continue
        into.night(date).hr_samples = samples


# --------------------------------------------------------------------------
# Field mapping
# --------------------------------------------------------------------------


def _apply_sleep(record: NightRecord, item: dict) -> None:
    record.source_device_id = item.get("device_id") or record.source_device_id
    record.sleep_start = _parse_datetime(item.get("sleep_start_time"))
    record.sleep_end = _parse_datetime(item.get("sleep_end_time"))
    record.light_sleep_s = _as_int(item.get("light_sleep"))
    record.deep_sleep_s = _as_int(item.get("deep_sleep"))
    record.rem_sleep_s = _as_int(item.get("rem_sleep"))
    record.unrecognized_sleep_s = _as_int(item.get("unrecognized_sleep_stage"))
    record.total_interruption_s = _as_int(item.get("total_interruption_duration"))
    record.sleep_score = _as_int(item.get("sleep_score"))
    record.sleep_charge = _as_int(item.get("sleep_charge"))
    record.continuity = _as_float(item.get("continuity"))
    record.continuity_class = _as_int(item.get("continuity_class"))
    record.sleep_cycles = _as_int(item.get("sleep_cycles"))


def _apply_recharge(record: NightRecord, item: dict) -> None:
    record.hr_avg_bpm = _as_int(item.get("heart_rate_avg"))
    record.beat_to_beat_avg_ms = _as_int(item.get("beat_to_beat_avg"))
    record.hrv_rmssd_ms = _as_int(item.get("heart_rate_variability_avg"))
    record.breathing_rate_avg = _as_float(item.get("breathing_rate_avg"))
    record.nightly_recharge_status = _as_int(item.get("nightly_recharge_status"))
    record.ans_charge = _as_float(item.get("ans_charge"))
    record.ans_charge_status = _as_int(item.get("ans_charge_status"))
    hrv = item.get("hrv_samples")
    record.hrv_samples = hrv if isinstance(hrv, dict) else None
    breathing = item.get("breathing_samples")
    record.breathing_samples = breathing if isinstance(breathing, dict) else None


def _available_dates(
    token: str, timeout_s: float, opener, result: FetchResult
) -> list[dt.date]:
    """The dates Polar reports it holds sleep for."""
    body = _get_json(
        "/v3/users/sleep/available",
        token,
        timeout_s,
        opener,
        result,
        what="available sleep dates",
        quiet_404=True,
    )
    if not isinstance(body, dict):
        return []
    items = body.get("available") or body.get("nights") or []
    parsed = (
        _parse_date(item.get("date")) for item in items if isinstance(item, dict)
    )
    return sorted(d for d in parsed if d is not None)


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------


def _api_request(
    path: str,
    token: str,
    method: str = "GET",
    data: bytes | None = None,
    content_type: str | None = None,
) -> urllib.request.Request:
    """Build an authenticated request, refusing any non-allowlisted path.

    Registration is permitted explicitly; every other path must match
    :data:`ALLOWED_PATH_PREFIXES`. That is what keeps the transaction families
    unreachable from here.
    """
    bare = path.split("?", 1)[0]
    allowed = bare == REGISTER_PATH or any(
        bare.startswith(prefix) for prefix in ALLOWED_PATH_PREFIXES
    )
    if not allowed:
        raise DisallowedEndpointError(
            f"path not allowed by this client: {bare} — AccessLink transaction "
            "endpoints delete data on commit and are deliberately unreachable "
            "from here."
        )
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    if content_type:
        headers["Content-Type"] = content_type
    return urllib.request.Request(
        API_BASE + path, data=data, method=method, headers=headers
    )


def _get_json(
    path: str,
    token: str,
    timeout_s: float,
    opener,
    result: FetchResult,
    what: str,
    quiet_404: bool = False,
) -> dict | list | None:
    """One authenticated GET -> parsed JSON; failures become notes on ``result``."""
    request = _api_request(path, token)
    payload, status, notes = _send(
        request, timeout_s, opener, what, quiet_404=quiet_404
    )
    if status == 401:
        result.auth_failed = True
    if status == 429:
        result.rate_limited = True
    result.notes.extend(notes)
    return payload


def _send(
    request: urllib.request.Request,
    timeout_s: float,
    opener,
    what: str,
    ok_statuses: tuple[int, ...] = (),
    quiet_404: bool = False,
) -> tuple[dict | list | None, int | None, list[str]]:
    """Send a request; never raise for network or HTTP trouble.

    Returns ``(payload, status, notes)``. ``status`` is None when the failure
    happened below HTTP (DNS, timeout, connection refused).
    """
    notes: list[str] = []
    # Resolved here rather than as a parameter default: a def-time default
    # captures urlopen permanently, so patching urllib.request.urlopen would
    # silently let a real request through instead of the intended fake.
    opener = opener or urllib.request.urlopen
    try:
        with opener(request, timeout=timeout_s) as response:
            raw = response.read()
            status = getattr(response, "status", None) or getattr(response, "code", 200)
    except urllib.error.HTTPError as exc:
        status = exc.code
        if status in ok_statuses:
            return None, status, notes
        notes.append(_http_note(what, status, exc, quiet_404))
        return None, status, [n for n in notes if n]
    except Exception as exc:  # noqa: BLE001 - any failure is a supported outcome
        notes.append(f"{what} failed: {type(exc).__name__}: {exc}")
        return None, None, notes

    if not raw:
        return None, status, notes
    try:
        return json.loads(raw), status, notes
    except (ValueError, TypeError) as exc:
        notes.append(f"{what}: unreadable response ({exc}).")
        return None, status, notes


def _http_note(
    what: str, status: int, exc: urllib.error.HTTPError, quiet_404: bool
) -> str:
    """The human-readable note for one HTTP failure ("" to stay silent)."""
    if status == 401:
        return (
            f"{what}: Polar rejected the stored token (401). Re-link the "
            "account from the person page."
        )
    if status == 429:
        reset = exc.headers.get("RateLimit-Reset") if exc.headers else None
        return f"{what}: rate limited by Polar (429)." + (
            f" Try again in {reset} s." if reset else ""
        )
    if status == 403:
        return (
            f"{what}: Polar returned 403 — the account has not accepted all "
            "mandatory consents. Accept them in Polar Flow, then link again."
        )
    if status == 404:
        return "" if quiet_404 else f"{what}: no data (404)."
    return f"{what} failed: HTTP {status}."


# --------------------------------------------------------------------------
# Coercion — Polar omits fields freely; absence is never an error
# --------------------------------------------------------------------------


def _as_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _parse_date(value: object) -> dt.date | None:
    if not isinstance(value, str):
        return None
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError:
        return None


def _parse_datetime(value: object) -> dt.datetime | None:
    """Polar sends local time with an offset; store it as naive UTC.

    Every other datetime column in this schema is naive UTC, so converting
    here keeps comparisons against ``Session.recorded_at`` honest.
    """
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(dt.UTC).replace(tzinfo=None)
