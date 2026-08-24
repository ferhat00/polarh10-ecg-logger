"""Polar Flow account linking and manual sync (opt-in).

This blueprint is registered **only** when ``ECGLOG_POLAR_ENABLED`` is set, so
with the feature off these URLs do not exist rather than merely refusing.

The OAuth round trip is guarded by a ``state`` value held in the Flask session:
the callback is a GET that creates a durable credential, so it must not be
possible to drive it by sending the wearer a crafted link.
"""

from __future__ import annotations

import datetime as dt
import secrets

from flask import Blueprint, Response, current_app, flash, redirect, request, url_for

# Aliased: bare ``session`` would read as ``db.session`` everywhere else here.
from flask import session as flask_session

from app.extensions import db
from app.models import Person, PolarAccount
from app.polar import client as polar_client
from app.polar import sync as polar_sync

bp = Blueprint("polar", __name__, url_prefix="/polar")

#: Where the pending link's state and person id live between the redirect out
#: and the callback back.
_STATE_KEY = "polar_oauth_state"
_PERSON_KEY = "polar_oauth_person_id"


def _credentials() -> tuple[str, str, str] | None:
    """(client_id, client_secret, redirect_uri), or None if not configured."""
    cfg = current_app.config
    client_id = cfg.get("POLAR_CLIENT_ID")
    client_secret = cfg.get("POLAR_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None
    return client_id, client_secret, cfg["POLAR_REDIRECT_URI"]


def _timeout() -> float:
    return float(current_app.config.get("POLAR_TIMEOUT_S", 10))


@bp.get("/link/<int:person_id>")
def link(person_id: int) -> Response:
    """Send the wearer to Polar Flow to authorise this client."""
    person = db.get_or_404(Person, person_id)
    creds = _credentials()
    if creds is None:
        flash(
            "Polar sync is enabled but ECGLOG_POLAR_CLIENT_ID / "
            "ECGLOG_POLAR_CLIENT_SECRET are not set — see docs/POLAR_FLOW.md.",
            "error",
        )
        return redirect(url_for("people.detail", person_id=person.id))

    client_id, _secret, redirect_uri = creds
    state = secrets.token_urlsafe(24)
    flask_session[_STATE_KEY] = state
    flask_session[_PERSON_KEY] = person.id
    return redirect(polar_client.authorization_url(client_id, redirect_uri, state))


@bp.get("/callback")
def callback() -> Response:
    """Exchange the authorisation code and store the account."""
    state = flask_session.pop(_STATE_KEY, None)
    person_id = flask_session.pop(_PERSON_KEY, None)
    people_index = url_for("people.list_people")

    if not state or state != request.args.get("state") or person_id is None:
        flash("That Polar authorisation did not match a link started here.", "error")
        return redirect(people_index)

    person = db.session.get(Person, person_id)
    if person is None:
        flash("The person being linked no longer exists.", "error")
        return redirect(people_index)
    back = url_for("people.detail", person_id=person.id)

    error = request.args.get("error")
    if error:
        flash(f"Polar refused the authorisation: {error}.", "error")
        return redirect(back)
    code = request.args.get("code")
    if not code:
        flash("Polar returned no authorisation code.", "error")
        return redirect(back)

    creds = _credentials()
    if creds is None:
        flash("Polar client credentials are not configured.", "error")
        return redirect(back)
    client_id, client_secret, redirect_uri = creds

    grant, notes = polar_client.exchange_code(
        code, client_id, client_secret, redirect_uri, timeout_s=_timeout()
    )
    if grant is None:
        for note in notes or ["Polar did not return an access token."]:
            flash(note, "error")
        return redirect(back)

    linked, reg_notes = polar_client.register_user(
        grant.access_token, person.slug, timeout_s=_timeout()
    )
    for note in reg_notes:
        flash(note, "error")
    if not linked:
        return redirect(back)

    # Re-linking replaces the row: one Flow account per person, and the old
    # token has no further use.
    if person.polar_account is not None:
        db.session.delete(person.polar_account)
        db.session.flush()
    db.session.add(
        PolarAccount(
            person_id=person.id,
            polar_user_id=grant.polar_user_id,
            access_token=grant.access_token,
            member_id=person.slug,
            linked_at=dt.datetime.now(dt.UTC),
        )
    )
    db.session.commit()
    flash(f"Linked {person.name} to Polar Flow. Sync to pull the last 28 nights.", "ok")
    return redirect(back)


@bp.post("/sync/<int:person_id>")
def sync(person_id: int) -> Response:
    """Pull the recent window for one person."""
    person = db.get_or_404(Person, person_id)
    back = url_for("people.detail", person_id=person.id)
    account = person.polar_account
    if account is None:
        flash("No Polar account is linked for this person.", "error")
        return redirect(back)

    start, end = polar_sync.default_window()
    result = polar_sync.sync_person(
        person, account, start, end, timeout_s=_timeout()
    )
    for note in result.notes:
        flash(note, "error")
    flash(f"Polar sync: {result.summary()}.", "ok")
    return redirect(back)


@bp.post("/unlink/<int:person_id>")
def unlink(person_id: int) -> Response:
    """Forget the stored token. Synced nights are kept."""
    person = db.get_or_404(Person, person_id)
    back = url_for("people.detail", person_id=person.id)
    if person.polar_account is None:
        flash("No Polar account is linked for this person.", "error")
        return redirect(back)
    db.session.delete(person.polar_account)
    db.session.commit()
    flash(
        "Unlinked. The stored token is deleted here; revoke it at Polar's end "
        "at https://account.polar.com. Nights already synced are kept.",
        "ok",
    )
    return redirect(back)
