"""Flask CLI commands.

* ``flask env-backfill`` — fill the environment columns of existing sessions
  after opting in to the weather lookup. Backfilled values appear on the
  session page and in trends immediately; the stored report HTML reflects
  processing time and refreshes on re-analyse.
* ``flask polar-sync`` — pull sleep and Nightly Recharge from a linked Polar
  Flow account into ``flow_night`` rows.
"""

from __future__ import annotations

import datetime as dt

import click
from flask import Flask, current_app
from flask.cli import with_appcontext
from sqlalchemy import select

from app.extensions import db
from app.models import Person, PolarAccount, ProcessingStatus, Session


@click.command("env-backfill")
@click.option("--person", "person_slug", default=None, help="Limit to one person's slug.")
@click.option("--refetch", is_flag=True, help="Refetch sessions that already have data.")
@click.option("--limit", type=int, default=None, help="Stop after N sessions.")
@with_appcontext
def env_backfill(person_slug: str | None, refetch: bool, limit: int | None) -> None:
    """Fetch environment data for existing sessions (opt-in feature)."""
    cfg = current_app.config
    if not cfg.get("WEATHER_ENABLED"):
        click.echo(
            "The weather lookup is off. Set ECGLOG_WEATHER_ENABLED=1 plus "
            "ECGLOG_HOME_LAT/ECGLOG_HOME_LON to opt in — see README."
        )
        return
    lat, lon = cfg.get("HOME_LAT"), cfg.get("HOME_LON")
    if lat is None or lon is None:
        click.echo("ECGLOG_HOME_LAT / ECGLOG_HOME_LON are not set — nothing to do.")
        return

    stmt = (
        select(Session)
        .where(
            Session.processing_status == ProcessingStatus.DONE,
            Session.recorded_at.is_not(None),
        )
        .order_by(Session.recorded_at)
    )
    if person_slug:
        person = db.session.scalar(select(Person).where(Person.slug == person_slug))
        if person is None:
            click.echo(f"No person with slug {person_slug!r}.")
            return
        stmt = stmt.where(Session.person_id == person.id)

    from app.environment.client import fetch_environment

    done = 0
    for session in db.session.scalars(stmt):
        if limit is not None and done >= limit:
            break
        if session.env_fetched_at is not None and not refetch:
            click.echo(f"session {session.id}: already fetched — skipped (use --refetch)")
            continue
        sample = fetch_environment(
            lat,
            lon,
            session.recorded_at,
            session.duration_s or 0.0,
            timeout_s=float(cfg.get("WEATHER_TIMEOUT_S", 10)),
        )
        if sample is None:
            click.echo(f"session {session.id}: nothing retrievable — left unchanged")
            continue
        session.env_temp_c = sample.temp_c
        session.env_apparent_temp_c = sample.apparent_temp_c
        session.env_humidity_pct = sample.humidity_pct
        session.env_pressure_hpa = sample.pressure_hpa
        session.env_pm25_ugm3 = sample.pm25_ugm3
        session.env_pm10_ugm3 = sample.pm10_ugm3
        session.env_ozone_ugm3 = sample.ozone_ugm3
        session.env_no2_ugm3 = sample.no2_ugm3
        session.env_aqi = sample.aqi
        session.env_daylight_h = sample.daylight_h
        session.env_source = sample.source
        session.env_fetched_at = dt.datetime.now(dt.UTC)
        db.session.commit()
        done += 1
        temp = f"{sample.temp_c:.1f} °C" if sample.temp_c is not None else "no temp"
        pm = f"PM2.5 {sample.pm25_ugm3:.0f}" if sample.pm25_ugm3 is not None else "no AQ"
        click.echo(f"session {session.id} ({session.recorded_at:%Y-%m-%d}): {temp} · {pm}")

    click.echo(f"{done} session(s) updated. Reports refresh on re-analyse.")


@click.command("polar-sync")
@click.option("--person", "person_slug", default=None, help="Limit to one person's slug.")
@click.option(
    "--from",
    "from_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Inclusive start date (default: 28 days ago).",
)
@click.option(
    "--to",
    "to_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Inclusive end date (default: today).",
)
@click.option(
    "--no-heart-rate",
    is_flag=True,
    help="Skip the 24/7 heart-rate samples (one request less per sync).",
)
@with_appcontext
def polar_sync(
    person_slug: str | None,
    from_date: dt.datetime | None,
    to_date: dt.datetime | None,
    no_heart_rate: bool,
) -> None:
    """Pull sleep and Nightly Recharge from linked Polar Flow accounts."""
    cfg = current_app.config
    if not cfg.get("POLAR_ENABLED"):
        click.echo(
            "Polar Flow sync is off. Set ECGLOG_POLAR_ENABLED=1 plus "
            "ECGLOG_POLAR_CLIENT_ID/ECGLOG_POLAR_CLIENT_SECRET to opt in — "
            "see docs/POLAR_FLOW.md."
        )
        return

    from app.polar.sync import default_window, sync_person

    window_start, window_end = default_window()
    start = from_date.date() if from_date else window_start
    end = to_date.date() if to_date else window_end
    if start > end:
        click.echo(f"--from ({start}) is after --to ({end}) — nothing to do.")
        return

    stmt = select(PolarAccount).join(Person).order_by(Person.name)
    if person_slug:
        person = db.session.scalar(select(Person).where(Person.slug == person_slug))
        if person is None:
            click.echo(f"No person with slug {person_slug!r}.")
            return
        stmt = stmt.where(PolarAccount.person_id == person.id)

    accounts = list(db.session.scalars(stmt))
    if not accounts:
        click.echo(
            "No linked Polar accounts. Link one from the person page "
            "(Link Polar Flow) and run this again."
        )
        return

    for account in accounts:
        person = account.person
        result = sync_person(
            person,
            account,
            start,
            end,
            with_heart_rate=not no_heart_rate,
            timeout_s=float(cfg.get("POLAR_TIMEOUT_S", 10)),
        )
        click.echo(f"{person.name} ({start} → {end}): {result.summary()}")
        for note in result.notes:
            click.echo(f"  · {note}")
        if result.empty:
            click.echo(f"  · {result.empty} date(s) had no data behind them")


def register_cli(app: Flask) -> None:
    app.cli.add_command(env_backfill)
    app.cli.add_command(polar_sync)
