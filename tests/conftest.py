"""Shared pytest fixtures: an app with an in-memory database and a client."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from flask import Flask
from flask.testing import FlaskClient

from app import create_app
from app.config import TestConfig
from app.extensions import db


@pytest.fixture()
def app(tmp_path) -> Iterator[Flask]:
    config = TestConfig()
    config.DATA_DIR = tmp_path / "data"
    app = create_app(config)
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app: Flask) -> FlaskClient:
    return app.test_client()
