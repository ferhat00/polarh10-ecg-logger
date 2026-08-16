"""WSGI entry point: ``flask --app wsgi run`` or any WSGI server."""

from app import create_app

app = create_app()
