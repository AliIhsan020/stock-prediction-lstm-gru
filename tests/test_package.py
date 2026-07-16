"""Smoke tests for the installed project package."""

import bist100_forecasting


def test_package_exposes_version() -> None:
    """The package exposes the version declared by the project."""
    assert bist100_forecasting.__version__ == "0.1.0"
