"""Offline test defaults: no model calls from the reply checks."""
import pytest


@pytest.fixture(autouse=True)
def _offline_reply_checks(monkeypatch):
    monkeypatch.setenv("SHOPSENSE_LLM_COMMITMENT_CHECK", "off")
    monkeypatch.setenv("SHOPSENSE_INJECTION_CLASSIFIER", "off")
    monkeypatch.setenv("SHOPSENSE_INJECTION_JUDGE", "off")
    # API tests stub the M5 workflow; the default engine (the M6 team) has its own test
    monkeypatch.setenv("SHOPSENSE_ENGINE", "workflow")
