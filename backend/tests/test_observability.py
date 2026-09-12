"""The Sentry scrubber, tested because it is a privacy guarantee.

Every request to this service carries a user's question in the body. Sentry
captures request data with unhandled exceptions by default, which means the one
field DocSentry exists to protect is the one most likely to be shipped to a
third party during an outage, at exactly the moment nobody is watching.

`send_default_pii=False` is not enough on its own, so `_scrub` strips the body
before an event leaves the process. These tests are what keep that true.
"""

import os

from app.observability import _scrub, init_sentry


def test_scrub_removes_the_request_body():
    event = {
        "request": {
            "url": "https://docsentry.example/api/chat",
            "method": "POST",
            "data": {"question": "What is my medical leave entitlement?"},
        }
    }

    scrubbed = _scrub(event, {})

    assert "data" not in scrubbed["request"]
    assert "medical leave" not in str(scrubbed)


def test_scrub_removes_cookies_and_auth_headers():
    event = {
        "request": {
            "cookies": {"session": "abc123"},
            "headers": {
                "Authorization": "Bearer secret-token",
                "Cookie": "session=abc123",
                "X-Api-Key": "key-value",
                "User-Agent": "curl/8.0",
                "Content-Type": "application/json",
            },
        }
    }

    scrubbed = _scrub(event, {})
    headers = scrubbed["request"]["headers"]

    assert "cookies" not in scrubbed["request"]
    assert "Authorization" not in headers
    assert "Cookie" not in headers
    assert "X-Api-Key" not in headers
    # Harmless headers are worth keeping: stripping everything would make an
    # event useless for debugging, which is the other way to get this wrong.
    assert headers["User-Agent"] == "curl/8.0"
    assert headers["Content-Type"] == "application/json"
    assert "secret-token" not in str(scrubbed)


def test_scrub_leaves_an_event_without_request_data_alone():
    """A worker exception has no request at all, and must not blow up the hook."""
    event = {"exception": {"values": [{"type": "RuntimeError"}]}}
    assert _scrub(event, {}) == event


def test_scrub_survives_a_request_shaped_unexpectedly():
    """before_send runs inside the SDK; raising there loses the event entirely."""
    assert _scrub({"request": "not-a-dict"}, {}) == {"request": "not-a-dict"}
    assert _scrub({"request": {"headers": "not-a-dict"}}, {}) == {
        "request": {"headers": "not-a-dict"}
    }


def test_init_sentry_is_inert_without_a_dsn(monkeypatch):
    """Cloning this repo and running it must not require a Sentry account."""
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    assert init_sentry() is False


def test_init_sentry_ignores_a_blank_dsn(monkeypatch):
    """An empty variable in a .env file is the normal way this arrives unset."""
    monkeypatch.setenv("SENTRY_DSN", "   ")
    assert init_sentry() is False


def test_init_sentry_never_raises_on_a_bad_dsn(monkeypatch):
    """Error reporting failing to start is not a reason to fail to start."""
    monkeypatch.setenv("SENTRY_DSN", "not-a-valid-dsn")
    assert init_sentry() is False
    assert os.environ["SENTRY_DSN"] == "not-a-valid-dsn"
