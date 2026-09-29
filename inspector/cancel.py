"""Cooperative cancellation for the long steps: fit and threshold measuring.

Stopping a request in the browser does not stop the Python behind it -- the
worker thread runs on to the end and saves as if nothing happened. So a cancel
only raises a flag, and the long loops call `check()` at points where
stopping is safe: before each photo is processed, and before anything is
written. A cancelled fit therefore never reaches `save_registration`.

A cancel counts only while a step wrapped in `cancellable()` is running, so a
stray click while idle cannot abort the next inspection, which passes the
same checkpoints. One flag for the process: the demo is a single-operator
tool, and a cancel means "stop what is running now".
"""
from __future__ import annotations

import threading
from contextlib import contextmanager

_active = threading.Event()
_requested = threading.Event()


class Cancelled(Exception):
    """Raised at the next checkpoint after `request()`."""


@contextmanager
def cancellable():
    """Mark a long step as running; any cancel is cleared when it ends."""
    _requested.clear()
    _active.set()
    try:
        yield
    finally:
        _active.clear()
        _requested.clear()


def request() -> bool:
    """Ask the running step to stop. False if nothing was running."""
    if not _active.is_set():
        return False
    _requested.set()
    return True


def check() -> None:
    if _requested.is_set():
        raise Cancelled("cancelled by the operator")
