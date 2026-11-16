from types import SimpleNamespace

import pytest

from agent.utils import logger


@pytest.mark.parametrize("tty,disabled,expected", [
    (False, False, False), (False, True, False),
    (True, False, True), (True, True, False),
])
def test_progress_is_interactive_only(monkeypatch, tty, disabled, expected):
    monkeypatch.setattr(logger.sys, "stderr", SimpleNamespace(isatty=lambda: tty))
    assert logger.progressbar_enabled({"noprogressbar": disabled}) == expected
