"""Keep README.md honest: its output blocks must be what the tool really prints."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"


def load_script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "update_readme", ROOT / "scripts" / "update_readme.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["update_readme"] = module
    spec.loader.exec_module(module)
    return module


def test_readme_has_all_demo_blocks() -> None:
    script = load_script()
    found = script.blocks(README.read_text(encoding="utf-8"))
    assert set(found) == set(script.COMMANDS)
    for name, body in found.items():
        assert body.startswith(f"$ triton-cachekit {name} --demo\n"), name
        assert len(body.splitlines()) > 8, name


@pytest.mark.triton
@pytest.mark.parametrize("name", ["inspect", "explain", "stats"])
def test_readme_output_matches_the_tool(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    script = load_script()
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    for variable in ("TRITON_CACHE_DIR", "TRITON_DEFAULT_FP_FUSION"):
        monkeypatch.delenv(variable, raising=False)
    documented = script.blocks(README.read_text(encoding="utf-8"))[name]
    assert script.normalise(script.generate(name)) == script.normalise(documented)


def test_readme_makes_no_unbacked_claims() -> None:
    raw = README.read_text(encoding="utf-8")
    text = re.sub(r"[\s>]+", " ", raw)
    assert "not verified" in text.lower() and "real GPU" in text
    assert "not affiliated" in text
    assert not re.search(r"img\.shields\.io/pypi", raw)  # not published on PyPI
    for number in (12429, 12367, 11817, 11999):
        assert f"triton-lang/triton/issues/{number}" in raw


def test_documented_commands_exist() -> None:
    from triton_cachekit.cli import build_parser

    parser = build_parser()
    choices = next(a for a in parser._actions if a.dest == "command").choices
    text = README.read_text(encoding="utf-8")
    used = set(re.findall(r"triton-cachekit ([a-z][a-z-]+)", text))
    assert used <= set(choices), used - set(choices)
