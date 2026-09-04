"""Every name the Modal glue imports from slopmarker must actually exist.

modal_app.py is excluded from mypy -- Modal's decorators are untyped, so every
@app.function would be reported as making its target untyped -- and its imports sit
inside function bodies so they never run until a container does. A name that does not
exist therefore fails forty minutes into a remote job rather than at the keyboard,
which is how a sweep died on `to_fp16` after a deploy that reported success.

Ruff does not catch it either: the import is in another module, and it is syntactically
fine. So check it directly.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "slopmarker"
IMPORT = re.compile(r"^\s*from (slopmarker[\w.]*) import (.+?)$", re.M)


def imported_names() -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for path in SRC.rglob("modal_app.py"):
        text = path.read_text(encoding="utf-8")
        for module, names in IMPORT.findall(text):
            for name in names.split(","):
                name = name.strip().rstrip("()").split(" as ")[0].strip()
                if name and name != "(":
                    found.append((path.name, module, name))
    return found


def test_there_are_imports_to_check() -> None:
    """A regex that silently matches nothing would make this file decorative."""
    assert len(imported_names()) > 10


@pytest.mark.parametrize(
    ("where", "module", "name"),
    imported_names(),
    ids=lambda v: str(v),
)
def test_imported_name_exists(where: str, module: str, name: str) -> None:
    imported = importlib.import_module(module)
    assert hasattr(imported, name), f"{where} imports {name} from {module}, which lacks it"
