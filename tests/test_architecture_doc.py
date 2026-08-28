"""Keep ``docs/ARCHITECTURE.md`` in step with the package it describes.

An architecture document is the file most likely to rot: nothing breaks when a
module is renamed out from under it, so it quietly drifts into describing a
codebase that no longer exists — which is worse than having no document, because
a reader has no way to tell which half is still true.

These tests make the document falsifiable in the same way ``test_docs.py`` makes
the README's numbers falsifiable. Three claims are checked:

* **Coverage, in both directions.** Every module in ``nl2sql/`` has a section,
  and every section names a module that exists. A new module with no section
  fails; so does a section left behind by a deleted one.
* **Every symbol it names is real.** The prose refers to functions, classes and
  constants as ``nl2sql.module.name``; each is resolved by import and
  ``getattr``, so a rename that the document does not follow fails here rather
  than misleading a reader.
* **The one number it quotes is current.** The repair budget is stated in prose
  and is also a constant in the code; the two must agree.

The regexes are anchored to the document's own conventions (``### `nl2sql/x.py```
headings, backticked dotted paths). Each test asserts that its pattern matched
*something* before checking the matches, so rewriting the surrounding prose into
a shape these tests cannot read fails loudly instead of silently disabling the
check.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = REPO_ROOT / "docs" / "ARCHITECTURE.md"
PACKAGE_DIR = REPO_ROOT / "nl2sql"

# "### `nl2sql/runner.py`" — one section heading per module.
_SECTION_RE = re.compile(r"^### `nl2sql/(\w+)\.py`", re.MULTILINE)
# "`nl2sql.runner.validate`" / "`nl2sql.generator.MAX_REPAIR_ATTEMPTS`".
# Only dotted paths inside backticks count, so a bare mention in prose is not
# mistaken for an API claim. Trailing attribute chains (``a.b.c.d``) are matched
# up to the first attribute; deeper members are checked via their own mention.
_SYMBOL_RE = re.compile(r"`nl2sql\.(\w+)\.(\w+)")
# "a budget of **1 attempt**" — the repair budget, stated in prose.
_REPAIR_BUDGET_RE = re.compile(r"budget of \*\*(\d+) attempt")

# `__init__.py` carries only the version string and has no architecture to
# describe; requiring a section for it would be documentation as bookkeeping.
_UNDOCUMENTED_BY_DESIGN = frozenset({"__init__"})


@pytest.fixture(scope="module")
def doc() -> str:
    return DOC_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def package_modules() -> set[str]:
    """Return the module names in ``nl2sql/`` that the document must cover."""
    return {
        path.stem
        for path in PACKAGE_DIR.glob("*.py")
        if path.stem not in _UNDOCUMENTED_BY_DESIGN
    }


@pytest.fixture(scope="module")
def documented_modules(doc: str) -> set[str]:
    sections = _SECTION_RE.findall(doc)
    assert sections, "ARCHITECTURE.md no longer has any '### `nl2sql/<name>.py`' sections"
    return set(sections)


def test_every_module_has_a_section(
    documented_modules: set[str], package_modules: set[str]
) -> None:
    missing = sorted(package_modules - documented_modules)
    assert not missing, (
        f"ARCHITECTURE.md has no section for: {missing}. "
        "A new module needs a paragraph explaining what boundary it owns."
    )


def test_every_section_names_a_real_module(
    documented_modules: set[str], package_modules: set[str]
) -> None:
    stale = sorted(documented_modules - package_modules)
    assert not stale, (
        f"ARCHITECTURE.md documents modules that do not exist in nl2sql/: {stale}"
    )


def test_every_symbol_named_in_the_doc_exists(doc: str) -> None:
    """Resolve every ``nl2sql.<module>.<name>`` the document mentions."""
    references = sorted(set(_SYMBOL_RE.findall(doc)))
    assert references, "ARCHITECTURE.md no longer names any nl2sql symbols"

    unresolved: list[str] = []
    for module_name, attribute in references:
        try:
            module = importlib.import_module(f"nl2sql.{module_name}")
        except ImportError:
            unresolved.append(f"nl2sql.{module_name} (module not importable)")
            continue
        if not hasattr(module, attribute):
            unresolved.append(f"nl2sql.{module_name}.{attribute}")

    assert not unresolved, (
        f"ARCHITECTURE.md names symbols that do not exist: {unresolved}"
    )


def test_repair_budget_claim_matches_the_constant(doc: str) -> None:
    from nl2sql import generator

    match = _REPAIR_BUDGET_RE.search(doc)
    assert match, "ARCHITECTURE.md no longer states the repair budget as 'budget of **N attempt...'"
    assert int(match.group(1)) == generator.MAX_REPAIR_ATTEMPTS, (
        f"ARCHITECTURE.md claims a repair budget of {match.group(1)}, "
        f"but generator.MAX_REPAIR_ATTEMPTS is {generator.MAX_REPAIR_ATTEMPTS}"
    )
