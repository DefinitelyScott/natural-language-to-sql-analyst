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
* **Its non-goals do not deny a module that exists.** A non-goal is a claim of
  *absence*, which is the one kind of claim the two coverage tests structurally
  cannot check: it names no module and no symbol, so there is nothing for them
  to fail to resolve.

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
# The "## Deliberate non-goals" section, up to the next heading or end of file.
_NON_GOALS_RE = re.compile(r"^## Deliberate non-goals$(.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL)

# Blanket denials that a module in ``nl2sql/`` would falsify, mapped to the
# module that falsifies them.
#
# This exists because the document really did carry "No caching or persistence
# layer" for several commits after ``nl2sql/cache.py`` landed. Neither coverage
# test could see it: ``cache`` had a section and every symbol resolved, so the
# only false sentence in the file was the one asserting the module was not
# there.
#
# Deliberately a short hand-written map of *unqualified* phrases rather than an
# attempt to read English. Scoping the claim is the intended fix, not a
# loophole: "No result cache" is true and stays true, while "No caching" is a
# statement about the whole repo that a single new module makes false. Add an
# entry when a module lands whose existence someone might later blanket-deny.
#
# Only phrases that have actually appeared are listed. A speculative entry
# ("no persistence") looks like extra safety but is untested by construction —
# nothing in the repo's history exercises it, so it may or may not match the
# wording a future author reaches for, and it makes the map look better covered
# than it is.
_BLANKET_DENIALS = {
    "no caching": "cache",
}

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


@pytest.fixture(scope="module")
def non_goals(doc: str) -> str:
    """Return the body of the 'Deliberate non-goals' section, lowercased.

    Scoped to that one section on purpose. The rest of the document explains how
    the modules work and uses the same vocabulary while doing it — the
    ``cache.py`` section has a bullet beginning "Which backends can be cached" —
    so a search over the whole file would flag prose that is describing a
    module rather than denying it.
    """
    match = _NON_GOALS_RE.search(doc)
    assert match, "ARCHITECTURE.md no longer has a '## Deliberate non-goals' section"
    return match.group(1).lower()


def contradicted_denials(non_goals: str) -> list[str]:
    """Return the blanket denials in ``non_goals`` that a real module falsifies.

    Takes the section text rather than reading the file so the detection can be
    exercised on a string, which is what lets the guard below be proven capable
    of failing.
    """
    return [
        f"{phrase!r} (contradicted by nl2sql/{module}.py)"
        for phrase, module in sorted(_BLANKET_DENIALS.items())
        if phrase in non_goals and (PACKAGE_DIR / f"{module}.py").exists()
    ]


def test_non_goals_do_not_deny_a_module_that_exists(non_goals: str) -> None:
    """A non-goal may not blanket-deny something ``nl2sql/`` implements.

    The fix when this fails is to narrow the claim to what is still true, not to
    delete it: the module almost certainly left some genuine limit in place, and
    naming that limit is more useful than either the false absolute or silence.
    """
    contradicted = contradicted_denials(non_goals)
    assert not contradicted, (
        "ARCHITECTURE.md's non-goals deny capabilities the package now has: "
        f"{contradicted}"
    )


def test_the_denial_guard_catches_the_sentence_it_was_written_for() -> None:
    """Pin the guard against the historical claim, and against current prose.

    A guard whose phrase list has gone stale — or was mistyped — passes on every
    input and reads exactly like a guard that is working. The first assertion
    fixes that by replaying the sentence the document actually carried; the
    second checks the narrowed replacement is accepted, so the guard is shown to
    discriminate rather than merely to fire.
    """
    assert contradicted_denials("no caching or persistence layer.") == [
        "'no caching' (contradicted by nl2sql/cache.py)",
    ]
    assert contradicted_denials("no result cache. nl2sql/cache.py caches sql") == []


def test_repair_budget_claim_matches_the_constant(doc: str) -> None:
    from nl2sql import generator

    match = _REPAIR_BUDGET_RE.search(doc)
    assert match, "ARCHITECTURE.md no longer states the repair budget as 'budget of **N attempt...'"
    assert int(match.group(1)) == generator.MAX_REPAIR_ATTEMPTS, (
        f"ARCHITECTURE.md claims a repair budget of {match.group(1)}, "
        f"but generator.MAX_REPAIR_ATTEMPTS is {generator.MAX_REPAIR_ATTEMPTS}"
    )
