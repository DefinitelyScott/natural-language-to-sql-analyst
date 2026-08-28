"""Keep ``CONTRIBUTING.md`` in step with the repo it gives instructions for.

A contributor guide rots the same way an architecture document does — nothing
breaks when a file is renamed or a pin is bumped out from under it — but it
fails worse. An architecture document that has drifted misleads a reader; a
contributor guide that has drifted hands them commands that do not run and a
checklist whose steps no longer exist, which is exactly when they are least
equipped to tell the guide is wrong.

These tests make its concrete claims falsifiable, in the same style as
``test_docs.py`` (README numbers) and ``test_architecture_doc.py`` (module
sections and symbols). Five claims are checked:

* **Every repo path it names exists.** Paths appear in backticks; each is
  resolved against the repo root, so a rename that the guide does not follow
  fails here.
* **Every CLI subcommand it shows is real.** The guide teaches the workflow
  through ``python -m nl2sql.cli <subcommand>`` invocations; each is run with
  ``--help`` so a removed or renamed subcommand fails rather than being
  discovered by a contributor typing it.
* **The three verification steps are the ones the push helper runs.** The guide
  claims ``scripts/push.sh`` runs the same three; if the helper stops running
  one, the guide is advertising a gate that no longer exists.
* **The Python floor matches ``pyproject.toml``.** A contributor sizes their
  interpreter against this number.
* **The lint configuration it quotes is current** — the ruff version CI pins,
  the line length, and the selected rule families.

``pyproject.toml`` is read with a regex rather than ``tomllib`` on purpose:
``tomllib`` arrived in 3.11 and this project's floor — the very claim the fourth
test checks — is 3.10, so parsing it that way would make the test unrunnable on
the oldest interpreter CI supports.

Each test asserts its pattern matched *something* before checking the matches,
so rephrasing the surrounding prose into a shape these tests cannot read fails
loudly instead of silently disabling the check.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nl2sql import cli

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = REPO_ROOT / "CONTRIBUTING.md"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PUSH_SCRIPT_PATH = REPO_ROOT / "scripts" / "push.sh"

# A backticked repo-relative path: `nl2sql/llm.py`, `evals/gold.jsonl`,
# `.github/workflows/ci.yml`. Only extensions of files tracked in git are
# listed — `data/store.db` is generated and gitignored, so requiring it to
# exist would fail on a fresh clone before the build script has been run.
_PATH_RE = re.compile(r"`([\w./-]+\.(?:py|md|jsonl|toml|yml|sh|txt))`")
# "python -m nl2sql.cli explain" — the subcommand the guide is demonstrating.
_SUBCOMMAND_RE = re.compile(r"python -m nl2sql\.cli (\w+)")
# "Python **3.10** is the floor" / `requires-python = ">=3.10"`.
_PYTHON_FLOOR_RE = re.compile(r"Python \*\*(\d+\.\d+)\*\* is\s+the floor")
_REQUIRES_PYTHON_RE = re.compile(r'requires-python\s*=\s*">=\s*(\d+\.\d+)"')
# "CI pins **ruff 0.16.2**" / `pip install ruff==0.16.2`.
_RUFF_VERSION_RE = re.compile(r"ruff (\d+\.\d+\.\d+)\*\*")
_RUFF_PIN_RE = re.compile(r"ruff==(\d+\.\d+\.\d+)")
# "Lines wrap at **100** characters" / `line-length = 100`.
_LINE_LENGTH_RE = re.compile(r"Lines wrap at \*\*(\d+)\*\* characters")
_LINE_LENGTH_CONFIG_RE = re.compile(r"^line-length\s*=\s*(\d+)", re.MULTILINE)
# "The selected families are `E`, `W`, ... and `SLF`" — the sentence listing them.
_FAMILY_SENTENCE_RE = re.compile(r"The selected families are\s+([^.]+)\.", re.DOTALL)
_BACKTICKED_CODE_RE = re.compile(r"`([A-Z]+)`")
# `select = [ "E", # comment ... ]` in the ruff lint config.
_SELECT_BLOCK_RE = re.compile(r"^select\s*=\s*\[(.*?)^\]", re.MULTILINE | re.DOTALL)

#: The commands the guide presents as the three checks, in the form that must
#: also appear in ``scripts/push.sh``. Matched as substrings because the helper
#: invokes them through a ``$PYTHON`` variable rather than a literal ``python``.
_VERIFICATION_STEPS = ("scripts/build_sample_db.py", "pytest -q", "evals/evaluate.py")


@pytest.fixture(scope="module")
def doc() -> str:
    return DOC_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def pyproject() -> str:
    return PYPROJECT_PATH.read_text(encoding="utf-8")


def test_every_path_named_in_the_guide_exists(doc: str) -> None:
    paths = sorted(set(_PATH_RE.findall(doc)))
    assert paths, "CONTRIBUTING.md no longer names any repo files in backticks"

    missing = [path for path in paths if not (REPO_ROOT / path).exists()]
    assert not missing, (
        f"CONTRIBUTING.md points contributors at files that do not exist: {missing}"
    )


def test_every_cli_subcommand_shown_is_real(doc: str) -> None:
    """Each ``python -m nl2sql.cli <sub>`` in the guide must parse.

    ``--help`` exercises the argument parser without touching the database, so
    this checks the subcommand exists without depending on a built ``store.db``.
    argparse exits 0 after printing help; any other exit code means the
    subcommand was rejected.
    """
    subcommands = sorted(set(_SUBCOMMAND_RE.findall(doc)))
    assert subcommands, "CONTRIBUTING.md no longer demonstrates any nl2sql.cli commands"

    unknown: list[str] = []
    for subcommand in subcommands:
        with pytest.raises(SystemExit) as exit_info:
            cli.main([subcommand, "--help"])
        if exit_info.value.code != 0:
            unknown.append(subcommand)

    assert not unknown, (
        f"CONTRIBUTING.md shows nl2sql.cli subcommands that do not exist: {unknown}"
    )


def test_the_three_checks_are_the_ones_push_sh_runs(doc: str) -> None:
    """The guide's three checks and the push helper's gate must not diverge.

    The guide tells a contributor that ``scripts/push.sh`` runs the same three
    steps and refuses to push if any fails. If the helper drops one, the guide
    is advertising a safety net that is no longer there — the failure mode is a
    contributor trusting a gate instead of running the check themselves.
    """
    push_script = PUSH_SCRIPT_PATH.read_text(encoding="utf-8")
    for step in _VERIFICATION_STEPS:
        assert step in doc, f"CONTRIBUTING.md no longer lists '{step}' as a check"
        assert step in push_script, (
            f"CONTRIBUTING.md claims scripts/push.sh runs '{step}', but it does not"
        )


def test_python_floor_matches_pyproject(doc: str, pyproject: str) -> None:
    claimed = _PYTHON_FLOOR_RE.search(doc)
    assert claimed, (
        "CONTRIBUTING.md no longer states the Python floor as 'Python **X.Y** is the floor'"
    )

    configured = _REQUIRES_PYTHON_RE.search(pyproject)
    assert configured, "pyproject.toml no longer declares requires-python as '>=X.Y'"

    assert claimed.group(1) == configured.group(1), (
        f"CONTRIBUTING.md claims a Python floor of {claimed.group(1)}, "
        f"but pyproject.toml requires-python is >={configured.group(1)}"
    )


def test_ruff_version_matches_the_ci_pin(doc: str) -> None:
    claimed = _RUFF_VERSION_RE.search(doc)
    assert claimed, "CONTRIBUTING.md no longer states the pinned ruff version as 'ruff X.Y.Z**'"

    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    pinned = _RUFF_PIN_RE.search(workflow)
    assert pinned, "ci.yml no longer pins ruff with 'ruff==X.Y.Z'"

    assert claimed.group(1) == pinned.group(1), (
        f"CONTRIBUTING.md says CI pins ruff {claimed.group(1)}, "
        f"but ci.yml installs ruff=={pinned.group(1)}"
    )


def test_line_length_matches_pyproject(doc: str, pyproject: str) -> None:
    claimed = _LINE_LENGTH_RE.search(doc)
    assert claimed, (
        "CONTRIBUTING.md no longer states the wrap column as 'Lines wrap at **N** characters'"
    )

    configured = _LINE_LENGTH_CONFIG_RE.search(pyproject)
    assert configured, "pyproject.toml no longer sets a ruff line-length"

    assert claimed.group(1) == configured.group(1), (
        f"CONTRIBUTING.md says lines wrap at {claimed.group(1)} characters, "
        f"but pyproject.toml sets line-length = {configured.group(1)}"
    )


def test_lint_families_match_pyproject(doc: str, pyproject: str) -> None:
    """The families the guide names are exactly the ones ruff is configured with.

    Compared as sets: the guide lists them in configuration order for
    readability, but the claim being made is about membership, and pinning the
    order too would fail on a harmless reordering of the config.
    """
    sentence = _FAMILY_SENTENCE_RE.search(doc)
    assert sentence, (
        "CONTRIBUTING.md no longer lists the lint families in a "
        "'The selected families are ...' sentence"
    )
    claimed = set(_BACKTICKED_CODE_RE.findall(sentence.group(1)))
    assert claimed, "CONTRIBUTING.md's lint-family sentence names no backticked family codes"

    block = _SELECT_BLOCK_RE.search(pyproject)
    assert block, "pyproject.toml no longer has a multi-line ruff lint 'select = [...]' block"
    configured = set(re.findall(r'"([A-Z]+)"', block.group(1)))

    assert claimed == configured, (
        f"CONTRIBUTING.md names lint families {sorted(claimed)}, "
        f"but pyproject.toml selects {sorted(configured)}"
    )
