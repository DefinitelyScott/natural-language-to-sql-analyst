"""Read-only HTTP front end over the offline question catalog.

``nl2sql ask`` answers one question per process, which is the right shape for a
terminal and the wrong one for anything else — a notebook, a dashboard, or a
teammate who does not have the repo checked out. This module serves the same
operations the CLI offers (answer, dry run, list the catalog, report health)
over HTTP, so the analyst-facing surface is reachable from a browser or a
``requests`` call without shelling out.

It is deliberately thin. Every endpoint delegates to the same
:mod:`nl2sql.generator` and :mod:`nl2sql.catalog` functions the CLI calls, so
the HTTP surface cannot drift from the command-line one, and nothing here knows
how to build or execute SQL.

Five decisions worth stating, because they are the ones the design turns on:

* **Offline backend only.** There is no ``llm=true`` parameter. Adding one
  would let an unauthenticated query string spend money on model calls, and
  this service has no authentication to gate that behind. The CLI keeps
  ``--llm`` because the person typing it is the person paying.

* **The database is fixed at startup** — by :func:`create_app` or the
  ``NL2SQL_DB`` environment variable, never by the request. A ``?db=``
  parameter would turn a read-only analytics endpoint into an
  arbitrary-SQLite-file reader for anyone who can reach the port.

* **GET, not POST.** Every operation is a side-effect-free read, so GET is the
  honest verb: cacheable, survivable in a browser address bar, and reachable
  with nothing but ``curl``. The question travels in the query string, which is
  safe here precisely because the database is not also a parameter.

* **The row and time caps are the server's, not the caller's.** The CLI lets
  ``--max-rows`` and ``--timeout-ms`` go as high as the user likes — their
  process, their machine. Over HTTP they are bounded by :data:`MAX_ROWS_LIMIT`
  and :data:`TIMEOUT_LIMIT_MS`, because the caller is not the one who pays for
  an unbounded scan.

* **No ``cached`` or ``repairs`` fields.** Both belong to the LLM path — the
  offline backend is never cached and never repaired — so over an offline-only
  surface they would be constants dressed up as data. The CLI reports them
  because there they can vary.

Requires the optional ``api`` extra::

    pip install -e ".[api]"
    uvicorn nl2sql.api:app

Then read the generated OpenAPI docs at ``/docs``.
"""

from __future__ import annotations

import os
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from . import catalog, generator, llm, runner

# These mirror the CLI's defaults on purpose: both front ends should point at
# the same sample database and the same gold set unless told otherwise, so that
# a question answered in the terminal answers identically over HTTP.
_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_DB_PATH = os.path.join(_REPO_ROOT, "data", "store.db")
DEFAULT_GOLD_PATH = os.path.join(_REPO_ROOT, "evals", "gold.jsonl")

#: Default and maximum rows a single request may return.
DEFAULT_MAX_ROWS = 1000
MAX_ROWS_LIMIT = 10_000

#: Maximum execution deadline a caller may ask for, in milliseconds. There is
#: no "no deadline" option (the CLI's ``--timeout-ms 0``): a request that can
#: occupy the server indefinitely is a denial of service whether or not anyone
#: meant it that way.
TIMEOUT_LIMIT_MS = 30_000

#: Nearest catalog questions offered when nothing matches. Same limit the CLI
#: uses for its "Did you mean" line, for the same reason: three covers a
#: near-miss in phrasing without turning the error into a menu.
SUGGESTION_LIMIT = 3


class HealthResponse(BaseModel):
    """Liveness plus the two facts that explain most failures downstream."""

    status: str = Field(description="Always 'ok' when the service is running.")
    database: str = Field(description="Path the service reads, fixed at startup.")
    database_present: bool = Field(
        description=(
            "False when that file does not exist. Reported rather than raised: "
            "a health check that fails on a missing sample database tells you "
            "less than one that says which of the two things is wrong."
        )
    )
    rules: int = Field(description="Number of rules in the offline catalog.")


class AskResponse(BaseModel):
    """An executed answer: the SQL that ran, and the rows it returned."""

    question: str
    sql: str
    columns: list[str]
    rows: list[list[Any]] = Field(
        description="Row values in column order, as SQLite returned them."
    )
    row_count: int
    truncated: bool = Field(
        description=(
            "True when the row cap cut the result short. Without it a capped "
            "result is indistinguishable from a complete one."
        )
    )


class ShadowedRule(BaseModel):
    """A later rule that also matched but lost under first-match resolution."""

    index: int
    pattern: str


class ExplainResponse(BaseModel):
    """A dry run: the SQL a question resolves to, and how it got there."""

    question: str
    backend: str
    sql: str
    matched_rule: int | None
    matched_pattern: str | None
    shadowed_rules: list[ShadowedRule]
    is_safe: bool
    safety_error: str | None


class RuleEntry(BaseModel):
    """One catalog rule, with an example question that routes to it."""

    index: int
    pattern: str
    example: str | None


def _no_rule_detail(question: str, gold_path: str) -> dict[str, Any]:
    """Build the 422 body for a question no rule answers.

    Suggestions come from the live matcher (via :func:`catalog.build_catalog`)
    rather than straight from the gold file, so every question offered is one
    the service provably answers — a suggestion that would fail the same way
    the caller's did is worse than none.

    A missing or malformed gold file costs the suggestions and nothing else.
    The client's problem is an unrecognized question; failing the request over
    a file they never mentioned would replace a useful error with a confusing
    one.
    """
    try:
        examples = catalog.load_example_questions(gold_path)
    except (OSError, ValueError):
        examples = []

    entries = catalog.build_catalog(llm.OfflineBackend(), examples)
    suggestions = catalog.suggest_questions(
        question, catalog.answerable_questions(entries), limit=SUGGESTION_LIMIT
    )
    return {
        "error": "no offline rule matches this question",
        "question": question,
        "suggestions": suggestions,
    }


def create_app(
    *, db_path: str | None = None, gold_path: str | None = None
) -> FastAPI:
    """Build the application, bound to one database for its whole lifetime.

    A factory rather than a module-level app so tests (and anyone serving a
    database other than the sample one) can bind a path without mutating global
    state or setting an environment variable first. ``app`` below is the
    default instance uvicorn loads.

    Paths resolve in the order: argument, environment (``NL2SQL_DB`` /
    ``NL2SQL_GOLD``), packaged default.
    """
    database = db_path or os.environ.get("NL2SQL_DB") or DEFAULT_DB_PATH
    gold = gold_path or os.environ.get("NL2SQL_GOLD") or DEFAULT_GOLD_PATH

    # One backend for the app's lifetime. Constructing it compiles the whole
    # regex catalog, which is wasted work to repeat per request, and it holds
    # no per-request state.
    backend = llm.OfflineBackend()

    app = FastAPI(
        title="nl2sql analyst",
        version="0.1.0",
        summary="Ask a SQLite analytics database questions in plain English.",
        description=(
            "A read-only HTTP surface over the offline rule catalog. The "
            "database is fixed at startup and every query is validated, "
            "executed read-only, row-capped and deadlined before it returns."
        ),
    )

    def require_database() -> str:
        """Return the database path, or fail the request if it is not there.

        503 rather than 500: the service is correctly configured and the file
        is simply absent, which is an availability problem the operator fixes
        by building it — so the message says how.
        """
        if not os.path.exists(database):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"database not found at {database} — run "
                    "`python scripts/build_sample_db.py`"
                ),
            )
        return database

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        """Report liveness, the bound database, and the catalog size."""
        return HealthResponse(
            status="ok",
            database=database,
            database_present=os.path.exists(database),
            rules=backend.rule_count(),
        )

    @app.get(
        "/ask",
        response_model=AskResponse,
        tags=["query"],
        responses={
            422: {"description": "No rule in the catalog answers this question."},
            503: {"description": "The configured database file is missing."},
            504: {"description": "The query exceeded its execution deadline."},
        },
    )
    def ask(
        q: Annotated[
            str,
            Query(
                min_length=1,
                description="The question, in plain English.",
                examples=["Show revenue by category"],
            ),
        ],
        max_rows: Annotated[
            int, Query(ge=1, le=MAX_ROWS_LIMIT, description="Row cap for this request.")
        ] = DEFAULT_MAX_ROWS,
        timeout_ms: Annotated[
            int,
            Query(
                ge=1,
                le=TIMEOUT_LIMIT_MS,
                description="Cancel the query after this many milliseconds.",
            ),
        ] = runner.DEFAULT_TIMEOUT_MS,
    ) -> AskResponse:
        """Answer a question: generate SQL, execute it read-only, return rows."""
        database_path = require_database()
        try:
            answer = generator.answer_question(
                database_path,
                q,
                use_llm=False,
                use_cache=False,
                max_rows=max_rows,
                timeout_ms=timeout_ms,
            )
        except llm.NoRuleMatchError as exc:
            # 422 rather than 404: the request is well-formed and names no
            # resource — it carries instructions the server understood and
            # could not process, which is what 422 is for. It does collide with
            # FastAPI's own validation 422, so the body carries `suggestions`
            # and lets a client tell the two apart without guessing.
            raise HTTPException(
                status_code=422, detail=_no_rule_detail(q, gold)
            ) from exc
        except runner.QueryTimeoutError as exc:
            # 504: the deadline is the service's own, so this is the server
            # failing to produce a response in time, not a bad request.
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except (runner.UnsafeQueryError, generator.QueryFailedError) as exc:
            # 500, deliberately: the catalog generated SQL that its own
            # validator rejected or the engine could not run. The caller did
            # nothing wrong and can do nothing about it — it is a defect here.
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        result = answer.result
        return AskResponse(
            question=answer.question,
            sql=answer.sql,
            columns=result.columns,
            rows=[list(row) for row in result.rows],
            row_count=len(result),
            truncated=result.truncated,
        )

    @app.get(
        "/explain",
        response_model=ExplainResponse,
        tags=["query"],
        responses={
            422: {"description": "No rule in the catalog answers this question."},
            503: {"description": "The configured database file is missing."},
        },
    )
    def explain(
        q: Annotated[
            str, Query(min_length=1, description="The question, in plain English.")
        ],
    ) -> ExplainResponse:
        """Show the SQL a question resolves to, without executing it.

        Reports which rule matched, which later rules also matched but were
        shadowed by it, and whether the SQL passes the read-only validator.
        Never touches the data, so it is safe to point at SQL you do not trust.
        """
        database_path = require_database()
        try:
            exp = generator.explain_question(
                database_path, q, use_llm=False, use_cache=False
            )
        except llm.NoRuleMatchError as exc:
            raise HTTPException(
                status_code=422, detail=_no_rule_detail(q, gold)
            ) from exc

        return ExplainResponse(
            question=exp.question,
            backend=exp.backend,
            sql=exp.sql,
            matched_rule=exp.matched_rule,
            matched_pattern=exp.matched_pattern,
            shadowed_rules=[
                ShadowedRule(index=index, pattern=pattern)
                for index, pattern in exp.shadowed_rules
            ],
            is_safe=exp.is_safe,
            safety_error=exp.safety_error,
        )

    @app.get("/rules", response_model=list[RuleEntry], tags=["catalog"])
    def rules(
        search: Annotated[
            str | None,
            Query(
                description=(
                    "Show only rules whose example question or pattern "
                    "contains this text (case-insensitive)."
                )
            ),
        ] = None,
    ) -> list[RuleEntry]:
        """List the catalog in matching order, with an example per rule.

        Needs no database: the rules live in the process. An empty list is a
        valid response to a search that matched nothing — over HTTP that is an
        answer ("nothing covers this topic"), not an error, which is where this
        departs from the CLI's grep-like exit code.
        """
        try:
            examples = catalog.load_example_questions(gold)
        except (OSError, ValueError):
            # Costs the examples, not the listing: the rules are the answer
            # here and they come from the backend, not the file.
            examples = []

        entries = catalog.build_catalog(backend, examples)
        if search:
            entries = catalog.filter_catalog(entries, search)
        return [
            RuleEntry(index=e.index, pattern=e.pattern, example=e.example)
            for e in entries
        ]

    return app


#: The default application, for ``uvicorn nl2sql.api:app``.
app = create_app()
