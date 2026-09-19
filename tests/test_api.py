"""Tests for the HTTP front end (`nl2sql.api`).

The interesting assertions here are not "does FastAPI route a GET" — that is
FastAPI's own test suite's job. They are the three things this module actually
decides:

* the status code each failure maps to (an unanswerable question, a missing
  database, a query that outruns its deadline),
* the caps a caller cannot exceed, and
* the parameters that deliberately do not exist (``db``, ``llm``), since a
  later edit adding either would be a security regression that no other test
  in the repo would notice.
"""

import os

import pytest

# The API is an optional extra, so a contributor who installed only the core
# dependencies skips this file rather than failing it. CI installs the extra,
# so these do run there.
pytest.importorskip("fastapi", reason="install the 'api' extra to test the HTTP front end")
pytest.importorskip("httpx", reason="fastapi.testclient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402 - after importorskip

from nl2sql import api  # noqa: E402 - after importorskip

DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "store.db")
GOLD = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evals", "gold.jsonl")

needs_db = pytest.mark.skipif(not os.path.exists(DB), reason="sample DB not built")


@pytest.fixture
def client():
    """A client bound to the sample database, whether or not it exists."""
    return TestClient(api.create_app(db_path=DB, gold_path=GOLD))


@pytest.fixture
def clientless_db():
    """A client bound to a database path that does not exist."""
    return TestClient(api.create_app(db_path="/nonexistent/nope.db", gold_path=GOLD))


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #


def test_health_reports_the_bound_database_and_catalog_size(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["database"] == DB
    assert body["rules"] > 0


def test_health_succeeds_but_flags_a_missing_database(clientless_db):
    """Health reports the fault instead of failing on it.

    A liveness check that 503s because the sample data was never built tells an
    operator less than one that answers and names which of the two things is
    wrong.
    """
    response = clientless_db.get("/health")
    assert response.status_code == 200
    assert response.json()["database_present"] is False


# --------------------------------------------------------------------------- #
# /ask
# --------------------------------------------------------------------------- #


@needs_db
def test_ask_returns_sql_and_rows(client):
    body = client.get("/ask", params={"q": "How many customers do we have?"}).json()
    assert "SELECT" in body["sql"].upper()
    assert body["columns"] == ["customer_count"]
    assert body["rows"] == [[120]]
    assert body["row_count"] == 1
    assert body["truncated"] is False


@needs_db
def test_ask_reports_truncation_when_the_row_cap_bites(client):
    """A capped result must say so, or it reads as the complete answer.

    "Revenue by category" returns one row per category (4 in the sample DB), so
    a cap of 2 reliably truncates it.
    """
    body = client.get(
        "/ask", params={"q": "Show revenue by category", "max_rows": 2}
    ).json()
    assert body["row_count"] == 2
    assert body["truncated"] is True


@needs_db
def test_ask_rejects_a_row_cap_above_the_server_limit(client):
    """The caps belong to the server. A caller may lower them, not raise them."""
    over = client.get(
        "/ask", params={"q": "How many customers do we have?", "max_rows": api.MAX_ROWS_LIMIT + 1}
    )
    assert over.status_code == 422
    at_limit = client.get(
        "/ask", params={"q": "How many customers do we have?", "max_rows": api.MAX_ROWS_LIMIT}
    )
    assert at_limit.status_code == 200


@needs_db
def test_ask_rejects_a_deadline_above_the_server_limit(client):
    """There is no "run forever" option, unlike the CLI's --timeout-ms 0."""
    assert (
        client.get(
            "/ask",
            params={"q": "How many customers do we have?", "timeout_ms": api.TIMEOUT_LIMIT_MS + 1},
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/ask", params={"q": "How many customers do we have?", "timeout_ms": 0}
        ).status_code
        == 422
    )


@needs_db
def test_ask_returns_422_and_suggestions_for_an_unanswerable_question(client):
    """The 422 body has to distinguish itself from FastAPI's validation 422.

    Both are 422s on the same endpoint, so the discriminator is the body: a
    catalog miss carries `suggestions`, and every suggestion must be a question
    the service actually answers — an offer that fails the same way the
    caller's question just did would be worse than no offer at all.
    """
    response = client.get("/ask", params={"q": "What is the airspeed of a swallow?"})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "suggestions" in detail

    for suggestion in detail["suggestions"]:
        assert client.get("/ask", params={"q": suggestion}).status_code == 200


@needs_db
def test_ask_suggests_a_near_miss_in_phrasing(client):
    """The suggestions are useful, not merely present."""
    detail = client.get("/ask", params={"q": "revenue for each category please"}).json()[
        "detail"
    ]
    assert any("categor" in s.lower() for s in detail["suggestions"])


def test_ask_returns_503_when_the_database_is_missing(clientless_db):
    response = clientless_db.get("/ask", params={"q": "How many customers do we have?"})
    assert response.status_code == 503
    assert "build_sample_db" in response.json()["detail"]


@needs_db
def test_ask_returns_504_when_the_query_outruns_its_deadline(client):
    """A deadline the query cannot meet is a gateway timeout, not a 500.

    1 ms is below the floor of any real query against the sample DB, which
    makes this deterministic without needing a pathological query.
    """
    response = client.get(
        "/ask", params={"q": "Which products are most frequently bought together?", "timeout_ms": 1}
    )
    assert response.status_code == 504


@needs_db
def test_ask_ignores_unknown_parameters_rather_than_honouring_them(client):
    """`db`, `llm` and `examples` must not be wired up by a later edit.

    All three are deliberate omissions: `db` would make this an arbitrary-file
    reader for anyone who can reach the port, `llm` would let an
    unauthenticated query string spend money, and `examples` would be `db`
    again with a different extension — a caller-supplied path to any JSONL the
    server process can open. FastAPI drops query parameters the signature does
    not declare, so passing them is inert today — this test is what fails if
    someone declares them.
    """
    body = client.get(
        "/ask",
        params={
            "q": "How many customers do we have?",
            "db": "/etc/passwd",
            "llm": "true",
            "examples": "/etc/passwd",
        },
    ).json()
    assert body["rows"] == [[120]]


# --------------------------------------------------------------------------- #
# /explain
# --------------------------------------------------------------------------- #


@needs_db
def test_explain_reports_the_matched_rule_without_running_it(client):
    body = client.get(
        "/explain", params={"q": "How many customers do we have?"}
    ).json()
    assert body["backend"] == "offline"
    assert body["matched_rule"] is not None
    assert body["is_safe"] is True
    assert body["safety_error"] is None
    # A dry run returns SQL, never rows.
    assert "rows" not in body


@needs_db
def test_explain_lists_shadowed_rules(client):
    """Shadowed matches are what make a catalog's ordering auditable."""
    body = client.get(
        "/explain",
        params={"q": "How many orders contain products from more than one category?"},
    ).json()
    assert body["shadowed_rules"], "expected the broad order-count rule to also match"
    assert all(
        {"index", "pattern"} <= set(rule) for rule in body["shadowed_rules"]
    )


@needs_db
def test_explain_returns_422_for_an_unanswerable_question(client):
    response = client.get("/explain", params={"q": "What is the airspeed of a swallow?"})
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# /rules
# --------------------------------------------------------------------------- #


def test_rules_lists_the_whole_catalog_without_a_database(clientless_db):
    """The catalog lives in the process, so it must not need the data file."""
    body = clientless_db.get("/rules").json()
    assert len(body) == api.llm.OfflineBackend().rule_count()
    assert all({"index", "pattern", "example"} <= set(entry) for entry in body)


def test_rules_search_filters_the_listing(client):
    filtered = client.get("/rules", params={"search": "revenue"}).json()
    assert filtered
    assert len(filtered) < len(client.get("/rules").json())


def test_rules_search_with_no_match_is_an_empty_list_not_an_error(client):
    """Departs from the CLI, which exits 1 like grep.

    Over HTTP "nothing covers this topic" is a successful answer to a question
    that was asked correctly; reserving the error channel for actual faults is
    what lets a client treat a non-200 as one.
    """
    response = client.get("/rules", params={"search": "zzzz-no-such-topic"})
    assert response.status_code == 200
    assert response.json() == []


def test_rules_degrades_to_no_examples_when_the_gold_file_is_missing():
    """The rules are the answer; the examples are a convenience on top."""
    client = TestClient(api.create_app(db_path=DB, gold_path="/nonexistent/gold.jsonl"))
    body = client.get("/rules").json()
    assert len(body) == api.llm.OfflineBackend().rule_count()
    assert all(entry["example"] is None for entry in body)


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def test_create_app_prefers_its_argument_over_the_environment(monkeypatch):
    monkeypatch.setenv("NL2SQL_DB", "/from/the/environment.db")
    client = TestClient(api.create_app(db_path=DB, gold_path=GOLD))
    assert client.get("/health").json()["database"] == DB


def test_create_app_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("NL2SQL_DB", "/from/the/environment.db")
    client = TestClient(api.create_app(gold_path=GOLD))
    assert client.get("/health").json()["database"] == "/from/the/environment.db"


def test_openapi_schema_documents_every_route(client):
    """The generated schema is the contract this service publishes."""
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/ask", "/explain", "/rules"} <= set(paths)


def test_no_route_declares_a_withheld_parameter(client):
    """The withheld parameters stay withheld on *every* route, not just /ask.

    The inert-parameter test above can only speak for the endpoint it calls,
    and `/explain` is the one most likely to drift: unlike `/ask` it has a
    CLI counterpart that does take `--examples`, so wiring it up here would
    feel like closing a gap rather than opening one. Reading the published
    schema catches that on whichever route it appears, and catches a path
    parameter or request body as readily as a query string.
    """
    paths = client.get("/openapi.json").json()["paths"]
    declared = {
        (path, param["name"])
        for path, operations in paths.items()
        for operation in operations.values()
        for param in operation.get("parameters", [])
    }
    withheld = {
        name for _, name in declared if name in {"db", "llm", "examples"}
    }
    assert not withheld, f"withheld parameters are now declared: {sorted(withheld)}"
