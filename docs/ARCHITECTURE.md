# Architecture

The README explains *what* this project does and how to run it. This document
explains *how it is put together and why* — the module boundaries, the path a
question takes through them, and the design decisions that are not obvious from
reading any single file.

`tests/test_architecture_doc.py` parses this file and checks its claims against
the code: every module has a section here, every symbol named here exists, and
the repair budget quoted below matches `nl2sql.generator.MAX_REPAIR_ATTEMPTS`.
A rename that this document does not follow fails the suite rather than quietly
turning into stale prose.

## The one-sentence shape

A question and a rendered schema go into a **backend**, which returns a SQL
string; that string is untrusted until a **validator plus a SQLite authorizer**
clear it; only then is it executed against a read-only connection, and the rows
come back to a formatter.

```
question ─┐
          ├─► generator ──► llm.Backend ──► SQL string ──► runner ──► QueryResult ──► output
schema ───┘   (orchestrates)  (offline rules            (validate +      (rows)      (table/
                               or LLM call)              authorize +                  csv/json)
                                                         read-only exec)
```

Two properties fall out of drawing the boundary there:

* **The backend has no authority.** It returns text. Everything that decides
  whether that text is allowed to touch the database lives in `runner`, on the
  other side of a function call the backend cannot influence. Swapping a
  deterministic rule matcher for a language model therefore changes what gets
  attempted, never what is permitted.
* **Generation is separable from execution.** Because SQL is an ordinary return
  value rather than something the backend runs itself, a dry run
  (`nl2sql explain`) is not a special mode — it is simply the same pipeline
  stopped one step early.

## Modules

### `nl2sql/schema.py`

Introspects the SQLite database and renders it as compact prompt context —
tables, columns with types, primary and foreign keys, and the distinct values of
columns that behave like categories.

The value hints exist because structure alone does not tell a model which
literals a `WHERE` clause has to match. A guessed category (`'north'` for
`'North'`) does not raise; it returns zero rows, and an empty result is the one
wrong answer neither the read-only validator nor the repair loop can see. Only
non-key text columns with at most `DEFAULT_MAX_DISTINCT` (12) short distinct
values are listed — above that a column is free-form data, and listing it would
copy rows into every prompt. The cap does the classifying, so no per-column
allowlist has to be kept in step with the schema.

The schema is rendered *once per question* and passed down as a plain string,
rather than each layer re-reading the database. That matters for the repair
loop: `nl2sql.generator.answer_question` hands the same schema text to the
retry that the failed query was written from, so a repair cannot be scored
against a schema the first attempt never saw.

### `nl2sql/llm.py`

Holds the `nl2sql.llm.Backend` protocol and its two implementations.

`nl2sql.llm.OfflineBackend` is an ordered list of `(regex, SQL)` rules, matched
**first-rule-wins**. That single choice drives most of the offline design:

* Ordering is semantic, not cosmetic. A narrow pattern must be registered ahead
  of a broad one — "orders in the last 30 days" would otherwise be swallowed by
  "how many orders" — so adding a rule can silently break an existing one.
* Because ordering can break things, it is made inspectable rather than trusted.
  `nl2sql.llm.OfflineBackend.matching_rule_indexes` returns *every* rule a
  question matches, not just the winner. `explain` prints the losers as
  "shadowed", and `tests/test_rule_catalog.py` asserts across the gold set that
  no rule is unreachable.
* `to_sql` routes through that same method instead of short-circuiting on the
  first hit, so the diagnostic and the behaviour cannot drift apart.

`nl2sql.llm.LLMBackend` sends the schema and question to an OpenAI-compatible
chat model at temperature 0. It additionally implements
`nl2sql.llm.RepairingBackend`; `OfflineBackend` deliberately does not, because
its SQL is hand-written and keyed to a fixed rule — re-asking would return the
identical string, so a retry could only burn time. The distinction is a runtime
`isinstance` check against a `runtime_checkable` protocol, which keeps "can this
backend be repaired" a property of the backend rather than a flag the caller has
to remember to set.

### `nl2sql/cache.py`

Reuse of SQL the LLM backend has already written, so a repeated question costs
no model call. Two pieces: `nl2sql.cache.SqlCache`, a JSON file keyed by
`nl2sql.cache.cache_key` digests, and `nl2sql.cache.CachedBackend`, a wrapper
that consults it before delegating.

The design question that matters is *what belongs in the key*, and the answer is
everything that determines the value: the question verbatim, the rendered
schema, and — folded into one opaque `cache_identity` string the backend
supplies — the model name and a fingerprint of the system prompt. The prompt is
the component most easily forgotten and the most damaging to omit: a cache blind
to it replays yesterday's answer after the edit made to change that answer.

Two boundaries keep the wrapper from leaking into the rest of the pipeline:

* **Which backends can be cached is a capability, not a class check.**
  `nl2sql.cache.CacheableBackend` is a `runtime_checkable` protocol satisfied by
  anything that reports a `cache_identity`. `LLMBackend` does; `OfflineBackend`
  deliberately does not, so the default path — and therefore the whole test
  suite and CI — never touches a cache file.
* **A wrapper must not change what it wraps.** `CachedBackend` defines `repair`,
  so wrapping a backend that cannot repair would advertise a capability the
  delegate lacks and break `answer_question`'s protocol check from the inside.
  It refuses that at construction instead.

A hit and a miss are indistinguishable downstream — same SQL, same validator,
same execution — so `nl2sql.cache.CachedBackend.lookup` returns the hit flag
alongside the SQL and the CLI reports it, rather than any of the pipeline
branching on it. Reads and writes are best-effort throughout: an unreadable or
foreign-format file is a miss, an unwritable directory costs the caching and
nothing else. An optimization that can break an answer is not one.

### `nl2sql/generator.py`

Orchestration, and the only module that knows the full sequence.

Two entry points are kept separate on purpose:
`nl2sql.generator.answer_question` generates **and executes**;
`nl2sql.generator.explain_question` generates and inspects **without touching
the data**. Splitting them is what makes the dry run trustworthy — an
explanation has no code path that reaches `runner.run`, so pointing it at SQL
you do not yet trust is safe by construction rather than by discipline.

`answer_question` owns the repair loop, with a budget of **1 attempt**
(`nl2sql.generator.MAX_REPAIR_ATTEMPTS`). The repairs it fixes are shallow ones
— a hallucinated column, a function SQLite lacks — and the model is told exactly
what went wrong, so a second failure almost always means a misread schema that
more retries will not resolve. A small fixed budget also bounds the worst-case
cost of one question to two model calls and two executions, instead of leaving
latency and token spend open-ended.

Failed attempts are recorded on the `Answer` rather than discarded, and the CLI
reports them on stderr. An answer that needed a retry is weaker evidence than
one that ran first time, and hiding that would overstate how well the backend
performed.

### `nl2sql/runner.py`

The trust boundary. Generated SQL passes three independent layers:

1. `nl2sql.runner.validate` — a string-level check (single statement,
   SELECT/WITH only, denylisted write and DDL keywords). It runs first because
   it produces specific, actionable error messages.
2. A SQLite **authorizer** callback, consulted by the engine for every operation
   while a statement compiles, allowing only `SELECT`, `READ`, `FUNCTION` and
   `RECURSIVE`.
3. A wall-clock **deadline** (`nl2sql.runner.DEFAULT_TIMEOUT_MS`), armed by
   `nl2sql.runner._install_deadline` as a progress handler and reported as
   `nl2sql.runner.QueryTimeoutError`.

The two layers are not redundant, and the order is not arbitrary. A denylist can
only reject what someone thought to name; the authorizer inverts the question
and refuses anything not on a short allowlist, which covers constructs the regex
never anticipated. Running the readable check first means the common case — a
model that emitted two statements — produces a sentence a human can act on
rather than a generic `not authorized`. Opening the connection with `mode=ro`
protects the target file; only the authorizer covers the whole engine surface
(`ATTACH`, for instance, reaches *other files on disk*).

The first two layers answer *may this run*; the third answers *for how long*.
They are different questions, and the first two cannot reach the second: a
cross join is read-only, single-statement, and touches nothing outside the
allowlist, yet would run until the process was killed. The deadline is a
progress handler rather than a watchdog thread because SQLite invokes it on the
calling thread between virtual-machine instructions — a cancellation point the
engine is already prepared for, with no concurrency to get wrong. The handler
sets its own flag before requesting the abort, so `run` can tell a deadline
apart from any other `OperationalError` without matching on message text.
`QueryTimeoutError` subclasses `RuntimeError`, not `sqlite3.DatabaseError`,
which is what keeps it out of the repair loop in `nl2sql.generator`.

`nl2sql.runner.run` fetches one row beyond `max_rows` purely as a probe: a
cursor gives no "there is more" signal, so a result that happens to be exactly
`max_rows` long is otherwise indistinguishable from one that was cut short. The
extra row is discarded and only its existence is reported, as
`QueryResult.truncated` — which is what lets an export warn instead of silently
dropping rows.

### `nl2sql/output.py`

Three pure formatters — `nl2sql.output.format_table`,
`nl2sql.output.format_csv`, `nl2sql.output.format_json` — sharing a
`(columns, rows)` signature so the CLI can pick one by name.

Only the table formatter takes a row limit. A terminal preview should stay
readable; an export should be complete, so the machine-readable formats apply no
limit of their own. They still cannot *promise* completeness, because
`runner.run` capped the rows before they ever arrived — hence the truncation
warning, which the CLI writes to stderr in every format so it can never land
inside a redirected CSV or JSON file.

### `nl2sql/catalog.py`

Pairs each offline rule with an example question, for `nl2sql rules`.

The examples are read from the evaluation gold set rather than kept in a second
hand-maintained list, and the pairing is recomputed from the live matcher on
every call. So an example cannot go stale: if a newly added broad rule starts
shadowing an older one, the older rule loses its example immediately instead of
continuing to advertise a question that no longer reaches it. A rule with no
example is listed as `example=None` rather than dropped — an entry nothing
reaches is precisely the defect the catalog tests exist to catch, so hiding it
would hide the evidence.

The same pairing powers the "Did you mean" line `ask` and `explain` print when
no rule matches. `nl2sql.catalog.suggest_questions` ranks candidates by Jaccard
overlap of content words, with a `difflib` character ratio as tiebreaker only —
overlap is the readable signal (a suggestion is offered because it names the
same things), while raw string similarity would let a long candidate outrank a
short exact-topic match. Candidates are drawn through
`nl2sql.catalog.answerable_questions`, so every suggestion is one the live
matcher provably routes to a rule; suggesting a question that would fail the
same way the user's just did would be worse than suggesting nothing. When
nothing overlaps, nothing is offered — a list padded to a fixed length reads as
a guess and costs the reader a check per entry.

### `nl2sql/cli.py`

Argument parsing and presentation only; it holds no analytical logic.

The consistent rule is that **stdout carries data and stderr carries
diagnostics**. The generated SQL, repair notices and truncation warnings all go
to stderr in `csv`/`json` mode, which is what makes
`nl2sql ask "…" --format csv > sales.csv` produce a clean file. Exit codes
follow the same intent: `explain` exits 1 on SQL the validator would reject and
`rules --search` exits 1 on no matches, so both can gate a shell script without
anyone parsing prose.

## Evaluation

`evals/evaluate.py` is the part of the repo that makes any quality claim
checkable, and it is deliberately not a unit test.

It measures **execution accuracy**: generate SQL, run it, run the gold query,
and compare the *result sets*. String-matching SQL would penalise a correct
query written differently, which measures phrasing rather than correctness.
Column names are excluded from the comparison for the same reason — a correct
query may alias differently.

Comparison is order-insensitive by default, with each gold row carrying an
`ordered` flag. For a ranking ("top 5 customers") or a sequence ("revenue by
month") the row order *is* part of the answer, and comparing those
order-insensitively would over-credit the backend on exactly the questions where
ordering is the hard part.

Every question yields a record, not just a tally, because a bare accuracy number
tells you *that* something regressed and never *why*. `--json` archives those
records and `--compare` diffs a run against them: two runs can post identical
accuracy while failing a different set of questions, so a per-question diff is
the only way to see a regression and a fix cancelling each other out.

## Testing strategy

Tests are grouped by the invariant they defend, not by module:

* **Behaviour** — `test_offline_backend.py`, `test_runner.py`, `test_repair.py`,
  `test_output.py`, `test_schema.py`, `test_cli.py`, `test_explain.py`,
  `test_suggestions.py`.
* **Catalog invariants** — `test_rule_catalog.py` pins one gold question per
  rule and asserts no rule is unreachable or shadowed into inertness.
* **Harness correctness** — `test_evaluate.py`, `test_report_comparison.py`.
  The thing that grades the system needs grading too; a harness that scores a
  wrong answer as correct is worse than no harness.
* **Documentation** — `test_docs.py` checks the figures quoted in the README
  against `evals/gold.jsonl`, and `test_architecture_doc.py` checks this file
  against the package. Prose that claims a number is a claim like any other, and
  a stale figure in a README reads as a metric measured once and never
  re-measured.

CI runs ruff, then the suite on Python 3.10–3.12, then the evaluation harness —
which exits non-zero unless every gold question passes, so a regression in
generated SQL fails the build even when every unit test still passes.

## Deliberate non-goals

* **The offline backend is not a parser.** It is a fixed catalog of recognised
  question patterns, chosen so the repo is fully runnable and verifiable with no
  API key. Growing it into a general NL grammar would trade the property that
  makes it useful — you can read any rule and know exactly what it does — for
  coverage the LLM backend already provides.
* **No caching or persistence layer.** Every question re-introspects the schema
  and re-runs the query. At this database size that is imperceptible, and the
  absence of stale state keeps the whole pipeline reproducible from a clean
  clone.
* **No write path, ever.** Not "writes are discouraged" — the authorizer denies
  them at compile time, so there is no code path through this project that can
  modify the database.
