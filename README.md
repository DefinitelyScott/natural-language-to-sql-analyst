# nl2sql-analyst

A natural-language-to-SQL analytics assistant. Ask a question in plain English,
get back a SQL query, the results, and a short explanation — over a real SQLite
database. Ships with an **offline mode** so the whole project runs and is testable
without any API key, plus an optional LLM backend for open-ended questions.

This project was built to demonstrate three things together: relational data
modeling and SQL, clean Python packaging and testing, and practical LLM
engineering (prompting, guardrails, and evaluation).

## Why this exists

"Text-to-SQL" demos are easy to fake and hard to trust. The interesting part is
not generating *a* query — it's knowing whether the query is *correct*. So the
centerpiece here is an **evaluation harness** that runs generated SQL against a
gold query and compares the *result sets*, not the SQL strings. That is the
honest way to measure a text-to-SQL system.

## What's inside

```
nl2sql-analyst/
├── nl2sql/
│   ├── schema.py        # introspect the DB and render schema context for prompts
│   ├── llm.py           # LLM client (OpenAI-compatible) + offline rule-based fallback
│   ├── generator.py     # orchestrates NL question -> SQL
│   ├── runner.py        # read-only, guarded SQL execution
│   ├── output.py        # table / CSV / JSON result formatters
│   └── cli.py           # `nl2sql ask "..."` / `nl2sql schema`
├── scripts/build_sample_db.py   # generates a synthetic retail database
├── evals/
│   ├── gold.jsonl       # question / gold-SQL pairs (+ order-sensitivity flag)
│   └── evaluate.py      # result-set comparison harness
└── tests/               # pytest suite (incl. test_docs.py, which checks the
                         # counts quoted below against evals/gold.jsonl)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Build the sample database (synthetic retail data)
python scripts/build_sample_db.py

# 2. Ask a question (offline mode — no API key needed)
python -m nl2sql.cli ask "What were total sales by month in 2024?"

# 3. Run the evaluation harness
python evals/evaluate.py
```

Example output:

```
Question: What were total sales by month in 2024?

SQL:
  SELECT strftime('%Y-%m', o.order_date) AS month,
         ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
  FROM orders o
  JOIN order_items oi ON oi.order_id = o.id
  WHERE o.order_date >= '2024-01-01' AND o.order_date < '2025-01-01'
  GROUP BY month
  ORDER BY month;

Results (12 rows):
  month    revenue
  2024-01  18432.55
  2024-02  17790.12
  ...
```

## Offline mode vs. LLM mode

`nl2sql` resolves a question to SQL in two ways:

1. **Offline (default).** A rule-based matcher over a fixed catalog of
   analytical question patterns (listed below). Deterministic, free, and used by
   the test suite and CI, which keeps the repo runnable and verifiable by anyone
   who clones it.
2. **LLM.** If `OPENAI_API_KEY` is set and you pass `--llm`, the question and the
   rendered schema are sent to an OpenAI-compatible chat model, which returns
   SQL. The generated SQL still passes through the same read-only guardrails.

```bash
export OPENAI_API_KEY=sk-...
python -m nl2sql.cli ask "Which 5 customers spent the most last year?" --llm
```

### The offline question catalog

Matching is **first-rule-wins**, so specific patterns are registered ahead of
broad ones — "orders in the last 30 days" must not be swallowed by "how many
orders". The catalog currently covers:

**Counts and totals** — customer count, order count, product count, total
revenue.

**Group-by breakdowns** — revenue by category, by region, and by region ×
category; top products by revenue; best-selling product by units, both overall
and within each category; each category's share of total revenue
(`SUM(...) OVER ()`); categories whose revenue beats the mean (a scalar subquery
in `WHERE`).

**Time series** — total sales by month, revenue by quarter, revenue by day of
week, new customers by month, unique active customers per month, orders in the
last 30 days.

**Window functions** — month-over-month revenue growth (`LAG`); cumulative
running-total revenue by month (an explicit `ROWS BETWEEN UNBOUNDED PRECEDING
AND CURRENT ROW` frame); the top-spending customer within each region (a
partitioned greatest-N-per-group ranking via `ROW_NUMBER()`); customer spend
quartiles (`NTILE(4)`); and the average gap between a customer's consecutive
orders (a partitioned `LAG` over `order_date`).

**Per-order statistics** — average order value overall, by region, and as a
monthly trend; median order value (a `LIMIT`/`OFFSET` middle-row query, since
SQLite has no `MEDIAN` function); average units per order (basket size).

**Customer behavior (RFM-style)** — top 5 customers by spend; repeat customers;
average revenue per customer; average customer lifespan; at-risk (lapsed)
customers, whose most recent order predates a recency cutoff anchored to the
data's newest order; the new-vs-returning revenue split; the distribution of
orders per customer (a nested aggregation — a purchase-frequency histogram); and
market-basket affinity (the product pairs most often bought together, via a
self-join of `order_items`).

**Customer segmentation** — RFM scoring: every buying customer is scored 1–5 on
recency, frequency and monetary value with three `NTILE(5)` windows, and labelled
with the combined RFM cell. This is the pattern that *composes* the single-lens
rules above — at-risk is recency-only, spend quartiles are monetary-only — which
matters because a customer can be a heavy spender and still be lapsing. The
recency window sorts `DESC` while the other two sort `ASC`, so that 5 always
means "best"; inverting that is the classic RFM bug and it is invisible in the
output, so a test pins the sort direction of each window.

**Cohort analysis** — monthly cohort retention: customers are grouped into
acquisition cohorts by the month of their *first* order, and each cohort's
retention is reported per month offset (one row per grid cell). This is the only
pattern that measures behavior relative to each customer's own start date rather
than the calendar, so the month offset is computed as arithmetic on `YYYY-MM`
(year × 12 + month) rather than with `julianday()`, which measures days and would
drift across months of unequal length.

Every pattern in this catalog has a matching row in `evals/gold.jsonl`, so each
one is measured by the evaluation harness rather than merely asserted here.

First-rule-wins has one failure mode: a pattern registered behind a broader one
that also matches its questions can never win, so it becomes dead code that
still *looks* implemented. `tests/test_rule_catalog.py` guards the whole catalog
against that — it resolves every gold question and asserts the mapping from
questions to rules is one-to-one, so a rule that no question reaches (shadowed,
or missing a gold row) and a rule that two questions share both fail the suite.
`OfflineBackend.matching_rule_indexes()` makes this checkable by returning
*every* rule a question matches, not just the winning one; `to_sql` takes the
first entry of that same list, so the diagnostic and the router cannot drift.

## Exporting results

By default `ask` prints a human-readable table (previewing 20 rows for
readability). Pass `--format csv` or `--format json` for a machine-readable
form with no preview limit. In these modes only the data is written to stdout —
the generated SQL goes to stderr — so you can redirect straight to a file:

```bash
python -m nl2sql.cli ask "Show revenue by category" --format csv > revenue.csv
python -m nl2sql.cli ask "Show revenue by region" --format json > revenue.json
```

Every format is still bounded by the `--max-rows` safety cap (default 1000).
A capped result is *flagged*, not silently shortened: `runner` fetches one row
past the cap purely to learn whether more existed, and the CLI writes a warning
to stderr (never to stdout, so a redirected export stays clean) naming the cap
and how to raise it. This matters most for exports — a partial CSV that looks
complete is worse than one that admits it is partial.

## Inspecting the schema

`schema` prints the introspected schema exactly as it is rendered into the LLM
prompt — the same text the model sees. `--counts` adds per-table row counts,
which is handy for sanity-checking the sample data build:

```bash
python -m nl2sql.cli schema
python -m nl2sql.cli schema --counts
```

## Safety guardrails

Generated SQL is never trusted blindly. `runner.py` enforces:

- single-statement, `SELECT`-only execution (no `INSERT/UPDATE/DELETE/DDL`);
- a connection opened in read-only mode;
- a row cap on returned results, reported via `QueryResult.truncated` so a
  capped result is never mistaken for a complete one.

## Evaluating quality

`evals/evaluate.py` loads `gold.jsonl`, generates SQL for each question, executes
both the generated and gold queries, and compares the resulting tables. It
reports execution accuracy (fraction of questions whose generated result set
matches the gold result set).

```
$ python evals/evaluate.py
Evaluated 38 questions  |  execution accuracy: 38/38 (100%)  [offline backend]
```

Run it against the LLM backend with `--llm` to benchmark a model.

### Per-question results

A single accuracy number tells you *that* a backend regressed, not *why*. Every
question therefore produces a structured record — the SQL that was generated,
whether the run errored or merely disagreed, the row counts on both sides, and
the first row where the two result sets diverge. Failures print that diagnostic
inline:

```
Failures:
  - What is the average order value?  [mismatch: first differing row (in sorted order, index 0): generated (612.44) vs gold (598.31)]
  - Show revenue by quarter in 2024.  [error: UnsafeQueryError: only SELECT/WITH queries are allowed]
```

A row-count difference is reported on its own, because when the two result sets
are different lengths the first positional disagreement is usually an artifact
of the misalignment rather than the real defect. The reported index is read
against the ordering the comparison actually used, which is why the message says
which one that was: order-sensitive questions are compared as returned, the rest
in sorted order.

`--json` writes the same records to a file, so a CI run can archive them and two
runs can be diffed:

```bash
python evals/evaluate.py --json eval-report.json
```

```json
{
  "backend": "offline",
  "total": 38,
  "passed": 38,
  "execution_accuracy": 1.0,
  "questions": [
    {
      "question": "What were total sales by month in 2024?",
      "ordered": true,
      "status": "pass",
      "gold_sql": "SELECT strftime('%Y-%m', o.order_date) AS month, ...",
      "generated_sql": "SELECT strftime('%Y-%m', o.order_date) AS month, ...",
      "generated_rows": 12,
      "gold_rows": 12,
      "detail": null
    }
  ]
}
```

`execution_accuracy` is a fraction rather than a rounded percentage so a machine
consumer keeps full precision; only the console line rounds.

### Comparing two runs

Accuracy alone cannot answer "did anything break". Two runs can post the same
number while failing a *different* set of questions — a regression and a fix
cancelling out — which is the normal case when benchmarking prompt or model
changes against the LLM backend, where accuracy is rarely 100%. `--compare`
diffs the current run against a report previously written by `--json` and
buckets every question by what changed:

```bash
python evals/evaluate.py --llm --json baseline.json   # record a baseline
# ...change the prompt or model...
python evals/evaluate.py --llm --compare baseline.json
```

The comparison block is printed after the usual run summary:

```
Comparison vs baseline  |  accuracy 89.5% -> 89.5%
  [REGRESSED] What is the best-selling product in each category?
  [fixed] What is the median order value?
```

Questions present in only one report are reported as `[new question]` or
`[dropped question]` rather than as a fix or a regression: a newly added gold
question that fails means the gold set grew to cover something the backend never
handled, not that something broke. Reports are keyed by question text, so a
report containing the same question twice is rejected rather than diffed
ambiguously.

The diff deliberately does not change the exit code. A regressed question is by
definition failing now, so it already makes the run exit non-zero; gating on it
a second time would add a condition that can never fire on its own. The
comparison's job is to say *which* questions moved.

### What counts as a matching result

Two details decide whether the reported accuracy is meaningful:

- **Column names are ignored.** A correct query may alias `revenue` as
  `total_revenue`; penalizing that would measure phrasing, not correctness.
- **Row order is checked only where it is part of the answer.** Each gold row
  carries an `ordered` flag. For a scalar aggregate ("how many customers do we
  have?") order is meaningless and rows are compared as a set. For a *ranking*
  ("the top 5 customers by spend") or a *sequence* ("revenue by month"), the
  right rows in the wrong order are a wrong answer, so those rows set
  `"ordered": true` and are compared as returned. 25 of the 38 gold questions
  are order-sensitive.

The flag is a judgment about the question, not a mechanical "does the gold SQL
have an `ORDER BY`" check — a gold query may sort purely so its output reads
nicely (one row per region, listed alphabetically) without the order carrying
any meaning. Those rows are deliberately left unordered.

Those counts are not maintained by hand. `tests/test_docs.py` parses them back
out of this README and compares them to `evals/gold.jsonl`, so adding a question
without refreshing the numbers fails the suite.

## Tests

```bash
pytest -q
```

## License

MIT — see [LICENSE](LICENSE).
