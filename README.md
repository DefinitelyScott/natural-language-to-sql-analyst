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
│   └── cli.py           # `nl2sql ask "..."` / `explain "..."` / `schema`
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
in `WHERE`); revenue by price tier — fixed-threshold `CASE` *banding* of a
continuous variable, the complement of the `NTILE` spend quartiles below: bands
have fixed, meaningful boundaries whose populations move with the data, while
quantiles have equal-count populations whose boundaries move. Tiers are assigned
from the catalog list price, not the transacted unit price, so a product cannot
straddle bands if a sale happened at a different price.

**Time series** — total sales by month, revenue by quarter, revenue by day of
week, new customers by month, unique active customers per month, orders in the
last 30 days.

**Period-over-period comparison** — first-half vs second-half 2024 revenue for
every product, with the absolute and percentage change between them. This is the
catalog's only *conditional aggregation*: `SUM(CASE WHEN ... THEN ... ELSE 0 END)`
pivots two date ranges into side-by-side columns in a single pass, so the change
the question asks for can be subtracted within one row. `ELSE 0` (not `ELSE
NULL`) keeps a product sold in only one half in the result instead of letting
NULL propagate through the subtraction and drop exactly the products whose
change is most extreme; `NULLIF` guards the percentage against a product with no
first-half sales.

**Window functions** — month-over-month revenue growth (`LAG`); cumulative
running-total revenue by month (an explicit `ROWS BETWEEN UNBOUNDED PRECEDING
AND CURRENT ROW` frame); a 7-day moving average of daily revenue (a *bounded*
`ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` frame over a gap-free calendar spine
— the catalog's only recursive CTE and only `LEFT JOIN`; without the spine's
zero-filled days, a rows-based frame would silently span more than a week
whenever a day had no orders); the top-spending customer within each region (a
partitioned greatest-N-per-group ranking via `ROW_NUMBER()`); customer spend
quartiles (`NTILE(4)`); and the average gap between a customer's consecutive
orders (a partitioned `LAG` over `order_date`).

**Per-order statistics** — average order value overall, by region, and as a
monthly trend; median order value (a `LIMIT`/`OFFSET` middle-row query, since
SQLite has no `MEDIAN` function); average units per order (basket size); and
multi-category orders — the count and share of orders mixing more than one
product category (basket *breadth*, the cross-sell measure basket size is not).
A CTE computes `COUNT(DISTINCT category)` per order so repeated products in one
category count once, and the share's denominator is the `orders` table itself so
the percentage stays "of all orders" rather than "of orders with items".
This group also holds the catalog's only *drill-down*: the 10 largest orders by
value, returned as individual orders with their date and customer attached.
Every other rule aggregates rows away; this one surfaces the specific outliers
an aggregate points at — a spike in a monthly total is usually one or two
unusually large orders, and this is the query that finds them.

**Customer behavior (RFM-style)** — top 5 customers by spend; repeat customers;
average revenue per customer; average customer lifespan; at-risk (lapsed)
customers, whose most recent order predates a recency cutoff anchored to the
data's newest order; the new-vs-returning revenue split; the distribution of
orders per customer (a nested aggregation — a purchase-frequency histogram); and
market-basket affinity (the product pairs most often bought together, via a
self-join of `order_items`).

**Revenue concentration** — the Pareto ("80-20") view: customers are ranked by
lifetime revenue, split into five equal groups with `NTILE(5)`, and each quintile
reports its revenue, its share of total revenue, and the *cumulative* share
through that quintile. The cumulative column is what separates this from the
spend quartiles below — quartiles say how much a tier is worth, concentration
says how much of the whole it accounts for, which is the question a running share
answers and a per-bucket total does not. (On this synthetic dataset the curve is
shallow: the top quintile takes ~36% of revenue, not 80%. The sample data is
generated from a uniform draw, so it has no long tail to find.)

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

## Explaining a query before running it

`explain` is a dry run: it generates the SQL for a question and reports how that
SQL was produced, without executing anything.

```bash
python -m nl2sql.cli explain "How many orders were placed in the last 30 days?"
```

```
Question: How many orders were placed in the last 30 days?
Backend:  offline

Matched offline rule #28: orders.*last (30|thirty) days
Also matched (shadowed, in catalog order):
  #38: how many orders|number of orders|total orders|order count

SQL (not executed):
  SELECT COUNT(*) AS recent_orders FROM orders WHERE order_date >= date((SELECT MAX(order_date) FROM orders), '-30 day')

Safety: passes the read-only validator.
```

Two things it is good for:

- **Auditing catalog ordering.** Matching is first-rule-wins, so a rule that
  matches but is registered later is inert. `explain` names those shadowed rules
  explicitly, which is how you tell a deliberate ordering from an accidental one
  when adding a pattern. (`tests/test_rule_catalog.py` enforces the same
  property across the whole gold set; `explain` is the interactive view of it.)
- **Pre-flighting untrusted SQL.** With `--llm` the model's output is shown and
  run through the same read-only validator `ask` uses — but never executed. The
  command exits `1` when the SQL would be rejected, so it can gate a script the
  way a linter does.

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

- single-statement, `SELECT`-only execution (no `INSERT/UPDATE/DELETE/DDL`),
  checked at the string level first because a string check can say *why* it
  refused ("multiple statements are not allowed");
- an **engine-level authorizer** as a second, independent layer: SQLite
  consults a callback for every operation while compiling a statement, and
  anything outside a four-item read-only allowlist (`SELECT`, table/column
  reads, function calls, recursive CTEs) is denied. The string check is a
  denylist and a denylist only rejects what it thought to name — the
  authorizer inverts that, so a construct the regex never anticipated (e.g.
  an `ATTACH`, which would open other files on disk) is still refused;
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
Evaluated 44 questions  |  execution accuracy: 44/44 (100%)  [offline backend]
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
  "total": 44,
  "passed": 44,
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
  `"ordered": true` and are compared as returned. 30 of the 44 gold questions
  are order-sensitive.

The flag is a judgment about the question, not a mechanical "does the gold SQL
have an `ORDER BY`" check — a gold query may sort purely so its output reads
nicely (one row per region, listed alphabetically) without the order carrying
any meaning. Those rows are deliberately left unordered.

Those counts are not maintained by hand. `tests/test_docs.py` parses them back
out of this README and compares them to `evals/gold.jsonl`, so adding a question
without refreshing the numbers fails the suite.

## Tests and linting

```bash
pytest -q
ruff check .   # same config CI runs, from pyproject.toml
```

CI runs the lint as its own job alongside the test matrix. The ruff ruleset is
deliberately curated rather than `ALL` — each enabled family (pyflakes,
bugbear's `zip(strict=)` and mutable-default checks, unused-argument,
blind-except, private-member access) catches a class of defect this codebase
actively guards against, so any violation is signal. Tests are exempted from
the unused-argument and private-member rules: stubs implement a protocol's full
signature without consulting it, and reaching into internals is what a unit
test is for.

## License

MIT — see [LICENSE](LICENSE).
