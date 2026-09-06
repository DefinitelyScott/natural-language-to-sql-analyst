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
│   ├── catalog.py       # pairs each offline rule with an example question
│   ├── cache.py         # on-disk reuse of SQL the LLM backend already wrote
│   └── cli.py           # `nl2sql ask` / `explain` / `rules` / `schema`
├── scripts/build_sample_db.py   # generates a synthetic retail database
├── evals/
│   ├── gold.jsonl       # question / gold-SQL pairs (+ order-sensitivity flag)
│   ├── precision.jsonl  # questions the catalog must decline (over-match guard)
│   ├── paraphrases.jsonl # rephrasings that must route to the same rule
│   └── evaluate.py      # result-set comparison harness
├── docs/ARCHITECTURE.md # module boundaries, data flow, and design rationale
├── CONTRIBUTING.md      # how to add a question pattern, and the invariants to keep
└── tests/               # pytest suite (incl. test_docs.py, which checks the
                         # counts quoted below against evals/gold.jsonl)
```

For *why* the pieces fit together the way they do — where the trust boundary
sits, why offline matching is first-rule-wins, why the repair budget is one —
see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). For the step-by-step of adding
a question pattern — where a rule has to sit, when it needs the unscoped-only
guard, what the gold row's `ordered` flag changes — see
[CONTRIBUTING.md](CONTRIBUTING.md).

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
   SQL. The generated SQL still passes through the same read-only guardrails,
   and if it fails to execute the model gets one chance to rewrite it — see
   [Repairing SQL that fails](#repairing-sql-that-fails).

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
(`SUM(...) OVER ()`); category purchase penetration — the share of the buyer
base that bought from each category, which is the *reach* complement of that
revenue share and can disagree with it (a cheap add-on can touch nearly every
customer off a thin slice of revenue, and a big-ticket line can do the reverse).
The buyer set is de-duplicated on (category, customer) so a repeat buyer counts
once, and the denominator is the customers who placed any order rather than
`COUNT(*) FROM customers`, so never-buying customers do not dilute a reach
figure they cannot contribute to; the denominator is returned as its own column
so the percentage can be re-derived from what is printed. This rule is
registered ahead of the revenue-share rule, which would otherwise answer "what
share of customers bought from each category" with revenue percentages.
Also here: categories whose revenue beats the mean (a scalar subquery
in `WHERE`); revenue by price tier — fixed-threshold `CASE` *banding* of a
continuous variable, the complement of the `NTILE` spend quartiles below: bands
have fixed, meaningful boundaries whose populations move with the data, while
quantiles have equal-count populations whose boundaries move. Tiers are assigned
from the catalog list price, not the transacted unit price, so a product cannot
straddle bands if a sale happened at a different price.
This group also holds the catalog's only *compound* `SELECT`: revenue by
category with a grand-total row appended. Every other rule returns rows from a
single query; this one `UNION ALL`s two result sets at different levels of
aggregation, because SQLite has neither `ROLLUP` nor `GROUPING SETS`. The total
is derived from the same CTE as the category rows rather than re-scanned from
`order_items`, so the report *foots* by construction — it is the sum of the rows
printed above it, not a second measurement that could disagree at the cent.
`ORDER BY (category = 'Total')` uses SQLite's 0/1 booleans to pin the total last
whatever the revenue ordering does, which matters because the total is by
construction the largest value in the column and `revenue DESC` alone would sort
it to the top.

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

**Window functions** — month-over-month revenue growth (`LAG`); the same trend
broken out per product category (a `LAG` partitioned by category, so each
category is compared against its own previous month rather than across the
category boundary — this rule is registered ahead of both the whole-business
growth rule and the plain "revenue by category" rule, because either would
otherwise answer a two-dimension question by silently dropping one of the
dimensions and still returning a plausible-looking table); cumulative
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
Also here: the repeat purchase rate per product — of the customers who ever
bought a product, the share who came back and bought it again. This is the
product-level counterpart of the repeat-customer count, and it measures
something that count cannot: whether an *individual* product earns a second
purchase (a consumable) or is bought once and done (a durable). The modelling
decision that makes it a re-purchase measure rather than a basket-size one is
`COUNT(DISTINCT o.id)` — a repeat buyer bought the product on two separate
*orders*, so three units in one basket is still one purchase decision. The
denominator is each product's own buyer base, not the customer table, so the
rate is comparable across products with very different reach, and both counts
are returned beside the percentage so a rate computed off a thin buyer base is
visible rather than hidden. It is registered ahead of the broad repeat-customer
rule, which would otherwise answer "which products have the most repeat
customers?" with a single business-wide number that silently drops the product
dimension.
This group also holds the catalog's only *relational division* (a "for all"
query): the customers who ordered in every quarter of 2024. Every other rule
asks which rows satisfy a condition — an EXISTS-shaped question — but "in
*every* quarter" quantifies over a set, which no plain `WHERE` clause can
express. The division is written as `HAVING COUNT(DISTINCT quarter) = 4` rather
than the textbook double-`NOT EXISTS`: one aggregate reads as "covered four
distinct quarters" instead of burying the same test in two layers of negation.
The `DISTINCT` is load-bearing (five orders all in Q1 cover one quarter, not
five), and the 4 is deliberately a constant — it is a fact of the calendar, so
deriving it from the quarters present in the data would quietly weaken the test
in exactly the case where a quarter had no orders at all.

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

**Activation** — time to first order, by signup cohort: for each month of
signups, how many of those customers ever ordered (the activation rate) and how
long the ones who did took to get there. It is the only pattern anchored on
`customers.signup_date` as a per-customer clock rather than as a bucket to count
signups in, and the only one whose join has to be a `LEFT JOIN` — a customer who
never ordered is precisely the numerator's complement, and an inner join would
drop them and report 100% activation for every month.

Two reading caveats are inherent to the metric rather than to this
implementation, and both are why the report is split by cohort instead of
blended into one average. The newest cohorts are *censored*: they have had less
time to convert than older ones, so a declining activation rate down the last
rows is a property of the observation window. And a cohort that signed up before
the first order in the database carries the gap to that date inside its average,
which is why the 2023 cohorts show a far larger `avg_days_to_first_order` than
the 2024 ones. Both are visible per cohort and invisible in a single number.

**Acquisition mix** — which category each customer's *first* order came from,
reported as the share of the customer base each category brought in. Activation
above asks whether and how quickly a signup converts; this asks what they
converted *on*. Its shape is two stacked `ROW_NUMBER()` windows: one picks a
single first order per customer — over `(order_date, id)` rather than
`MIN(order_date)`, since two orders can share a date and `MIN` would then match
both and count that customer twice — and one keeps the highest-spending category
within that order.

That second window is an attribution choice, and the one thing worth defending
here: 78 of the 115 customers who have ordered in the sample data have a first
order spanning more than one category, so "the" acquiring category is not
something the data contains — it has to be defined. Attributing the customer to
the category they spent the most in treats the largest line as the reason for the
visit and partitions the base exactly once, so `customers_acquired` sums to the
number of customers who have ordered and `pct_of_customers` sums to 100. Counting
the customer once per category present in the order is equally defensible and
answers a slightly different question, but it produces a percentage column
summing to ~196% on this data, which reads as a bug in a report even when it is
not. The cost of the choice is that a narrowly-lost second category leaves no
trace; that is a basket question, and the market-basket pattern above already
covers what rides along in an order.

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

### Listing the catalog from the command line

The prose above describes the catalog; `rules` prints it. Each row is a rule in
matching order, with an example question that actually routes to it:

```bash
python -m nl2sql.cli rules
python -m nl2sql.cli rules --search region
python -m nl2sql.cli rules --format json
```

```
4 offline rule(s), in matching order:

rule  example                                           pattern
6     Who is the top-spending customer in each region?  (top|best|highest)[-\s]*(spending|spender)?\s*custo...
12    Show revenue by region and category.              (revenue|sales).*(region.*categor|categor.*region)|...
19    What is the average order value by region?        (average|avg).*order value.*region
33    Show revenue by region                            (revenue|sales).*by region
```

(Patterns abbreviated here for width; the command prints them in full.)

This is the discoverability counterpart to `explain`: `explain` says how a
question you already have resolves, `rules` says which questions exist to ask.
The rule numbers are the same ones `explain` reports, so a dry run and the
listing can be read against each other.

The examples are not a second hand-maintained list — they are drawn from
`evals/gold.jsonl` and re-matched through the live router on every run, so a
rule is only ever shown with a question that currently reaches it. If a newly
added broad pattern starts shadowing an older rule, that rule loses its example
here (and prints `(no example)`) the moment it happens, rather than continuing
to advertise a question that now routes elsewhere. A rule with no example is
listed rather than hidden, for the same reason.

`rules` needs no database — it inspects the in-process rule catalog only. A
`--search` that matches nothing exits 1, following grep, so a script can gate on
whether the catalog covers a topic.

### When the catalog has no rule for your question

`ask` and `explain` do not just fail — they offer the nearest questions the
catalog *can* answer, on stderr:

```bash
python -m nl2sql.cli ask "revenue by store"
```

```
Error: Offline backend has no rule for this question. Set OPENAI_API_KEY and use --llm for open-ended questions.

Did you mean:
  Show revenue by category
  Show revenue by region
  What is the total revenue?

Run `nl2sql rules` to list every question the offline backend answers, or `nl2sql rules --search <text>` to filter it.
```

Candidates are ranked by how many content words they share with what you typed
(filler words like "how" and "the" are ignored), with a character-similarity
tiebreak. They come from the same live-matched pairing `rules` prints, so every
question offered is one the backend provably answers rather than one that would
fail the same way yours did.

A question sharing no content word with the catalog gets no suggestions — only
the pointer to `rules`. Padding the list to a fixed length would read as a guess
and cost you a check per entry:

```bash
python -m nl2sql.cli ask "what is the meaning of life?"
```

```
Error: Offline backend has no rule for this question. Set OPENAI_API_KEY and use --llm for open-ended questions.

Run `nl2sql rules` to list every question the offline backend answers, or `nl2sql rules --search <text>` to filter it.
```

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
python -m nl2sql.cli schema --no-values   # structure only
```

Text columns holding at most **12 distinct values** are annotated with those
values:

```
TABLE customers (
  id INTEGER PRIMARY KEY,
  name TEXT,
  region TEXT,  -- one of: 'East', 'North', 'South', 'West'
  signup_date TEXT
)
```

Structure alone would tell a model that `region` exists and holds text, but not
that the four strings in it are capitalised compass points. Asked to filter on
one, a model has to guess the literal — `'north'`, `'Northern'`, a region that
is not in the data — and a wrong guess does not raise an error. It returns zero
rows, which reads like a real answer to a question with no matches. Neither the
read-only validator nor the repair loop catches that, because nothing failed;
listing the values is what prevents it.

The cap is what makes this safe to leave on. Columns above it — `signup_date`,
`order_date`, customer names — are free-form data, not categories, and are left
out: listing them would bloat every prompt and copy row data into it for no
gain. Keys are skipped by kind for the same reason. Pass `--no-values` (or
`schema_context(..., max_distinct=0)`) for structure only.

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
  capped result is never mistaken for a complete one;
- an **execution deadline** of **5000 ms** by default, enforced by a SQLite
  progress handler that checks the clock every thousand virtual-machine
  instructions and cancels the statement once the budget is spent.

The first two layers decide whether a query may run; neither bounds how long it
runs for. A read-only `SELECT` can still be ruinously expensive — an accidental
cross join is the usual way, and it is exactly the mistake a model writing SQL
from a schema makes — and without a deadline it would hold the process open
indefinitely. `--max-rows` does not help: it bounds how much comes back, not
how much work SQLite does to produce it.

```bash
python -m nl2sql.cli ask "Show revenue by category" --timeout-ms 250
python -m nl2sql.cli ask "Show revenue by category" --timeout-ms 0   # no deadline
```

A cancelled query raises `runner.QueryTimeoutError`, which deliberately sits
*outside* the `sqlite3.DatabaseError` hierarchy. That is what keeps it out of
the repair loop below: a repair is worth making when the engine names something
to fix, and a deadline names nothing — the query may be perfectly correct and
merely expensive, so a rewrite would be a guess costing another full timeout.

## Repairing SQL that fails

An LLM writing SQL from a schema gets things wrong in a specific, recognizable
way: a column that does not exist, a join on the wrong key, a function SQLite
does not have. The engine names the problem precisely when it refuses to
compile the query — so `answer_question` hands that error, and the query that
caused it, back to the backend for one rewrite:

```
question ──► SQL ──► run ──► result
               ▲       │
               │       ▼ (error)
               └── repair(sql, error)      at most once
```

Details that matter:

- **The rewrite is re-validated.** It goes back through the same validator,
  authorizer and read-only connection as the first attempt, so the loop can
  change *what* is run, never what is allowed to run. A repair that returns
  `DROP TABLE` is rejected exactly as the first attempt would have been.
- **One attempt, not "until it works."** The errors a repair actually fixes are
  shallow ones, and the model is told exactly what was wrong; if a second
  rewrite is needed, the cause is usually a misreading of the schema that more
  retries will not resolve. A fixed budget also keeps the worst case of one
  question bounded and obvious — at most two model calls.
- **Guardrail rejections are repairable too.** A model that emitted two
  statements has made an ordinary mistake and can be told so; since the rewrite
  is re-validated, nothing rejected on the first pass can slip through on the
  second.
- **A repair is disclosed, not hidden.** Each failed attempt is recorded on
  `Answer.repairs` and reported on stderr by `ask`, because a result that
  needed a retry is a weaker signal than one that ran first time.
- **The offline backend opts out.** Repairing is an optional protocol
  (`llm.RepairingBackend`); offline SQL is hand-written and keyed to a fixed
  rule, so re-asking would return the same string. When the budget is exhausted
  — or the backend cannot repair at all — the failure surfaces as a
  `QueryFailedError` listing every attempt, rather than a raw SQLite traceback.

`tests/test_repair.py` drives the loop with scripted stub backends, so the
retry, the budget and the guardrail-on-rewrite property are all covered offline,
with no API key.

## Caching generated SQL

At `temperature=0`, asking a model the same question against the same schema
returns the same query — so `ask --llm` and `explain --llm` write what they
generate to `data/sql_cache.json` and replay it next time. Repeating a question
then costs nothing and returns instantly, and iterating on a rule or a report
stops being metered by the API.

```bash
python -m nl2sql.cli ask "Which 5 customers spent the most?" --llm   # calls the model
python -m nl2sql.cli ask "Which 5 customers spent the most?" --llm   # replays, no call
python -m nl2sql.cli ask "Which 5 customers spent the most?" --llm --no-cache
```

A cache is only as good as its key, and this one covers **every input that
decides the answer**:

- **the question, verbatim** — not lowercased or whitespace-collapsed, because
  the model gets the exact string and normalizing would assert an equivalence
  the cache is in no position to verify;
- **the rendered schema**, so rebuilding the database makes answers written
  against the old shape unreachable rather than wrong;
- **the model name**, so switching models does not replay the old one;
- **a fingerprint of the system prompt**, so editing the prompt invalidates
  everything written under it. This is the one people forget, and it is the one
  that bites: a cache blind to the prompt hides the exact change you edited the
  prompt to see.

Some deliberate limits:

- **Only the SQL is cached, never the rows.** The data can change under a
  cached query; the query text the model writes for fixed inputs cannot.
- **Repairs are not cached.** A repair is keyed on the SQL that failed and the
  error it raised, which only exist after an execution.
- **Failures are not cached.** A generation that raises is retried for real
  next time, not replayed as a permanent failure.
- **The offline backend is never cached.** It resolves a question with a regex
  scan, so a cache would only add a file read to the cheapest path there is.
- **The eval harness does not use the cache.** Scoring replayed answers would
  measure the cache, not the backend.
- **A cached answer is disclosed, not hidden.** `ask` notes the replay on
  stderr and `explain` prints a `Source: local cache` line, for the same reason
  a repair is reported: you should never mistake a reproduction for a fresh run.

The cache file is plain JSON, capped at 500 entries, gitignored, and safe to
delete at any time. A corrupt or unreadable one reads as a miss and is rewritten
on the next successful generation — an optimization must never be able to break
an answer.

## Evaluating quality

`evals/evaluate.py` loads `gold.jsonl`, generates SQL for each question, executes
both the generated and gold queries, and compares the resulting tables. It
reports execution accuracy (fraction of questions whose generated result set
matches the gold result set).

```
$ python evals/evaluate.py
Evaluated 51 questions  |  execution accuracy: 51/51 (100%)  [offline backend]
```

Run it against the LLM backend with `--llm` to benchmark a model.

The harness executes generated SQL directly and does **not** run the repair
loop, so what it reports is first-attempt accuracy. That is the number worth
tracking when comparing prompts or models — folding in a retry would let a
weaker first attempt hide behind a second one — but it means the figure
understates what `ask --llm` recovers from in practice.

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
  "total": 51,
  "passed": 51,
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

### Paraphrase robustness

Execution accuracy has a blind spot the gold set cannot see past: it holds
exactly one phrasing per rule. A rule that matches only its own gold question
still scores 100%, and a newly added, broadly phrased rule inserted ahead of an
older one can quietly capture some of the older rule's phrasings without ever
touching the one phrasing the gold set uses.

`evals/paraphrases.jsonl` pins that down. Each record pairs a gold question with
an alternate phrasing and asserts the two route to the *same rule*:

```json
{"canonical": "Show revenue by category", "paraphrase": "Break down sales by category"}
```

Routing is compared, not the SQL. Two rules can emit identical SQL today and
diverge tomorrow, so equal SQL would let a paraphrase drift onto the wrong rule
unnoticed. Anchoring each canonical to a gold question is what makes routing
enough: the gold row already proves the rule they both reach returns the right
answer.

A record may instead carry a `known_gap` — a phrasing the catalog does *not*
reach today, with the reason:

```json
{"canonical": "Show revenue by region", "paraphrase": "Break down revenue per region",
 "known_gap": "the region rule accepts only 'by region', while the category rule accepts 'by' or 'per'"}
```

Known gaps are recorded rather than left out. A set assembled only from
phrasings that already work would measure nothing about the matcher's reach, and
would quietly reward narrowing a rule. They are reported but do not fail the
run, and they are excluded from the ratio's denominator, so documenting a gap
can never improve the headline. The set currently holds **46 gating pairs and
9 known gaps**:

```
Paraphrase robustness: 46/46 rephrasings route to the canonical rule
  Known gaps (not gating): 9
```

Every rule the gold set reaches carries at least one rephrasing, and
`tests/test_paraphrase_guard.py` asserts it: a ratio is only as wide as the set
behind it, and an earlier version of this set covered 31 of the catalog's rules
while still printing 100%. Widening it to the whole catalog immediately found a
real defect — *"for every region, which customer spent the most?"* was falling
through to the global top-spenders rule, which answered it with one overall
ranking and silently dropped the per-region grouping.

The check gates the exit code for the same reason the precision guard does: a
paraphrase that drifts onto another rule still returns a well-formed, correctly
labelled table, so no accuracy number moves when it breaks. A known gap that
starts routing correctly is flagged as `[NOW ROUTING]` — the fix is to drop its
`known_gap` and let it join the gating set, so a later regression cannot undo it
silently. `tests/test_paraphrase_guard.py` fails on a stale one, and on the two
counts quoted above.

### Gold independence

Execution accuracy compares a generated query against a gold query, and that
comparison only carries information when the two were **written separately**. If
the gold SQL for a question is the offline rule's own SQL, the harness runs one
query twice and compares it to itself. The match is then guaranteed by
construction — it would still hold if the rule computed revenue by *region* for a
question asking about categories — so the row proves the SQL parses and executes,
and nothing about whether it answers what was asked.

That is not a hypothetical here. The harness measures it and prints it under the
other two checks:

```
Gold independence: 43/51 gold queries are written independently of the rule they test
  Self-comparing (not gating): 8 — these prove the SQL runs, not that it answers the question
    [COPY] rule #25: How many customers do we have?
    [COPY] rule #43: Show revenue by day of week.
    ...
```

So eight rows of the 100% above are still self-referential, and the honest
reading of the headline is "51/51, of which 43 are real comparisons". Publishing
that number is the point: an eval set is a claim about a system, and a claim
nobody has audited for tautologies is worth less than a smaller one that has
been.

The fix is per-question — rewrite the gold query a different way that computes
the same answer (a different join order, a subquery where the rule uses a CTE, a
window function where the rule uses `ORDER BY ... LIMIT`) — so the backlog is
worked down rather than cleared at once. A rewrite only counts if it reaches the
answer by a different route: restating `COUNT(*) FROM products` as
`COUNT(DISTINCT id) FROM products` clears the text comparison without adding a
second opinion, which raises the ratio while proving nothing. The whole-table
counts still on the list are there for that reason, and may never come off it.
Until the backlog is worked down it is a **ratchet**, not a gate:
`tests/test_gold_independence.py` records the 8 remaining copies by name and
fails if a new one appears, so a pattern added with copy-pasted gold SQL is
caught immediately, while the existing backlog stays visible instead of turning
every run red. It also fails if a rewritten query is left on the list, so the
backlog can only shrink. The harness measures and reports; the test decides what
is allowed to change.

Six rows have been rewritten so far, each taking a different route to the same
answer: per-order subtotals instead of one flat sum over the join fan-out
(monthly sales), a correlated subquery instead of `JOIN` plus `GROUP BY` (top
customers), aggregation before the customer join instead of after (largest
orders), `DISTINCT` in a subquery instead of `COUNT(DISTINCT ...)` (monthly
active customers), `julianday` arithmetic instead of a `date(..., '-30 day')`
string comparison (orders in the last 30 days), and the join order reversed
(revenue by region and category). Because `gold.jsonl` cannot carry comments,
the disagreement each rewrite is now capable of producing is recorded in
`REWRITE_RATIONALE` in `tests/test_gold_independence.py`, and a test keeps that
list from outliving the rows it describes.

Two limits worth stating. The comparison normalizes whitespace, trailing
semicolons and case, and stops there — it is not a SQL parser, so a gold query
that is a copy with one column renamed reads as independent. The count is
therefore a **lower bound**: the true number of tautological rows can only be
higher than the one printed. And independence is a property of the *offline*
run only; under `--llm` the model writes its own SQL and cannot copy a gold query
it never saw, so the check is skipped rather than reported as vacuously perfect.

### What counts as a matching result

Two details decide whether the reported accuracy is meaningful:

- **Column names are ignored.** A correct query may alias `revenue` as
  `total_revenue`; penalizing that would measure phrasing, not correctness.
- **Row order is checked only where it is part of the answer.** Each gold row
  carries an `ordered` flag. For a scalar aggregate ("how many customers do we
  have?") order is meaningless and rows are compared as a set. For a *ranking*
  ("the top 5 customers by spend") or a *sequence* ("revenue by month"), the
  right rows in the wrong order are a wrong answer, so those rows set
  `"ordered": true` and are compared as returned. 36 of the 51 gold questions
  are order-sensitive.

The flag is a judgment about the question, not a mechanical "does the gold SQL
have an `ORDER BY`" check — a gold query may sort purely so its output reads
nicely (one row per region, listed alphabetically) without the order carrying
any meaning. Those rows are deliberately left unordered.

Those counts are not maintained by hand. `tests/test_docs.py` parses them back
out of this README and compares them to `evals/gold.jsonl`, so adding a question
without refreshing the numbers fails the suite.

## Tests, linting and types

```bash
pytest -q
ruff check .   # same config CI runs, from pyproject.toml
mypy           # --strict over nl2sql/, evals/, scripts/; config in pyproject.toml
```

The shipped code type-checks clean under `mypy --strict`, enforced by CI as its
own job against the oldest supported interpreter. That is a gate rather than a
badge: turning it on surfaced a protocol (`llm.RepairingBackend`) that declared
only the method it adds and not the one every holder of it also calls, and a
sample-data builder that reused one variable name for two different row shapes.
`tests/` is excluded on purpose — a test double stubs most of the protocol it
stands in for, so strict annotations there would cost noise and buy nothing.

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
