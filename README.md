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
│   └── cli.py           # `nl2sql ask "..."`
├── scripts/build_sample_db.py   # generates a synthetic retail database
├── evals/
│   ├── gold.jsonl       # question / gold-SQL pairs
│   └── evaluate.py      # result-set comparison harness
└── tests/               # pytest suite
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

1. **Offline (default).** A small rule-based matcher handles a fixed catalog of
   analytical question patterns — from simple counts and group-by aggregations
   to per-order averages (order value and basket size), time-series buckets
   (by month, by quarter, and by day of week) and
   window-function queries for month-over-month revenue growth, each
   category's share of total revenue, and the top-spending customer within
   each region (a partitioned greatest-N-per-group ranking). Deterministic,
   free, and used by the test suite and CI. This keeps the repo runnable and
   verifiable by anyone who clones it.
2. **LLM.** If `OPENAI_API_KEY` is set and you pass `--llm`, the question and the
   rendered schema are sent to an OpenAI-compatible chat model, which returns
   SQL. The generated SQL still passes through the same read-only guardrails.

```bash
export OPENAI_API_KEY=sk-...
python -m nl2sql.cli ask "Which 5 customers spent the most last year?" --llm
```

## Exporting results

By default `ask` prints a human-readable table (truncated to 20 rows for
readability). Pass `--format csv` or `--format json` to get the full result set
in a machine-readable form. In these modes only the data is written to stdout —
the generated SQL goes to stderr — so you can redirect straight to a file:

```bash
python -m nl2sql.cli ask "Show revenue by category" --format csv > revenue.csv
python -m nl2sql.cli ask "Show revenue by region" --format json > revenue.json
```

## Safety guardrails

Generated SQL is never trusted blindly. `runner.py` enforces:

- single-statement, `SELECT`-only execution (no `INSERT/UPDATE/DELETE/DDL`);
- a connection opened in read-only mode;
- a row cap on returned results.

## Evaluating quality

`evals/evaluate.py` loads `gold.jsonl`, generates SQL for each question, executes
both the generated and gold queries, and compares the resulting tables. It
reports execution accuracy (fraction of questions whose generated result set
matches the gold result set).

```
$ python evals/evaluate.py
Evaluated 16 questions  |  execution accuracy: 16/16 (100%)  [offline backend]
```

Run it against the LLM backend with `--llm` to benchmark a model.

## Tests

```bash
pytest -q
```

## License

MIT — see [LICENSE](LICENSE).
