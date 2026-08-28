# Contributing

This is a small, single-maintainer project, so this file is less about process
and more about the invariants that are easy to break by accident. Most of them
are enforced by tests; this document is where the *reasoning* behind them lives,
so a failing assertion reads as a rule rather than a puzzle.

If you are looking for what the modules do and why the boundaries sit where they
do, read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first. This file assumes it.

## Getting set up

The runtime has no third-party dependencies — it is standard library only, which
is deliberate (see the non-goals in `docs/ARCHITECTURE.md`). Python **3.10** is
the floor, matching `requires-python` in `pyproject.toml`; CI runs the suite on
3.10, 3.11 and 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/build_sample_db.py
```

`scripts/build_sample_db.py` writes `data/store.db`, which is gitignored. It is
seeded, so everyone's copy holds the same rows and any number quoted in a doc or
a test is reproducible rather than local to one machine. Rebuild it after
changing the generator — the gold SQL in `evals/gold.jsonl` is checked against
whatever is on disk, so a stale database fails the evaluation harness in a way
that looks like a backend regression.

## The three checks

```bash
python scripts/build_sample_db.py
pytest -q
python evals/evaluate.py
```

These are the same three steps `scripts/push.sh` runs, in the same order, and it
refuses to push unless all three succeed. `.github/workflows/ci.yml` runs them
again on every push and pull request, plus `ruff check .` as a separate job.

Running them locally before committing is not a formality. The evaluation
harness is the only check that executes the catalog's SQL against real rows: the
unit tests can confirm a question routes to the rule you meant, and still pass
while that rule returns the wrong numbers.

## Adding a question to the offline catalog

This is the most common change, and the one with the most non-obvious rules. The
offline backend resolves a question by scanning an ordered list of regexes and
taking the **first** match, so where a rule sits is part of its behaviour.

Before writing anything, check whether the catalog already covers the topic:

```bash
python -m nl2sql.cli rules --search revenue
```

Then:

1. **Write the SQL first and run it.** Point it at `data/store.db` and read the
   result. A rule whose regex is perfect and whose SQL is subtly wrong is the
   worst outcome here, because it returns a plausible, correctly-labelled table.

2. **Register the rule** in `OfflineBackend.__init__` (`nl2sql/llm.py`) as a
   `(compiled pattern, sql)` pair, placed *ahead* of any broader rule that also
   matches your phrasings. "Orders in the last 30 days" has to be registered
   before "how many orders", or it can never win.

3. **If the SQL aggregates a whole table with no date filter**, prefix the
   pattern with `_UNSCOPED_ONLY`. Broad, unscoped rules are phrased loosely on
   purpose, so a question that scopes the same aggregate to a period ("revenue
   last quarter") would otherwise fall through to them and be answered with the
   unscoped total — a right-looking number for a different question. The
   negative lookahead makes the rule decline instead, which routes the question
   to `NoRuleMatchError` and the "Did you mean" suggestions.

4. **Add exactly one row** to `evals/gold.jsonl`:
   `{"question": ..., "sql": ..., "ordered": ...}`. Set `ordered` to `true` when
   the row order is part of the answer — a ranking ("top 5 customers") or a
   sequence ("revenue by month"). Leave it `false` when the question is a set.
   The harness compares order-insensitively unless the flag is set, so a
   mislabelled ranking is scored more leniently than it should be.

5. **Comment the rule** with why it sits where it does and any SQL decision a
   reader would otherwise have to reverse-engineer (why a `LEFT JOIN`, why
   `COUNT(DISTINCT ...)`, why a constant rather than a derived value).

6. **Add unit tests** to `tests/test_offline_backend.py`: that the phrasings you
   intend route to the new rule, and that it neither shadows nor is shadowed by
   the neighbours it is most likely to collide with. `explain` is the tool for
   working that out — it prints the rule that won and every rule that matched
   but was shadowed by it:

   ```bash
   python -m nl2sql.cli explain "which products sell best in each category?"
   ```

7. **Describe the pattern** in the README's catalog section, in the group it
   belongs to. Say what it measures and what modelling decision makes it that
   measure rather than a neighbouring one.

8. **Run the three checks.** `tests/test_rule_catalog.py` resolves every gold
   question over the whole catalog and asserts the question-to-rule mapping is
   one-to-one, so it fails if the new rule is unreachable behind a broader one,
   or if it starts answering a question that belonged to another rule. The
   evaluation harness additionally re-routes every record in
   `evals/paraphrases.jsonl`, which is what catches a new rule stealing a
   rephrasing that used to belong to an older one.

## Adding a question the database cannot answer

`evals/precision.jsonl` is the other half of the measurement. Execution accuracy
over the gold set only measures recall — of the questions the catalog claims,
how many it answers correctly. It cannot see a rule that grew broad enough to
start answering questions it was never meant to, because those questions are not
in the gold set.

So when a plausible analytics question has no answer in this schema — profit
margin (no cost data), shipping cost (no fulfilment data) — add it to
`evals/precision.jsonl` with the reason. The guard asserts that **no** rule
matches it, and reports the offending rule when one does.

## Recording a rephrasing

`evals/paraphrases.jsonl` covers the third failure the other two checks miss.
The gold set holds exactly one phrasing per rule, so a rule that matches only
its own gold question scores 100% — and a broad new rule inserted ahead of an
older one can capture some of the older rule's phrasings without touching the
one phrasing the gold set uses.

When you add a rule, add one or two of the phrasings you tested it with as
paraphrase records against the gold question they belong to:

```json
{"canonical": "Show revenue by category", "paraphrase": "Break down sales by category"}
```

The check asserts both route to the same rule. If a natural phrasing does *not*
reach the rule, record it anyway with a `known_gap` explaining why. Gaps are
reported but do not fail the run and are excluded from the ratio's denominator,
so documenting one can never flatter the number — and if a later rule starts
routing it correctly, the run flags it as `[NOW ROUTING]` so you can drop the
`known_gap` and let the pair start gating.

## Documentation is tested

Three files carry claims that are checked back against the code, because a stale
number in a doc reads as a metric that was measured once and never re-measured:

| File | Pinned by | What is checked |
| --- | --- | --- |
| `README.md` | `tests/test_docs.py` | gold-question counts, the sample accuracy figures, the default query deadline |
| `docs/ARCHITECTURE.md` | `tests/test_architecture_doc.py` | every module has a section, every section names a real module, every symbol it mentions resolves |
| `CONTRIBUTING.md` | `tests/test_contributing_doc.py` | every path and CLI subcommand it names exists, and the tooling versions it quotes match the config |

Each of those tests asserts that its regex matched *something* before checking
the match, so rewriting the surrounding prose into a shape the test cannot read
fails loudly instead of quietly disabling the check. If you rephrase a pinned
claim, update the pattern in the same commit.

## Style

Lines wrap at **100** characters. CI pins **ruff 0.16.2** so a new release
adding rules cannot break an unrelated commit; bump it deliberately, alongside
whatever fixes the new rules require.

The lint ruleset is curated rather than `ALL`. The selected families are `E`,
`W`, `F`, `I`, `UP`, `B`, `ARG`, `BLE`, `SLF` — each one catches a class of
defect this codebase actively guards against, so a violation is signal rather
than noise. Tests are exempted from `ARG` and `SLF001`: a stub implements a
protocol's full signature without consulting it, and reaching into internals is
what a unit test is for.

Two conventions the linter cannot check:

- **Docstrings explain why, not what.** The signature already says what a
  function takes. The docstring's job is the decision — what else was tried,
  what breaks under the obvious alternative.
- **Prefer clear over clever.** Every line here should be defensible in a code
  review by the person who wrote it. A query that is 20% shorter and takes ten
  minutes to re-derive is a net loss.

## Commits and publishing

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`. Keep one logical change
per commit — a new question pattern plus its gold row, its tests and its README
paragraph is one change; two unrelated patterns are two.

Publish with the helper, which runs the three checks first and refuses to push
if any of them fail:

```bash
./scripts/push.sh "feat: add revenue-by-region question pattern + gold row"
```
