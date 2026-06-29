# Setup & one-command publishing

This repo is built incrementally and reviewed before each push. Day-to-day
effort is a single command, and a change can never be published unless it passes
its own tests and evaluation.

## One-time setup

1. Create a project virtual environment and install the dev dependency
   (`pytest`). This keeps things off your system Python:

   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   ```

   (`.venv/` is gitignored, so it is never committed. The optional `openai`
   package in `requirements.txt` is only needed for the `--llm` backend.)

2. Make the publish helper executable:

   ```bash
   chmod +x scripts/push.sh
   ```

That's it — you do not need to "activate" the venv. `scripts/push.sh`
automatically uses `.venv/bin/python` when it exists.

## Publishing an increment: one command

```bash
./scripts/push.sh "feat: short description of the change"
```

The helper:

1. builds the sample database (`scripts/build_sample_db.py`),
2. runs the test suite (`pytest`),
3. runs the evaluation harness (`evals/evaluate.py`),

and **only commits and pushes if all three succeed**. Omit the message and it
uses a safe default. To force a specific interpreter, run
`PYTHON=/path/to/python ./scripts/push.sh "..."`.

## Why publishing is not fully automatic

Pushing requires your GitHub credentials and must run on your machine; the
assistant that drafts each increment cannot trigger it. Keeping a human review
step also protects the portfolio — every public commit is one you have seen and
can explain in an interview.
