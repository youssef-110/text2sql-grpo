# Plan for Ahany's coding agent

You are helping build **text2sql-grpo**: a small (1.5B) model that turns English questions into SQLite queries. Youssef trains it with GRPO on his laptop GPU. You **do not have the model** and must not get it. Your job is everything around the model that can be built and tested with plain Python.

## Hard rules

1. **No downloads.** Don't run `pip install`, `ollama pull`, `huggingface-cli`, `git lfs` or any dataset download. Don't fetch web pages. The only network use allowed is `git pull` / `git push` to this repo.
2. **Python standard library only** (`sqlite3`, `json`, `unittest`, `urllib`, `http.server`, `argparse`, ...). Python 3.10+.
3. **Don't edit** `scripts/train_grpo.py`, `scripts/prepare_data.py` or anything under `data/` except `data/realworld/`. Youssef is running those.
4. You may edit `src/sqlenv.py` / `src/rewards.py` only to fix a real bug you found with a test. Explain the fix in `HANDOFF.md`.
5. Work on branch `ahany/agent-work`. Commit small, clear commits. Push that branch when done (step 6). Don't push to `main`.
6. Never commit `.sqlite`/`.db` binaries. Databases are stored as `.sql` text scripts.

Read first: `README.md`, `src/sqlenv.py` (executor + result comparison), `src/rewards.py` (prompt + reward), `scripts/evaluate.py` (how accuracy is measured).

## Task 1: Real-world eval set (most important)

The training data uses random fake rows. We need a **realistic, harder test** to prove the model is actually good.

- Create 4 databases as SQL scripts in `data/realworld/`: `football.sql` (leagues, clubs, players, matches, goals), `shop.sql` (customers, products, orders, order_items), `school.sql` (students, courses, enrollments, grades, teachers), `hr.sql` (employees, departments, salaries with dates, managers via self-join).
  - Each: `CREATE TABLE` with sensible types, primary/foreign keys, and **30–200 rows of believable data** per main table (write a small stdlib generator `data/realworld/make_data.py` with a fixed `random.seed`, and commit its output `.sql`).
- Write `data/realworld/questions.jsonl`, **at least 160 lines** (40 per DB), each `{"db": "football", "question": "...", "answer": "<gold SQLite query>", "level": "easy|medium|hard"}`.
  - Mix: ~30% easy (single table filter/count), ~40% medium (JOIN, GROUP BY, ORDER BY + LIMIT), ~30% hard (multi-join, HAVING, subqueries, self-join, dates with `strftime`, "top N per group", percentages).
  - Phrase questions like a real person would, not like SQL ("Which club scored the most away goals in 2024?").
  - Every gold query must run and return **at least one non-NULL row**. Avoid ambiguous questions (state tie-breaks, say "list names" vs "how many").
- Write `scripts/eval_realworld.py` (stdlib only):
  - Builds each DB in memory from its `.sql`, runs gold and predicted queries, compares with `src.sqlenv.results_match` (ordered only if gold has `ORDER BY`), runs read-only (`PRAGMA query_only=1`) and uses `src.sqlenv.run_query`.
  - Prompt the model with `src.rewards.build_messages(schema_text, question)`, where `schema_text` is the DB's `CREATE TABLE` statements only (no INSERTs).
  - Backends: `--ollama <name>` (reuse `ollama_generate` from `scripts/evaluate.py`), `--predictions file.jsonl` (score saved outputs) and `--oracle` (answers with the gold query; must give 100%, this is your self-test).
  - Prints accuracy overall, per DB and per level, and writes a JSON report with `--out`.
- Run `python scripts/eval_realworld.py --oracle` and make sure it prints 1.000.

## Task 2: Tests for the grader

`tests/test_sqlenv.py` and `tests/test_rewards.py` with `unittest`. Cover at least:
- `extract_sql`: plain ```sql block, ```SQL uppercase, no fence, two blocks (takes last), multiple statements → `None`, prose with no SQL → `None`.
- `results_match`: order matters only when ordered; `1` vs `1.0` vs `"1"` equal; float rounding; NULLs; duplicate rows count.
- `execution_match`: correct query, wrong filter, syntax error, `DROP TABLE`/`INSERT` rejected (DB is read-only), runaway cross join stops instead of hanging.
- `score`: values 1.1 / 1.0 / 0.2 / 0.1 / 0.0 for the right cases. `execution_reward` with TRL-style inputs (`completions=[[{"role":"assistant","content":...}]]`).

Run: `python -m unittest discover -s tests -v`. All must pass. If a test reveals a real bug, fix it (rule 4).

## Task 3: The demo app `app/ask.py`

A CLI people use with the finished model:

```
python app/ask.py path/to/my.sqlite            # interactive
python app/ask.py my.sqlite -q "how many orders last month?"
```

- Reads the schema from the `.sqlite` file (`sqlite_master`), builds the prompt with `build_messages`, calls Ollama `POST /api/chat` (model name from `--model`, default `text2sql`, URL from `--url`, default `http://localhost:11434`), extracts SQL with `extract_sql`.
- Shows the SQL, runs it **read-only** (open with `file:...?mode=ro` URI), prints a neat text table (max 50 rows, stdlib only).
- If SQLite raises an error, send one retry with the error message appended to the user turn, then give up cleanly.
- Friendly errors when Ollama isn't running or the model is missing.
- `tests/test_ask.py`: start a fake Ollama with `http.server` in a thread that returns a canned ```sql answer, then check the app prints the right rows and the retry path works. **No real model needed.**

## Task 4: Docs

- Add to `README.md`: a "Use the model" section for `app/ask.py`, and a "Real-world eval" section with the command.
- Write `HANDOFF.md`: what you built, how to run each piece, test results (paste the summary output), anything unfinished, and any bugs found in `src/`.

## Step 6: Finish

```
python -m unittest discover -s tests -v
python scripts/eval_realworld.py --oracle
git add -A && git commit -m "Real-world eval set, grader tests, ask.py demo"
git push -u origin ahany/agent-work
```

Then stop. Youssef's agent will review the branch, merge it, and run the real-world eval against the trained model.
