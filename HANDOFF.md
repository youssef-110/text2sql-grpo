# Handoff: branch `ahany/agent-work`

Everything here uses the Python standard library. No model, Ollama install or dataset was used.

## What was built

| Piece | Path |
|---|---|
| Four databases as SQL text, plus their seeded generator | `data/realworld/{football,shop,school,hr}.sql`, `make_data.py` |
| 160 questions, 40 per database (48 easy, 64 medium, 48 hard) | `data/realworld/questions.jsonl` |
| Real-world eval: `--ollama`, `--predictions`, `--oracle`, `--out` | `scripts/eval_realworld.py` |
| Grader tests | `tests/test_sqlenv.py`, `tests/test_rewards.py` |
| Checks on the eval set itself | `tests/test_realworld.py` |
| Demo CLI and its fake-Ollama tests | `app/ask.py`, `tests/test_ask.py` |

## How to run

```bash
python -m unittest discover -s tests -v
python scripts/eval_realworld.py --oracle
python scripts/eval_realworld.py --ollama text2sql --out results/realworld.json
python scripts/eval_realworld.py --predictions saved.jsonl   # lines of {"db", "question", "output"}
python app/ask.py my.sqlite -q "how many orders last month?"
python data/realworld/make_data.py                           # rewrites the .sql files from fixed seeds
```

## Test results

```
Ran 57 tests in 9.338s

OK
```

```
overall   160/160  1.000
football  40/40  1.000
hr        40/40  1.000
school    40/40  1.000
shop      40/40  1.000
easy      48/48  1.000
hard      48/48  1.000
medium    64/64  1.000
valid SQL 1.000
```

The oracle only proves the gold queries and the grading pipeline agree. It says nothing about the model.

## Bugs fixed in `src/sqlenv.py`

Each fix has a test that failed before it.

1. **The grader crashed on infinity.** `_norm` called `int()` on infinite values, so `SELECT 9e999 ...`, or a text cell such as `Inf` or `1e999`, raised `OverflowError`. `execution_match` calls `results_match` outside its `try`, so one such model output would stop a GRPO training run. Infinite floats now stay floats and those strings stay text. **A training run started before this merge still has the bug.**
2. **A semicolon inside a string rejected a valid query.** `SELECT a FROM t WHERE b = 'x;y'` made `extract_sql` return `None`. String literals are now removed before looking for a second statement. `SELECT 'a'; DROP TABLE t` is still rejected.

## Found but left alone

Worth a look when reviewing. None of these needed a change for the plan's tests, and the first would change training behaviour.

- `execution_match` treats a gold query as ordered if `order by` appears anywhere, including inside `OVER (...)` or a subquery. It also misses `ORDER  BY` with two spaces. `eval_realworld.py` has its own `is_ordered` that only looks at the outermost query, because the "top N per group" questions use window functions.
- Broken SQL in a clean fence scores 0.1, the same as working but wrong SQL with no fence. This matches the README's reward table, but it does pay for formatting alone.
- With no fence, `extract_sql` can take prose as SQL ("I can help with that" gives `with that`). It fails when run, so the reward is still 0, and `ask.py` spends its one retry on it.

## Unfinished or limited

- `--ollama` and `ask.py` have not been run against a real Ollama or the trained model, only the fake server in `tests/test_ask.py`.
- The `leagues` (4 rows) and `departments` (8 rows) lookup tables are under 30 rows, because more would not be believable. I read them as not being "main tables". All other tables have 30–200 rows.
- `ask.py` uses `run_query`'s 2M-step cap, so a very large legitimate scan is aborted and goes down the retry path.
- Grading compares columns by position, so the questions say which columns to return and in what order. A model that returns extra columns is marked wrong.
- All people, clubs and leagues are invented.
