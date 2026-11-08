"""Execution accuracy on the hand-written real-world question set in data/realworld/. Stdlib only.

Backends:
  --ollama <model name>      a local Ollama model
  --predictions file.jsonl   saved outputs, one {"db", "question", "output"} per line
  --oracle                   answers with the gold query (self-test, must print 1.000)
"""
import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate import ollama_generate  # noqa: E402
from src.rewards import build_messages  # noqa: E402
from src.sqlenv import _read_only_authorizer, extract_sql, results_match, run_query  # noqa: E402

DATA = ROOT / "data" / "realworld"
QUERY_ERRORS = (sqlite3.Error, sqlite3.Warning, ValueError, OverflowError)


def load_db(name):
    """(read-only in-memory connection, CREATE TABLE statements without any INSERTs)."""
    conn = sqlite3.connect(":memory:")
    conn.executescript((DATA / f"{name}.sql").read_text(encoding="utf-8"))
    tables = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name NOT GLOB 'sqlite_*'")
    schema = "\n\n".join(r[0] for r in tables)
    conn.execute("PRAGMA query_only = 1")
    conn.set_authorizer(_read_only_authorizer)
    return conn, schema


def load_questions():
    with open(DATA / "questions.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_ordered(sql):
    """True only for an ORDER BY on the outermost query; one inside OVER (...) or a subquery doesn't order the rows."""
    flat, prev = re.sub(r"'(?:[^']|'')*'", "''", sql), None
    while flat != prev:
        flat, prev = re.sub(r"\([^()]*\)", " ", flat), flat
    return re.search(r"\border\s+by\b", flat, re.I) is not None


def grade(conn, gold_sql, output):
    """(correct, ran without error, why it failed)."""
    sql = extract_sql(output) if output else None
    if sql is None:
        return False, False, "no usable SQL"
    gold = run_query(conn, gold_sql)
    try:
        pred = run_query(conn, sql)
    except QUERY_ERRORS as e:
        return False, False, str(e)
    return results_match(gold, pred, is_ordered(gold_sql)), True, ""


def accuracy(rows):
    return {"n": len(rows), "correct": sum(r["correct"] for r in rows),
            "accuracy": sum(r["correct"] for r in rows) / max(1, len(rows))}


def summarize(results):
    groups = {"db": defaultdict(list), "level": defaultdict(list)}
    for r in results:
        for key, group in groups.items():
            group[r[key]].append(r)
    return {"overall": accuracy(results),
            "valid_sql_rate": sum(r["runs"] for r in results) / max(1, len(results)),
            "by_db": {k: accuracy(v) for k, v in groups["db"].items()},
            "by_level": {k: accuracy(v) for k, v in groups["level"].items()}}


def load_predictions(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {(r["db"], r["question"]): r["output"] for r in rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ollama", metavar="MODEL")
    g.add_argument("--predictions", metavar="FILE")
    g.add_argument("--oracle", action="store_true")
    ap.add_argument("--ollama_url", default="http://localhost:11434")
    ap.add_argument("--out", default="", help="write a JSON report here")
    a = ap.parse_args()

    questions = load_questions()
    dbs = {name: load_db(name) for name in sorted({q["db"] for q in questions})}
    saved = load_predictions(a.predictions) if a.predictions else {}
    gen = ollama_generate(a.ollama, a.ollama_url) if a.ollama else None

    results = []
    for i, q in enumerate(questions, 1):
        conn, schema = dbs[q["db"]]
        if a.oracle:
            output = f"```sql\n{q['answer']}\n```"
        elif gen:
            output = gen([build_messages(schema, q["question"])])[0]
            print(f"{i}/{len(questions)}", file=sys.stderr, flush=True)
        else:
            output = saved.get((q["db"], q["question"]))  # a missing prediction scores as wrong
        correct, runs, error = grade(conn, q["answer"], output)
        results.append({**q, "output": output, "correct": correct, "runs": runs, "error": error})

    summary = summarize(results)
    o = summary["overall"]
    print(f"overall   {o['correct']}/{o['n']}  {o['accuracy']:.3f}")
    for section in ("by_db", "by_level"):
        for name, s in sorted(summary[section].items()):
            print(f"{name:<9} {s['correct']}/{s['n']}  {s['accuracy']:.3f}")
    print(f"valid SQL {summary['valid_sql_rate']:.3f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, indent=1)
    return 1 if a.oracle and o["correct"] != o["n"] else 0


if __name__ == "__main__":
    sys.exit(main())
