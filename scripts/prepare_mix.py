"""Build data/train_mix.jsonl: training rows whose schema is a whole database, not just the tables the answer needs.

Sources:
  SuperMax991/spider-text2sql   Spider train, full schema of each of its 140 databases (CC BY-SA 4.0)
  gretelai/synthetic_text_to_sql  multi-table SELECT examples, for domain variety (Apache 2.0)

Every row is kept only if its gold query runs in SQLite on its own schema. Rows are {question, schema, answer}.
"""
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sqlenv import build_db, extract_sql, run_query  # noqa: E402

TYPES = {"number": "INTEGER", "text": "TEXT", "time": "TEXT", "boolean": "INTEGER"}
GRETEL_ROWS = 3000


def ident(name):
    return name if re.fullmatch(r"\w+", name) else f'"{name}"'


def spider_schema(compact):
    """'t: a (number), b (text) | u: ...' -> CREATE TABLE statements."""
    out = []
    for table in compact.split(" | "):
        name, cols = table.split(": ", 1)
        defs = []
        for col in cols.split(", "):
            m = re.fullmatch(r"(.+) \((\w+)\)", col)
            defs.append(f"{ident(m.group(1))} {TYPES.get(m.group(2), 'TEXT')}")
        out.append(f"CREATE TABLE {ident(name)} ({', '.join(defs)})")
    return "\n\n".join(out)


def spider():
    rows, dropped = [], 0
    for r in load_dataset("SuperMax991/spider-text2sql", split="train"):
        gold = extract_sql(re.sub(r"\s+", " ", r["query"]).strip())
        try:
            schema = spider_schema(r["db_schema"])
            run_query(build_db(schema.replace("\n\n", ";\n"), gold, 11), gold)  # executescript needs the ';'
        except Exception:
            dropped += 1
            continue
        rows.append({"question": r["question"].strip(), "schema": schema, "answer": gold})
    print(f"spider: kept {len(rows)}, dropped {dropped}")
    return rows


def gretel(rng):
    ds = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    rows = []
    for i in idx:
        r = ds[i]
        gold = extract_sql(r["sql"])
        if gold is None or r["sql_context"].upper().count("CREATE TABLE") < 2:
            continue
        try:
            conn = sqlite3.connect(":memory:")
            conn.executescript(r["sql_context"])
            if not run_query(conn, gold):  # must return rows on its own sample data: proves it is valid SQLite
                continue
            schema = "\n\n".join(t[0] for t in conn.execute("SELECT sql FROM sqlite_master WHERE type='table'"))
        except Exception:
            continue
        rows.append({"question": r["sql_prompt"].strip(), "schema": schema, "answer": gold})
        if len(rows) == GRETEL_ROWS:
            break
    print(f"gretel: kept {len(rows)}")
    return rows


if __name__ == "__main__":
    rng = random.Random(0)
    rows = spider() + gretel(rng)
    rng.shuffle(rows)
    with open(ROOT / "data" / "train_mix.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(rows), "rows")
