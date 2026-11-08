"""Execution environment for text-to-SQL: build SQLite DBs from a schema, run queries, compare results.

Standard library only, so it runs anywhere (no model, no pip installs needed).
"""
from __future__ import annotations

import random
import re
import sqlite3
import string
from collections import Counter

ROWS_PER_TABLE = 25
QUERY_STEP_LIMIT = 2_000_000  # VM steps before a query is aborted (stops runaway cross joins)

_STR_LIT = re.compile(r"'((?:[^']|'')*)'|\"((?:[^\"]|\"\")*)\"")
_NUM_LIT = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
_SQL_BLOCK = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.S | re.I)


def extract_literals(sql: str) -> tuple[list[str], list[float]]:
    """String and number literals in a query; used to seed fake data so filters can match."""
    strings = [(a or b).replace("''", "'").replace('""', '"') for a, b in _STR_LIT.findall(sql)]
    without_strings = _STR_LIT.sub(" ", sql)
    nums = [float(n) if "." in n else int(n) for n in _NUM_LIT.findall(without_strings)]
    return strings, nums


def _random_value(rng: random.Random, col_type: str, strings: list[str], nums: list) -> object:
    t = col_type.upper()
    numeric = any(k in t for k in ("INT", "REAL", "NUM", "FLOAT", "DOUBLE", "DEC"))
    roll = rng.random()
    if roll < 0.08:
        return None
    if roll < 0.55:  # reuse a literal from the gold query so WHERE clauses hit rows
        pool = (nums or strings) if numeric else (strings + [str(n) for n in nums])
        if pool:
            v = rng.choice(pool)
            if isinstance(v, (int, float)) and rng.random() < 0.5:
                v = v + rng.choice([-1, 1]) * rng.randint(1, 3)  # near-miss values for <, > filters
            return v
    if numeric:
        return rng.randint(0, 100) if "INT" in t else round(rng.uniform(0, 100), 2)
    if strings and rng.random() < 0.3:
        return rng.choice(strings)
    return "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8)))


def build_db(schema_sql: str, gold_sql: str, seed: int, rows: int = ROWS_PER_TABLE) -> sqlite3.Connection:
    """In-memory DB with `schema_sql` tables filled with seeded random rows biased toward gold literals."""
    rng = random.Random(seed)
    strings, nums = extract_literals(gold_sql)
    conn = sqlite3.connect(":memory:")
    conn.executescript(schema_sql)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    # Shared small key pool so joins between tables actually match.
    keys = [rng.randint(1, 12) for _ in range(rows)]
    for table in tables:
        cols = conn.execute(f'PRAGMA table_info("{table}")').fetchall()  # cid, name, type, notnull, dflt, pk
        for i in range(rows):
            values = []
            for _, name, ctype, *_ in cols:
                lname = name.lower()
                if lname == "id" or lname.endswith("_id"):
                    values.append(keys[(i + rng.randint(0, 3)) % rows])
                else:
                    values.append(_random_value(rng, ctype, strings, nums))
            placeholders = ",".join("?" * len(values))
            conn.execute(f'INSERT INTO "{table}" VALUES ({placeholders})', values)
    conn.commit()
    conn.execute("PRAGMA query_only = 1")
    conn.set_authorizer(_read_only_authorizer)  # model output is untrusted: no ATTACH, PRAGMA or writes
    return conn


_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, getattr(sqlite3, "SQLITE_RECURSIVE", 33)}


def _read_only_authorizer(action, arg1, arg2, db_name, trigger):
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def run_query(conn: sqlite3.Connection, sql: str):
    """Rows for a single read-only statement, or raise."""
    steps = 0

    def guard():
        nonlocal steps
        steps += 1
        return 1 if steps * 1000 > QUERY_STEP_LIMIT else 0

    conn.set_progress_handler(guard, 1000)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.set_progress_handler(None, 0)


def _norm(v):
    if isinstance(v, float):
        return int(v) if v.is_integer() else round(v, 4)  # is_integer: int(inf) would raise
    if isinstance(v, str):
        try:
            f = float(v)
            return int(f) if f == int(f) else round(f, 4)
        except (ValueError, OverflowError):  # "abc", "NaN", "Inf", "1e999" stay text
            return v.strip()
    return v


def results_match(gold_rows, pred_rows, ordered: bool) -> bool:
    g = [tuple(_norm(v) for v in r) for r in gold_rows]
    p = [tuple(_norm(v) for v in r) for r in pred_rows]
    return g == p if ordered else Counter(g) == Counter(p)


def extract_sql(text: str) -> str | None:
    """Last ```sql block, else a bare SELECT/WITH statement. None if nothing usable."""
    blocks = _SQL_BLOCK.findall(text)
    sql = blocks[-1] if blocks else None
    if sql is None:
        m = re.search(r"((?:SELECT|WITH)\b.*)", text, re.S | re.I)
        sql = m.group(1) if m else None
    if not sql:
        return None
    sql = sql.strip().rstrip(";").strip()
    if ";" in _STR_LIT.sub(" ", sql) or not re.match(r"(SELECT|WITH)\b", sql, re.I):  # one read statement only
        return None
    return sql or None


SEEDS = (11, 22, 33)


def informative_seeds(schema_sql: str, gold_sql: str) -> list[int]:
    """Seeds whose DB gives the gold query a non-empty result. Empty list -> example is useless for reward."""
    out = []
    for s in SEEDS:
        try:
            conn = build_db(schema_sql, gold_sql, s)
            rows = run_query(conn, gold_sql)
            if rows and any(v is not None for r in rows for v in r):
                out.append(s)
        except sqlite3.Error:
            return []
    return out


def execution_match(schema_sql: str, gold_sql: str, pred_sql: str, seeds=SEEDS) -> tuple[bool, bool]:
    """(pred executed without error, pred results equal gold on every seed DB)."""
    ordered = "order by" in gold_sql.lower()
    executed = True
    for s in seeds:
        conn = build_db(schema_sql, gold_sql, s)
        gold = run_query(conn, gold_sql)
        try:
            pred = run_query(conn, pred_sql)
        except (sqlite3.Error, sqlite3.Warning, ValueError, OverflowError):
            return False, False
        if not results_match(gold, pred, ordered):
            return executed, False
    return executed, True
