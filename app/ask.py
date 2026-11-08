"""Ask a SQLite database questions in plain English, using the fine-tuned model in Ollama. Stdlib only.

  python app/ask.py my.sqlite                       interactive
  python app/ask.py my.sqlite -q "how many orders?"  one question
"""
import argparse
import sqlite3
import sys
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate import ollama_generate  # noqa: E402
from src.rewards import build_messages  # noqa: E402
from src.sqlenv import _read_only_authorizer, extract_sql, run_query  # noqa: E402

MAX_ROWS = 50
QUERY_ERRORS = (sqlite3.Error, sqlite3.Warning, ValueError, OverflowError)


class AskError(Exception):
    """A problem to show the user as one line, without a traceback."""


def open_db(path):
    """(read-only connection, schema text taken from sqlite_master)."""
    p = Path(path)
    if not p.is_file():
        raise AskError(f"No such database file: {path}")
    conn = sqlite3.connect(f"{p.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT sql FROM sqlite_master WHERE type IN ('table', 'view') "
                            "AND sql IS NOT NULL AND name NOT GLOB 'sqlite_*'").fetchall()
    except sqlite3.DatabaseError as e:
        conn.close()
        raise AskError(f"Can't read {path} as a SQLite database: {e}") from e
    if not rows:
        conn.close()
        raise AskError(f"{path} has no tables to ask about.")
    conn.set_authorizer(_read_only_authorizer)  # model output is untrusted, even on a read-only connection
    return conn, "\n\n".join(r[0] for r in rows)


def chat(messages, model, url):
    try:
        return ollama_generate(model, url.rstrip("/"))([messages])[0]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise AskError(f"Ollama doesn't have a model named '{model}'. Check `ollama list`, or use --model.") from e
        raise AskError(f"Ollama returned HTTP {e.code} at {url}.") from e
    except OSError as e:  # URLError and timeouts
        raise AskError(f"Can't talk to Ollama at {url} ({e}). Is it running? Try `ollama serve`.") from e
    except (ValueError, KeyError, TypeError, IndexError) as e:
        raise AskError(f"Unexpected reply from Ollama at {url}: {e}") from e


def format_table(columns, rows):
    shown = [["NULL" if v is None else str(v).replace("\n", " ") for v in r] for r in rows[:MAX_ROWS]]
    widths = [max(map(len, col)) for col in zip(columns, *shown)]

    def line(cells):
        return " | ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()

    out = [line(columns), "-+-".join("-" * w for w in widths), *map(line, shown)]
    if len(rows) > MAX_ROWS:
        out.append(f"(showing the first {MAX_ROWS} of {len(rows)} rows)")
    else:
        out.append(f"({len(rows)} row{'' if len(rows) == 1 else 's'})")
    return "\n".join(out)


def answer(conn, schema, question, model, url):
    messages = build_messages(schema, question)
    for attempt in (1, 2):
        sql = extract_sql(chat(messages, model, url))
        if sql is None:
            raise AskError("The model didn't return a single read-only query.")
        print(f"\nSQL: {sql}\n")
        try:
            # shortcut: run_query's step cap also aborts very large legitimate scans, raise it if real DBs hit that
            rows = run_query(conn, sql)
            columns = [d[0] for d in conn.execute(sql).description]  # run_query drops the column names
        except QUERY_ERRORS as e:
            if attempt == 2:
                raise AskError(f"SQLite rejected the corrected query too: {e}") from e
            print(f"SQLite error: {e}\nAsking the model to fix it...")
            messages[-1]["content"] += f"\n\nThis query:\n{sql}\nfailed with this SQLite error: {e}\nWrite a corrected query."
            continue
        print(format_table(columns, rows))
        return


def interactive(conn, schema, model, url):
    while True:
        try:
            question = input("\nQuestion (empty to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            return
        try:
            answer(conn, schema, question, model, url)
        except AskError as e:
            print(f"Error: {e}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Ask a SQLite database questions in plain English.")
    ap.add_argument("database", help="path to a .sqlite file")
    ap.add_argument("-q", "--question", help="ask one question and exit")
    ap.add_argument("--model", default="text2sql", help="Ollama model name (default: text2sql)")
    ap.add_argument("--url", default="http://localhost:11434", help="Ollama URL (default: http://localhost:11434)")
    a = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")  # a Windows console codepage must not crash on unusual cell text

    try:
        conn, schema = open_db(a.database)
        if a.question:
            answer(conn, schema, a.question, a.model, a.url)
        else:
            interactive(conn, schema, a.model, a.url)
    except AskError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
