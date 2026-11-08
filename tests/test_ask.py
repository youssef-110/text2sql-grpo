"""Drives the real CLI against a fake Ollama on localhost. No model needed."""
import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASK = ROOT / "app" / "ask.py"
ENV = {**os.environ, "NO_PROXY": "*", "PYTHONIOENCODING": "utf-8"}


def fenced(sql):
    return f"```sql\n{sql}\n```"


class FakeOllama(ThreadingHTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), Handler)
        self.replies, self.requests = [], []


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        if body["model"] == "missing":
            status, payload = 404, {"error": "model 'missing' not found"}
        else:
            status, payload = 200, {"message": {"role": "assistant", "content": self.server.replies.pop(0)}}
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class Ask(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = str(Path(self.tmp.name) / "my db.sqlite")  # a space, so the file: URI must be built properly
        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE pets (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        conn.executemany("INSERT INTO pets (name, age) VALUES (?, ?)", [("Rex", 3), ("Mia", None)])
        conn.commit()
        conn.close()
        self.server = FakeOllama()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def run_ask(self, *args, url=None, model="text2sql", db=None, stdin=None):
        return subprocess.run([sys.executable, str(ASK), db or self.db, "--url", url or self.url, "--model", model, *args],
                              capture_output=True, text=True, encoding="utf-8", env=ENV, input=stdin, timeout=60)

    def count_rows(self):
        conn = sqlite3.connect(self.db)
        n = conn.execute("SELECT COUNT(*) FROM pets").fetchone()[0]
        conn.close()
        return n

    def test_prints_sql_and_rows(self):
        self.server.replies = [fenced("SELECT name, age FROM pets ORDER BY id")]
        r = self.run_ask("-q", "list pets")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("SQL: SELECT name, age FROM pets ORDER BY id", r.stdout)
        lines = r.stdout.splitlines()
        self.assertIn("name | age", lines)
        self.assertIn("Rex  | 3", lines)
        self.assertIn("Mia  | NULL", lines)
        self.assertIn("(2 rows)", lines)

    def test_prompt_uses_schema_from_the_file(self):
        self.server.replies = [fenced("SELECT 1")]
        self.run_ask("-q", "how many pets?")
        request = self.server.requests[0]
        self.assertEqual(request["model"], "text2sql")
        self.assertIn("CREATE TABLE pets", request["messages"][1]["content"])
        self.assertIn("how many pets?", request["messages"][1]["content"])

    def test_retry_sends_the_error_and_succeeds(self):
        self.server.replies = [fenced("SELECT nope FROM pets"), fenced("SELECT name FROM pets WHERE id = 1")]
        r = self.run_ask("-q", "first pet?")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(self.server.requests), 2)
        self.assertIn("no such column: nope", self.server.requests[1]["messages"][1]["content"])
        self.assertIn("Rex", r.stdout)

    def test_gives_up_cleanly_after_one_retry(self):
        self.server.replies = [fenced("SELECT nope FROM pets"), fenced("SELECT nada FROM pets")]
        r = self.run_ask("-q", "first pet?")
        self.assertEqual(r.returncode, 1)
        self.assertEqual(len(self.server.requests), 2)
        self.assertIn("no such column: nada", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_write_through_cte_is_blocked(self):
        delete = fenced("WITH x AS (SELECT 1) DELETE FROM pets")
        self.server.replies = [delete, delete]
        r = self.run_ask("-q", "delete everything")
        self.assertEqual(r.returncode, 1)
        self.assertNotIn("Traceback", r.stderr)
        self.assertEqual(self.count_rows(), 2)

    def test_rows_are_capped_at_50(self):
        self.server.replies = [fenced("WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 60) SELECT i FROM n")]
        r = self.run_ask("-q", "count to 60")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("(showing the first 50 of 60 rows)", r.stdout)
        self.assertIn("50", r.stdout.splitlines())
        self.assertNotIn("51", r.stdout.splitlines())

    def test_no_sql_reply(self):
        self.server.replies = ["Sorry, I can't help."]
        r = self.run_ask("-q", "?")
        self.assertEqual(r.returncode, 1)
        self.assertIn("didn't return", r.stderr)

    def test_ollama_not_running(self):
        with socket.socket() as s:  # a port nobody is listening on once this closes
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        r = self.run_ask("-q", "?", url=f"http://127.0.0.1:{port}")
        self.assertEqual(r.returncode, 1)
        self.assertIn("Is it running?", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_model_missing(self):
        r = self.run_ask("-q", "?", model="missing")
        self.assertEqual(r.returncode, 1)
        self.assertIn("doesn't have a model named 'missing'", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_missing_database_file(self):
        r = self.run_ask("-q", "?", db=str(Path(self.tmp.name) / "nope.sqlite"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("No such database file", r.stderr)

    def test_file_that_is_not_a_database(self):
        junk = Path(self.tmp.name) / "junk.sqlite"
        junk.write_text("definitely not sqlite " * 100)
        r = self.run_ask("-q", "?", db=str(junk))
        self.assertEqual(r.returncode, 1)
        self.assertIn("Can't read", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_interactive_answers_then_quits(self):
        self.server.replies = [fenced("SELECT COUNT(*) AS n FROM pets")]
        r = self.run_ask(stdin="how many?\n\n")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("(1 row)", r.stdout)


if __name__ == "__main__":
    unittest.main()
