"""The eval set itself has to meet the plan: size, mix, and gold queries that actually answer the question."""
import json
import re
import sqlite3
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_realworld import DATA, is_ordered, load_db, load_questions  # noqa: E402
from src.sqlenv import extract_sql, results_match, run_query  # noqa: E402


class EvalSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions = load_questions()
        cls.dbs = {name: load_db(name) for name in ("football", "shop", "school", "hr")}

    @classmethod
    def tearDownClass(cls):
        for conn, _ in cls.dbs.values():
            conn.close()

    def test_size_and_level_mix(self):
        self.assertGreaterEqual(len(self.questions), 160)
        for name, n in Counter(q["db"] for q in self.questions).items():
            self.assertGreaterEqual(n, 40, name)
        share = {k: v / len(self.questions) for k, v in Counter(q["level"] for q in self.questions).items()}
        self.assertAlmostEqual(share["easy"], 0.3, delta=0.05)
        self.assertAlmostEqual(share["medium"], 0.4, delta=0.05)
        self.assertAlmostEqual(share["hard"], 0.3, delta=0.05)

    def test_questions_are_unique(self):
        pairs = Counter((q["db"], q["question"]) for q in self.questions)
        self.assertEqual([k for k, v in pairs.items() if v > 1], [])

    def test_gold_runs_and_returns_a_value(self):
        for q in self.questions:
            with self.subTest(q["question"]):
                self.assertEqual(extract_sql(q["answer"]), q["answer"])
                rows = run_query(self.dbs[q["db"]][0], q["answer"])
                self.assertTrue(any(v is not None for r in rows for v in r), "empty or all-NULL result")

    def test_limit_queries_have_a_tie_break(self):
        for q in self.questions:
            order = re.search(r"ORDER BY ([^()]*(?:\([^()]*\)[^()]*)*?) LIMIT \d+$", q["answer"])
            if order:
                with self.subTest(q["question"]):
                    self.assertIn(",", order.group(1))

    def test_gold_does_not_depend_on_scan_order(self):
        # Rebuilt because the shared connections deny PRAGMA; reversing scans exposes unstable results.
        for name in self.dbs:
            conn = sqlite3.connect(":memory:")
            conn.executescript((DATA / f"{name}.sql").read_text(encoding="utf-8"))
            for q in (q for q in self.questions if q["db"] == name):
                with self.subTest(q["question"]):
                    conn.execute("PRAGMA reverse_unordered_selects = 0")
                    forward = conn.execute(q["answer"]).fetchall()
                    conn.execute("PRAGMA reverse_unordered_selects = 1")
                    backward = conn.execute(q["answer"]).fetchall()
                    self.assertTrue(results_match(forward, backward, is_ordered(q["answer"])))
            conn.close()

    def test_is_ordered_only_for_outer_order_by(self):
        self.assertTrue(is_ordered("SELECT a FROM t ORDER BY a"))
        self.assertTrue(is_ordered("SELECT a FROM t order\n by a"))
        self.assertFalse(is_ordered("SELECT a FROM t"))
        self.assertFalse(is_ordered("SELECT a FROM (SELECT a, ROW_NUMBER() OVER (ORDER BY a) r FROM t) WHERE r = 1"))
        self.assertFalse(is_ordered("SELECT a FROM t WHERE b = 'order by'"))

    def test_jsonl_lines_have_the_four_keys(self):
        for line in (DATA / "questions.jsonl").read_text(encoding="utf-8").splitlines():
            self.assertEqual(set(json.loads(line)), {"db", "question", "answer", "level"})


if __name__ == "__main__":
    unittest.main()
