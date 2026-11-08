import sqlite3
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sqlenv import build_db, execution_match, extract_sql, results_match, run_query  # noqa: E402

SCHEMA = "CREATE TABLE t (id INTEGER, name TEXT, age INTEGER)"
GOLD = "SELECT name FROM t WHERE age > 30"


class ExtractSql(unittest.TestCase):
    def test_plain_block(self):
        self.assertEqual(extract_sql("```sql\nSELECT 1;\n```"), "SELECT 1")

    def test_uppercase_fence(self):
        self.assertEqual(extract_sql("```SQL\nSELECT 1\n```"), "SELECT 1")

    def test_no_fence(self):
        self.assertEqual(extract_sql("SELECT a FROM t;"), "SELECT a FROM t")

    def test_no_fence_after_prose(self):
        self.assertEqual(extract_sql("Here you go: SELECT a FROM t"), "SELECT a FROM t")

    def test_two_blocks_takes_last(self):
        self.assertEqual(extract_sql("```sql\nSELECT 1\n```\nbetter:\n```sql\nSELECT 2\n```"), "SELECT 2")

    def test_multiple_statements_rejected(self):
        self.assertIsNone(extract_sql("```sql\nSELECT 1; SELECT 2\n```"))
        self.assertIsNone(extract_sql("SELECT 1; DROP TABLE t"))

    def test_prose_without_sql(self):
        self.assertIsNone(extract_sql("Sorry, I can't answer that."))

    def test_non_read_statement_rejected(self):
        self.assertIsNone(extract_sql("```sql\nDROP TABLE t\n```"))

    def test_with_query_allowed(self):
        sql = "WITH x AS (SELECT 1) SELECT * FROM x"
        self.assertEqual(extract_sql(f"```sql\n{sql}\n```"), sql)

    def test_semicolon_inside_string_is_one_statement(self):
        sql = "SELECT a FROM t WHERE b = 'x;y'"
        self.assertEqual(extract_sql(f"```sql\n{sql}\n```"), sql)

    def test_semicolon_after_string_is_still_rejected(self):
        self.assertIsNone(extract_sql("SELECT 'a'; DROP TABLE t"))


class ResultsMatch(unittest.TestCase):
    def test_order_matters_only_when_ordered(self):
        self.assertFalse(results_match([(1,), (2,)], [(2,), (1,)], True))
        self.assertTrue(results_match([(1,), (2,)], [(2,), (1,)], False))

    def test_number_forms_are_equal(self):
        for a, b in [(1, 1.0), (1, "1"), (1.0, "1"), (2.5, "2.5")]:
            self.assertTrue(results_match([(a,)], [(b,)], False), (a, b))

    def test_float_rounding(self):
        self.assertTrue(results_match([(0.12341,)], [(0.12342,)], False))
        self.assertFalse(results_match([(0.1234,)], [(0.1235,)], False))

    def test_nulls(self):
        self.assertTrue(results_match([(None,)], [(None,)], False))
        self.assertFalse(results_match([(None,)], [(0,)], False))

    def test_duplicates_count(self):
        self.assertFalse(results_match([(1,), (1,), (2,)], [(1,), (2,), (2,)], False))
        self.assertFalse(results_match([(1,), (1,)], [(1,)], False))

    def test_strings_are_trimmed_not_casefolded(self):
        self.assertTrue(results_match([("Ann ",)], [("Ann",)], False))
        self.assertFalse(results_match([("Ann",)], [("ann",)], False))

    def test_infinity_does_not_crash(self):
        self.assertTrue(results_match([(float("inf"),)], [(float("inf"),)], False))
        self.assertFalse(results_match([(float("inf"),)], [(1,)], False))

    def test_text_that_parses_as_infinity_does_not_crash(self):
        self.assertTrue(results_match([("Inf",)], [("Inf",)], False))
        self.assertTrue(results_match([("1e999",)], [("1e999",)], False))


class ExecutionMatch(unittest.TestCase):
    def test_correct_query(self):
        self.assertEqual(execution_match(SCHEMA, GOLD, "SELECT name FROM t WHERE 30 < age"), (True, True))

    def test_wrong_filter(self):
        self.assertEqual(execution_match(SCHEMA, GOLD, "SELECT name FROM t WHERE age < 30"), (True, False))

    def test_syntax_error(self):
        self.assertEqual(execution_match(SCHEMA, GOLD, "SELEC name FRM t"), (False, False))

    def test_overflowing_result_is_graded_not_raised(self):
        self.assertEqual(execution_match(SCHEMA, GOLD, "SELECT 9e999 FROM t"), (True, False))

    def test_writes_rejected(self):
        for sql in ("DROP TABLE t", "INSERT INTO t VALUES (1, 'x', 2)", "ATTACH ':memory:' AS other"):
            self.assertEqual(execution_match(SCHEMA, GOLD, sql), (False, False), sql)

    def test_rejected_write_leaves_data_alone(self):
        conn = build_db(SCHEMA, GOLD, 11)
        before = run_query(conn, "SELECT COUNT(*) FROM t")
        with self.assertRaises(sqlite3.DatabaseError):
            run_query(conn, "DELETE FROM t")
        self.assertEqual(run_query(conn, "SELECT COUNT(*) FROM t"), before)
        conn.close()

    def test_runaway_cross_join_stops(self):
        start = time.monotonic()
        sql = "SELECT COUNT(*) FROM t a, t b, t c, t d, t e, t f"
        self.assertEqual(execution_match(SCHEMA, GOLD, sql), (False, False))
        self.assertLess(time.monotonic() - start, 30)


if __name__ == "__main__":
    unittest.main()
