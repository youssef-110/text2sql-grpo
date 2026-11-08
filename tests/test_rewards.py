import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rewards import build_messages, execution_reward, length_penalty, score  # noqa: E402

SCHEMA = "CREATE TABLE t (id INTEGER, name TEXT, age INTEGER)"
GOLD = "SELECT name FROM t WHERE age > 30"
WRONG = "SELECT name FROM t WHERE age < 30"
SEEDS = (11, 22, 33)


def fenced(sql):
    return f"```sql\n{sql}\n```"


class Score(unittest.TestCase):
    def check(self, text, expected):
        self.assertAlmostEqual(score(SCHEMA, GOLD, text, SEEDS), expected)

    def test_correct_in_fence(self):
        self.check(fenced(GOLD), 1.1)

    def test_correct_without_fence(self):
        self.check(GOLD, 1.0)

    def test_wrong_in_fence(self):
        self.check(fenced(WRONG), 0.2)

    def test_wrong_without_fence(self):
        self.check(WRONG, 0.1)

    def test_broken_in_fence_only_gets_format_bonus(self):
        self.check(fenced("SELECT nope FROM missing"), 0.1)

    def test_broken_without_fence(self):
        self.check("SELECT nope FROM missing", 0.0)

    def test_no_sql(self):
        self.check("I don't know.", 0.0)

    def test_multiple_statements(self):
        self.check(fenced(f"{GOLD}; DROP TABLE t"), 0.0)


class Trl(unittest.TestCase):
    def test_execution_reward_with_conversational_completions(self):
        completions = [[{"role": "assistant", "content": fenced(GOLD)}],
                       [{"role": "assistant", "content": fenced(WRONG)}],
                       [{"role": "assistant", "content": "no idea"}]]
        rewards = execution_reward(prompts=[[]] * 3, completions=completions, schema=[SCHEMA] * 3, answer=[GOLD] * 3,
                                   seeds=[list(SEEDS)] * 3, question=["ignored"] * 3)
        for got, want in zip(rewards, [1.1, 0.2, 0.0]):
            self.assertAlmostEqual(got, want)

    def test_execution_reward_with_plain_text(self):
        rewards = execution_reward([""], [GOLD], [SCHEMA], [GOLD], [list(SEEDS)])
        self.assertAlmostEqual(rewards[0], 1.0)

    def test_length_penalty(self):
        short = [{"role": "assistant", "content": "x" * 600}]
        long = [{"role": "assistant", "content": "x" * 601}]
        self.assertEqual(length_penalty([short, long]), [0.0, -0.1])


class Prompt(unittest.TestCase):
    def test_build_messages_carries_schema_and_question(self):
        msgs = build_messages(SCHEMA, "How old?")
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        self.assertIn(SCHEMA, msgs[1]["content"])
        self.assertIn("How old?", msgs[1]["content"])


if __name__ == "__main__":
    unittest.main()
