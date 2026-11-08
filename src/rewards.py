"""Prompt format and GRPO reward functions. Standard library only."""
from __future__ import annotations

from .sqlenv import execution_match, extract_sql

SYSTEM_PROMPT = (
    "You are a SQLite expert. Given a database schema and a question, write one SQLite query that answers it. "
    "Reply with only the query inside a ```sql code block."
)


def build_messages(schema: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
    ]


def _text(completion) -> str:
    # TRL passes conversational completions as [{"role": "assistant", "content": ...}]
    return completion[0]["content"] if isinstance(completion, list) else completion


def score(schema: str, gold: str, completion_text: str, seeds) -> float:
    """1.0 correct result, 0.1 runs but wrong, 0 broken. Small format bonus for a clean ```sql block."""
    sql = extract_sql(completion_text)
    if sql is None:
        return 0.0
    fmt = 0.1 if "```sql" in completion_text.lower() else 0.0
    executed, correct = execution_match(schema, gold, sql, seeds)
    if correct:
        return 1.0 + fmt
    return (0.1 if executed else 0.0) + fmt


def execution_reward(prompts, completions, schema, answer, seeds, **kwargs) -> list[float]:
    """TRL GRPO reward function. Extra dataset columns (schema, answer, seeds) arrive as kwargs lists."""
    return [score(s, a, _text(c), sd) for c, s, a, sd in zip(completions, schema, answer, seeds)]


def length_penalty(completions, **kwargs) -> list[float]:
    """Discourage rambling: the task wants just a query."""
    return [-0.1 if len(_text(c)) > 600 else 0.0 for c in completions]
