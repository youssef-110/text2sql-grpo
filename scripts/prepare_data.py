"""Build train/test JSONL from b-mc2/sql-create-context, keeping only examples the executor can grade.

Each row: {question, schema, answer, seeds, hard}. `seeds` = DB seeds where gold returns non-empty rows.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import load_dataset  # noqa: E402

from src.sqlenv import informative_seeds  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data"
HARD = re.compile(r"\b(JOIN|GROUP BY|HAVING|ORDER BY|LIMIT|DISTINCT|COUNT|SUM|AVG|MIN|MAX|IN \(|EXCEPT|INTERSECT|UNION|BETWEEN)\b", re.I)


def main(n_train: int = 6000, n_test: int = 500, easy_frac: float = 0.25):
    rng = random.Random(0)
    ds = load_dataset("b-mc2/sql-create-context", split="train")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    hard, easy = [], []
    for i in idx:
        row = ds[i]
        schema, gold = row["context"], row["answer"]
        seeds = informative_seeds(schema, gold)
        if not seeds:
            continue
        ex = {"question": row["question"], "schema": schema, "answer": gold, "seeds": seeds,
              "hard": bool(HARD.search(gold)) or schema.upper().count("CREATE TABLE") > 1}
        (hard if ex["hard"] else easy).append(ex)
        if len(hard) >= n_train + n_test and len(easy) >= int((n_train + n_test) * easy_frac):
            break
    n_easy = int(n_train * easy_frac)
    test = hard[:n_test] + easy[: int(n_test * easy_frac)]
    train = hard[n_test : n_test + n_train - n_easy] + easy[int(n_test * easy_frac) :][:n_easy]
    rng.shuffle(train)
    OUT.mkdir(exist_ok=True)
    for name, rows in (("train", train), ("test", test)):
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(name, len(rows), "hard:", sum(r["hard"] for r in rows))


if __name__ == "__main__":
    main()
