# text2sql-grpo

A small offline text-to-SQL model. Base: **Qwen2.5-Coder-1.5B-Instruct**, fine-tuned with **GRPO**. The reward runs the model's SQL against SQLite databases and compares the result with the gold query, so nobody has to hand-label anything. The final model is exported to GGUF and runs in Ollama on a laptop (RTX 4060, 8 GB).

## Layout

| Path | What |
|---|---|
| `src/sqlenv.py` | Builds seeded SQLite DBs from a schema, runs queries safely, compares results (stdlib only) |
| `src/rewards.py` | Prompt format + GRPO reward functions (stdlib only) |
| `scripts/prepare_data.py` | Builds `data/train.jsonl` / `data/test.jsonl` from `b-mc2/sql-create-context`, keeping only gradable examples |
| `scripts/train_grpo.py` | Unsloth + TRL GRPO training (4-bit + LoRA) |
| `scripts/evaluate.py` | Execution accuracy, using HF/LoRA weights or an Ollama model |
| `AGENT_PLAN_AHANY.md` | Work plan for the second contributor's agent (no model needed) |

## Reward

| Model output | Reward |
|---|---|
| Result matches gold on all 3 seeded DBs | 1.0 |
| Runs, wrong result | 0.1 |
| Broken / no SQL / not a single statement | 0 |
| Clean ```sql block | +0.1 |
| Over 600 chars | −0.1 |

The fake rows reuse the literals from the gold query, plus near-miss values, so `>` vs `>=` and wrong filters actually change the result.

## Run

```bash
py -3.11 -m venv .venv && .venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu128 && .venv/Scripts/pip install unsloth trl datasets "triton-windows<3.4"
.venv/Scripts/python scripts/prepare_data.py
.venv/Scripts/python scripts/evaluate.py --hf unsloth/Qwen2.5-Coder-1.5B-Instruct --out results/base.json
.venv/Scripts/python scripts/train_grpo.py --max_steps 400
.venv/Scripts/python scripts/evaluate.py --hf outputs/grpo/final_lora --out results/grpo.json
```
