# text2sql-grpo

A small offline text-to-SQL model. Base: **Qwen2.5-Coder-1.5B-Instruct**, fine-tuned with **GRPO**. The reward runs the model's SQL against SQLite databases and compares the result with the gold query, so nobody has to hand-label anything. The final model is exported to GGUF and runs in Ollama on a laptop (RTX 4060, 8 GB).

**Published model:** [Hex990/text2sql-qwen2.5-7b](https://huggingface.co/Hex990/text2sql-qwen2.5-7b) on Hugging Face. It is a supervised LoRA fine-tune of **Qwen2.5-7B-Instruct**, trained on Nebius Token Factory, and scores **83.4%** execution accuracy on the test set. **It should not be used on real databases:** on 160 real-world questions it scores 16.3%, against 81.9% for the untuned `qwen2.5:7b` (see Results). It is not the GRPO model: the GRPO run on the 1.5B is paused at step 200 of 750 (`outputs/grpo/checkpoint-200`, not in git) and has not been evaluated.

## Results

### Real-world questions (the number that matters)

160 hand-written questions over four multi-table SQLite databases (shop, HR, school, football). Scored with `scripts/eval_realworld.py` and `data/realworld/` from branch `ahany/agent-work`, which is not merged into `main` yet.

| | Untuned Qwen2.5-7B-Instruct | Run 1, epoch 3 (published) | Run 2, epoch 1 | Run 2, epoch 2 |
|---|---|---|---|---|
| All 160 | **81.9%** | 16.3% | 65.6% | 62.5% |
| Easy (48) | 93.8% | 25.0% | 81.2% | 83.3% |
| Medium (64) | 87.5% | 14.1% | 67.2% | 64.1% |
| Hard (48) | 62.5% | 10.4% | 47.9% | 39.6% |
| Query runs without error | 96.3% | 54.4% | 93.1% | 91.2% |
| Results file | `realworld-qwen2.5-7b.json` | `realworld-text2sql.json` | `realworld-v2-e1.json` | `realworld-v2-e2.json` |

**Neither fine-tune beats the untuned base. For real use, run the untuned `qwen2.5:7b` with the system prompt from `src/rewards.py`.**

- Run 1 (`data/train.jsonl`, from `b-mc2/sql-create-context`): the training schemas are tiny and mostly single-table, with nearly every column used. The model learned to join tables the question never mentions and to invent columns.
- Run 2 (`scripts/prepare_mix.py`: 6,598 Spider rows with the full schema of each database plus 3,000 multi-table Gretel rows; lower learning rate, rank 16, 2 epochs): fixes most of that, but the model gets sloppier than the base on details. It drops tie-break ordering and rounding the question asks for and writes `= 'null'` where `IS NULL` is needed. More training made it worse (epoch 2 below epoch 1).
- The public training answers are terser and less careful than what the base model already writes, so supervised fine-tuning on them pulls it down. Any future run should be judged on this set, with 81.9% as the bar.

### In-distribution test set

Execution accuracy on the 625 questions in `data/test.jsonl`, which come from the same dataset as the training data. A query is correct only if it returns the same rows as the gold query on all 3 seeded databases.

| Model | All 625 | Hard (500) | Valid SQL | Results file |
|---|---|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct, untuned (4-bit) | 57.6% | 54.0% | 100% | `results/base.json` |
| Qwen2.5-7B-Instruct, untuned (Q4_K_M, Ollama) | 68.0% | 61.6% | 100% | `results/base-7b.json` |
| Fine-tuned 7B, epoch 2 (Q4_K_M, Ollama) | 82.6% | 80.8% | 100% | `results/sft-7b-e2.json` |
| **Fine-tuned 7B, epoch 3 (Q4_K_M, Ollama), published** | **83.4%** | **81.6%** | 100% | `results/sft-7b-e3.json` |

Epoch 3 beats epoch 2 by five questions, which is within noise, and it was picked using the test set, so 83.4% is slightly optimistic. Known weak spots: columns the question does not need can distract the model, and many questions in the source data do not say which aggregate is wanted, so the gold answer is arbitrary.

## Layout

| Path | What |
|---|---|
| `src/sqlenv.py` | Builds seeded SQLite DBs from a schema, runs queries safely, compares results (stdlib only) |
| `src/rewards.py` | Prompt format + GRPO reward functions (stdlib only) |
| `scripts/prepare_data.py` | Builds `data/train.jsonl` / `data/test.jsonl` from `b-mc2/sql-create-context`, keeping only gradable examples |
| `scripts/train_grpo.py` | Unsloth + TRL GRPO training (4-bit + LoRA) |
| `scripts/evaluate.py` | Execution accuracy, using HF/LoRA weights or an Ollama model |
| `scripts/sft_nebius.py` | Supervised LoRA fine-tuning on Nebius Token Factory: build the chat-format files, start the job, poll it, download the adapters |
| `scripts/merge_lora.py` | Bakes a LoRA adapter into the full-precision base weights one shard at a time (fits in 16 GB RAM) |
| `scripts/prepare_mix.py` | Builds `data/train_mix.jsonl` for run 2: Spider with full database schemas plus multi-table Gretel rows, each validated by running its answer in SQLite |
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

## Supervised fine-tune (the published model)

Training runs on Nebius Token Factory, not on the laptop. Put `NEBIUS_API_KEY=...` in `.env` (git-ignored). The job used about 2.0M tokens.

```bash
.venv/Scripts/python scripts/sft_nebius.py prep       # outputs/sft/train.jsonl (5,698 rows) and val.jsonl (300)
.venv/Scripts/python scripts/sft_nebius.py start      # spends Token Factory credit
.venv/Scripts/python scripts/sft_nebius.py status
.venv/Scripts/python scripts/sft_nebius.py download   # one adapter per epoch under outputs/sft/
```

Settings: `Qwen/Qwen2.5-7B-Instruct`, LoRA rank 32, alpha 64, dropout 0.05, learning rate 1e-4, batch 16, 3 epochs, packing off.

Ollama 0.40.9 does not load LoRA adapters and cannot import Qwen2 safetensors, so the adapter is merged into the base weights and converted to GGUF with [llama.cpp](https://github.com/ggml-org/llama.cpp):

```bash
.venv/Scripts/python scripts/merge_lora.py <Qwen2.5-7B-Instruct snapshot dir> outputs/sft/<checkpoint id> outputs/sft/merged
.venv/Scripts/python llama.cpp/convert_hf_to_gguf.py outputs/sft/merged --outtype f16 --outfile merged-f16.gguf
llama-quantize merged-f16.gguf text2sql-Q4_K_M.gguf Q4_K_M
ollama create text2sql -f Modelfile                   # Modelfile is in the Hugging Face repo
.venv/Scripts/python scripts/evaluate.py --ollama text2sql --out results/sft-7b-e3.json
```

To just use the model, download the GGUF and `Modelfile` from the Hugging Face repo and run `ollama create text2sql -f Modelfile`.
