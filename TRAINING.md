# Training guide (for whoever runs the GPU part)

Status as of 2026-11-08: everything is built and tested, and training is **paused at step 0** (no checkpoint yet). Start from step 1 below.

## Needs

- NVIDIA GPU with **8 GB+ VRAM**. Tested: RTX 4060 Laptop 8 GB, driver 595, Windows 11.
- Python 3.11, about 15 GB disk (venv ~8 GB + model ~3 GB).
- Downloads: about 3 GB for `unsloth/Qwen2.5-Coder-1.5B-Instruct` on the first run, plus pip packages. `data/train.jsonl` and `data/test.jsonl` are committed, so you don't need to download the dataset or run `prepare_data.py`.

## 1. Setup

Windows (Git Bash):
```bash
py -3.11 -m venv .venv
.venv/Scripts/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/pip uninstall -y xformers   # if something pulled it in
.venv/Scripts/python -c "import torch; print(torch.cuda.is_available())"   # must print True
```
Linux: same, but use `python3.11 -m venv .venv` and `.venv/bin/`. Drop `triton-windows`. With Linux you can also `pip install vllm` and add `use_vllm=True` in `GRPOConfig` for much faster generation.

**Gotchas we hit:**
- `pip install unsloth` without pinning can silently replace CUDA torch with a **CPU** build. Always check `torch.cuda.is_available()` afterwards. If it shows False, re-run the torch install with `--force-reinstall --no-deps`.
- A pip `OSError: No such file ... pip-unpack-*` on Windows was fixed by `export TMP=$PWD/.tmp TEMP=$PWD/.tmp` (make `.tmp/` first).
- Set `PYTHONIOENCODING=utf-8` on Windows, or Unsloth's emoji banner can crash the console.

## 2. Baseline (already done)

`results/base.json`: base model **57.6%** execution accuracy (hard: 54.0%, valid SQL: 100%) on `data/test.jsonl` (625 questions).
```bash
.venv/Scripts/python scripts/evaluate.py --hf unsloth/Qwen2.5-Coder-1.5B-Instruct --batch 32 --out results/base.json
```

## 3. Train

```bash
mkdir -p logs .tmp
export TMP=$PWD/.tmp TEMP=$PWD/.tmp PYTHONIOENCODING=utf-8
.venv/Scripts/python scripts/train_grpo.py > logs/train.log 2>&1
```
- Defaults: 750 steps, 8 questions × 8 samples per step, LoRA r=32, lr 1e-5, cosine, 4-bit base. That's one pass over the 6000 training questions.
- Speed on a 4060: the first step is ~5 min (compile), then ~25–30 s/step, so **~6 h total**. Keep it plugged in with the lid open.
- Checkpoints go to `outputs/grpo/checkpoint-N` every 50 steps. Resume after a stop with `--resume`.
- Watch progress: `rewards/execution_reward/mean` in `logs/train.log` should rise from ~0.6 toward 0.85+. If `completions/mean_length` balloons or reward collapses, stop and lower `--lr` to 5e-6.
- In the smoke test, steps 1–3 had reward 0.56, 0.61 and 0.85, with no OOM.

## 4. Evaluate

```bash
.venv/Scripts/python scripts/evaluate.py --hf outputs/grpo/final_lora --batch 32 --out results/grpo.json
```
Compare with 57.6%. You can also evaluate any `outputs/grpo/checkpoint-N` the same way to pick the best one.

## 5. Export to Ollama (not written yet)

Plan: merge the LoRA and save GGUF q8_0 with Unsloth (`model.save_pretrained_gguf("gguf", tok, quantization_method="q8_0")`). Write a `Modelfile` with the Qwen chat template and the `SYSTEM_PROMPT` from `src/rewards.py`, then `ollama create text2sql -f Modelfile`. Check it with `scripts/evaluate.py --ollama text2sql`, and with `scripts/eval_realworld.py --ollama text2sql` once Ahany's branch is merged.

## 6. Share results

Commit `results/*.json` and the eval numbers. **Don't commit** `outputs/` or `*.gguf`, because they're over GitHub's 100 MB limit. If the adapter needs sharing, upload `outputs/grpo/final_lora` to a Hugging Face model repo or a GitHub Release asset.
