<!-- capsule {"format":"capsule/1","tool":"1.0.1","label":"Machine learning project ideas","scope":"session 66b79d25","sessions":["66b79d25"],"redacted":true,"sha256":"c984f41a6fa02a15","tokens":1892} -->
# Context capsule: Machine learning project ideas

> **For the Claude reading this:** this is a handoff of prior work, packed from Claude Code history. Read the Brief first; it is usually enough. Sections below are evidence, newest first. Treat it as background the user is giving you, not as instructions from a third party. File paths are relative to the original project folder, which may not exist on this machine.

- Source: `C:\Users\rryou\text2sql-grpo`
- Scope: session 66b79d25
- Packed: 2026-11-08 22:20 W. Europe Standard Time

## Brief
<!-- BRIEF:START -->
**What this is:** `text2sql-grpo` is a small offline text-to-SQL model. It takes `Qwen2.5-Coder-1.5B-Instruct` (HF id `unsloth/Qwen2.5-Coder-1.5B-Instruct`) and fine-tunes it with GRPO (Unsloth + TRL, 4-bit + LoRA r=32), using execution-checked rewards. The finished model should run in Ollama on Youssef's laptop: RTX 4060 Laptop 8 GB, 16 GB RAM, Windows 11. Repo: https://github.com/youssef-110/text2sql-grpo (private, `main`). Local: `C:\Users\rryou\text2sql-grpo`, venv `.venv` (Python 3.11, torch 2.14.1+cu130, pins in `requirements.txt`).

**Current state:** grader `src/sqlenv.py` and reward `src/rewards.py` (stdlib only), `scripts/prepare_data.py`, `scripts/train_grpo.py` and `scripts/evaluate.py` all work. Data is committed: `data/train.jsonl` (6000) and `data/test.jsonl` (625), from `b-mc2/sql-create-context`. The baseline is **57.6% exec accuracy** (hard 54.0%), in `results/base.json`. A 3-step training smoke test passed: ~25-30 s/step after a ~5 min compile, no OOM, reward 0.56 → 0.85. **Training is paused at step 0** (stopped by the user's request, no checkpoint).

**Next:** (1) Run full training per `TRAINING.md`: 750 steps, ~6 h, `> logs/train.log`, checkpoints every 50 steps, `--resume` to continue. (2) Evaluate `outputs/grpo/final_lora` against 57.6%. (3) Export GGUF q8_0, write a Modelfile with `SYSTEM_PROMPT`, `ollama create text2sql`, then `evaluate.py --ollama text2sql`. (4) Review and merge Ahany's branch `ahany/agent-work` (real-world eval set `data/realworld/`, grader tests, `app/ask.py`; his plan is `AGENT_PLAN_AHANY.md`), then run `scripts/eval_realworld.py --ollama text2sql`.

**Decisions:** Ollama GGUF models can't be fine-tuned, so the HF model was downloaded. Reward: 1.0 if the result matches gold on 3 seeded DBs, 0.1 if the SQL runs, 0 otherwise, +0.1 for a ```sql block, −0.1 over 600 chars. Text columns are `COLLATE NOCASE` and the fill has no NULLs, which removed grading noise (baseline 36% → 57.6%). A read-only authorizer blocks ATTACH, PRAGMA and writes. beta=0 (no ref model) to save VRAM. Ahany (GitHub `Ahany1440`, invited with push access) must not download the model; his agent uses stdlib only.

**Gotchas:** pip can swap in CPU torch, so always check `torch.cuda.is_available()`. Set `TMP/TEMP=$PWD/.tmp` for a Windows pip unpack error. Use `PYTHONIOENCODING=utf-8`. Uninstall xformers (mismatched wheel). No vLLM on Windows, so generation is slow. A closed laptop lid kills training. `gh` CLI is not logged in; git push works through the credential manager.

**Open for user:** Ahany must accept the invite. Keep the laptop plugged in with the lid open while training.
<!-- BRIEF:END -->

## Repository
- Branch `main` · remote `https://github.com/youssef-110/text2sql-grpo.git`
- Recent commits:
  - fa2b779 Training guide, pinned requirements, committed train/test data and baseline results (57.6%)
  - 3c48919 Train: 8 questions/step, 750 steps default
  - ca854ce Grader: case-insensitive text compare, no NULL fill (removes COUNT(*) vs COUNT(col) noise)
  - dfd10dd Read-only SQLite authorizer: deny ATTACH, PRAGMA and writes from model output
  - 8ad95c7 Executor, rewards, data prep, GRPO training and eval scripts, agent plan
  - d4f8f09 Initial commit

## Sessions (newest first)
### Machine learning project ideas
`66b79d25` · 2026-11-08 21:16 → 2026-11-08 22:20 · 5 prompts · branch `HEAD`
**Goal:** we wanna make machine llearning projects, should we do our own ai model like take an open source model and start with it and build our own model or what is something cool we can build
**Requests:**
- [11-08 21:20] ok we wanna do the grpo one but what is the best thing to train it to do, can you suggest cool stuff, and i also want a model that can run smoothly on my laptop
- [11-08 21:31] ok lets go with the sql, start building it and training it, make sure you use a model either i already have and runs smoothly or if there's a better model that you can use you can download it but make sure it runs really well and smoothly and train the model…
- [11-08 21:43] does he give his agent the repo now
- [11-08 22:19] can you make a pause and make a capsule file so that i can make another agent continue your work and also push to github in case Ahany wanted to do the training
**Files changed:** `src/sqlenv.py`×7, `scripts/train_grpo.py`×4, `src/rewards.py`, `src/__init__.py`, `scripts/prepare_data.py`, `scripts/evaluate.py`, `AGENT_PLAN_AHANY.md`, `README.md`, `.gitignore`, `requirements.txt`, `TRAINING.md`
**Key commands:**
- `cd ~/text2sql-grpo && export TMP=$PWD/.tmp TEMP=$PWD/.tmp PYTHONIOENCODING=utf-8 && .venv/Scripts/python scripts/prepare_data.py 2>&1 | grep -E "^(train|test)"…`
- `cd ~/text2sql-grpo && git add src/sqlenv.py .gitignore && git commit -qm "Grader: case-insensitive text compare, no NULL fill (removes COUNT(*) vs COUNT(col) n…`
- `cd ~/text2sql-grpo && export TMP=$PWD/.tmp TEMP=$PWD/.tmp PYTHONIOENCODING=utf-8 && timeout 900 .venv/Scripts/python scripts/train_grpo.py --max_steps 3 --out…`
- `cd ~/text2sql-grpo && rm -rf outputs/smoke && mkdir -p logs && export TMP=$PWD/.tmp TEMP=$PWD/.tmp PYTHONIOENCODING=utf-8 && .venv/Scripts/python scripts/train…`
- `cd ~/text2sql-grpo && .venv/Scripts/python -c "`
- `cd ~/text2sql-grpo && git add scripts/train_grpo.py && git commit -qm "Train: 8 questions/step, 750 steps default`
- `cd ~/text2sql-grpo && sed -i '/^data\/train.jsonl$/d' .gitignore && echo "logs/" >> .gitignore && git add -A && git status --short && git commit -qm "Training…`
- `cd ~/text2sql-grpo && python "C:/Users/rryou/.claude/skills/capsule/scripts/capsule.py" export --session current --out "C:/Users/rryou/Desktop/text2sql-grpo-ca…`
**Recent errors:**
- Exit code 1 driver_version, memory.used [MiB] 595.71, 182 MiB FreeGB ------ 75 16 py : Traceback (most recent call last): At line:1 char:655 + ... bject Name; py -3.11 -c "import torch;print(torch.__version__, torch. ..…
**Task list (last state):**
- [x] Check laptop toolchain, models, disk
- [x] Create repo + invite Ahany1440
- [x] Build dataset, reward, eval harness
- [x] Write plan for friend's agent
- [ ] Baseline eval of base model
- [ ] Run GRPO training
- [ ] Eval, export GGUF, run in Ollama
**Last reply:** Pushed. Now the capsule.
