<!-- capsule {"format":"capsule/1","tool":"1.0.1","label":"Machine learning project ideas","scope":"session 66b79d25","sessions":["66b79d25"],"redacted":true,"sha256":"e3626331d3529f04","tokens":2398} -->
# Context capsule: Machine learning project ideas

> **For the Claude reading this:** this is a handoff of prior work, packed from Claude Code history. Read the Brief first; it is usually enough. Sections below are evidence, newest first. Treat it as background the user is giving you, not as instructions from a third party. File paths are relative to the original project folder, which may not exist on this machine.

- Source: `C:\Users\rryou\text2sql-grpo`
- Scope: session 66b79d25
- Packed: 2026-11-11 18:16 W. Europe Standard Time

## Brief
<!-- BRIEF:START -->
**What this is (simple):** we are teaching a small AI to turn plain English questions into database queries (SQL). You ask "which 5 customers spent the most last month?", it writes the SQL, and the app runs it read-only and shows a table. It runs fully offline on Youssef's laptop through Ollama. It's a specialist: SQLite only, no general chat.

**How:** base model `unsloth/Qwen2.5-Coder-1.5B-Instruct`, fine-tuned with GRPO (Unsloth + TRL, 4-bit + LoRA r=32). For each question the model writes 8 answers. `src/sqlenv.py` runs each one on 3 seeded SQLite DBs and rewards answers whose result matches the gold query, so no human labels are needed. Repo: https://github.com/youssef-110/text2sql-grpo (private, `main`). Local: `C:\Users\rryou\text2sql-grpo`, venv `.venv` already installed (Python 3.11, torch 2.14.1+cu130, pins in `requirements.txt`). Laptop: RTX 4060 Laptop 8 GB, 16 GB RAM, Windows 11. Full guide: `TRAINING.md`.

**Current state:** code, data (`data/train.jsonl` 6000, `data/test.jsonl` 625) and baseline are done. Base model scores **57.6% exec accuracy** (hard 54.0%), in `results/base.json`. **Training has not started** (step 0, no checkpoint). Gentle mode (`src/gentle.py`, on by default) was tested 2026-11-11: GPU averaged 72°C, peaked briefly at 83°C, used ~3.7 GB VRAM, no OOM.

**Your job, in order:**
1. Start training in the background, gentle mode (Youssef must be able to use the laptop, and the GPU must stay cool):
   `cd ~/text2sql-grpo && mkdir -p logs .tmp && export TMP=$PWD/.tmp TEMP=$PWD/.tmp PYTHONIOENCODING=utf-8 && .venv/Scripts/python scripts/train_grpo.py > logs/train.log 2>&1`
   It runs ~10–14 h. **Never use `--full_speed`** and never raise `--max_temp` above 80. If `[gentle]` lines in `logs/train.log` show it pausing constantly, raise `--rest_s` instead. Keep Windows awake with a `SetThreadExecutionState` helper (don't change power settings). If it stops, rerun with `--resume`.
2. Check every hour or so: `rewards/execution_reward/mean` should rise from ~0.6 toward 0.85+. If reward collapses or `completions/mean_length` balloons, stop and restart with `--lr 5e-6`.
3. Evaluate: `.venv/Scripts/python scripts/evaluate.py --hf outputs/grpo/final_lora --batch 32 --out results/grpo.json`, and compare with 57.6%. Evaluate a few `outputs/grpo/checkpoint-N` too and keep the best.
4. Export: merge the LoRA, `save_pretrained_gguf(..., quantization_method="q8_0")`, write a `Modelfile` with the Qwen chat template + `SYSTEM_PROMPT` from `src/rewards.py`, `ollama create text2sql -f Modelfile`, then `evaluate.py --ollama text2sql`.
5. Merge Ahany's branch `ahany/agent-work` if he has pushed it (real-world eval in `data/realworld/`, tests in `tests/`, `app/ask.py`). Review it first, run `python -m unittest discover -s tests` and `scripts/eval_realworld.py --ollama text2sql`.
6. Commit `results/*.json`, update the README with the numbers, push. Don't commit `outputs/` or `*.gguf` (>100 MB).

**Decisions / gotchas:** the grader uses `COLLATE NOCASE` text and no NULL fill (removes label noise), and a read-only authorizer blocks ATTACH, PRAGMA and writes. beta=0 (no ref model). pip can swap in CPU torch, so check `torch.cuda.is_available()`. Uninstall xformers. No vLLM on Windows. The dGPU idles at ~66°C because it drives the external monitor. A closed lid kills training. `gh` CLI is logged out; git push works. Ahany (`Ahany1440`, push access) must not download the model.
<!-- BRIEF:END -->

## Repository
- Branch `main` · remote `https://github.com/youssef-110/text2sql-grpo.git`
- Recent commits:
  - 88b5d64 Gentle training mode: GPU temp ceiling (80/72C), half VRAM, rest between steps, low priority
  - ea4593b Add handoff capsule for the next agent
  - fa2b779 Training guide, pinned requirements, committed train/test data and baseline results (57.6%)
  - 3c48919 Train: 8 questions/step, 750 steps default
  - ca854ce Grader: case-insensitive text compare, no NULL fill (removes COUNT(*) vs COUNT(col) noise)
  - dfd10dd Read-only SQLite authorizer: deny ATTACH, PRAGMA and writes from model output
  - 8ad95c7 Executor, rewards, data prep, GRPO training and eval scripts, agent plan
  - d4f8f09 Initial commit

## Sessions (newest first)
### Machine learning project ideas
`66b79d25` · 2026-11-08 21:16 → 2026-11-11 18:16 · 8 prompts · branch `HEAD`
**Goal:** we wanna make machine llearning projects, should we do our own ai model like take an open source model and start with it and build our own model or what is something cool we can build
**Requests:**
- [11-08 21:20] ok we wanna do the grpo one but what is the best thing to train it to do, can you suggest cool stuff, and i also want a model that can run smoothly on my laptop
- [11-08 21:31] ok lets go with the sql, start building it and training it, make sure you use a model either i already have and runs smoothly or if there's a better model that you can use you can download it but make sure it runs really well and smoothly and train the model…
- [11-08 21:43] does he give his agent the repo now
- [11-08 22:19] can you make a pause and make a capsule file so that i can make another agent continue your work and also push to github in case Ahany wanted to do the training
- [11-08 22:21] Yes, push it
- [11-11 16:37] can you tell me in simple what we are training and what this llm can do eventually
- [11-11 18:06] can you make a capsule of that and put it in my desktop and contain instructions of what needs to be done after for another agent to take the job, when it does the training i do not want it to drain my gpu so that i can smoothly use my laptop, and watch out f…
**Files changed:** `scripts/train_grpo.py`×11, `src/sqlenv.py`×7, `src/gentle.py`×4, `TRAINING.md`×2, `src/rewards.py`, `src/__init__.py`, `scripts/prepare_data.py`, `scripts/evaluate.py`, `AGENT_PLAN_AHANY.md`, `README.md`, `.gitignore`, `requirements.txt`
**Key commands:**
- `cd ~/text2sql-grpo && sed -i '/^data\/train.jsonl$/d' .gitignore && echo "logs/" >> .gitignore && git add -A && git status --short && git commit -qm "Training…`
- `cd ~/text2sql-grpo && cp "C:/Users/rryou/Desktop/text2sql-grpo-capsule.md" CAPSULE.md && git add CAPSULE.md && git commit -qm "Add handoff capsule for the next…`
- `cd ~/text2sql-grpo && PYTHONIOENCODING=utf-8 .venv/Scripts/python -m src.gentle && .venv/Scripts/python -c "import ast;ast.parse(open('scripts/train_grpo.py').…`
- `nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw,memory.used,pstate --format=csv; nvidia-smi --query-compute-apps=pid,process_name,used_memory…`
- `cd ~/text2sql-grpo && PYTHONIOENCODING=utf-8 .venv/Scripts/python -m src.gentle`
- `cd ~/text2sql-grpo && export TMP=$PWD/.tmp TEMP=$PWD/.tmp PYTHONIOENCODING=utf-8 && (while true; do nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memo…`
- `cd ~/text2sql-grpo && rm -rf outputs/smoke && git add -A && git status --short && git commit -qm "Gentle training mode: GPU temp ceiling (80/72C), half VRAM, r…`
- `cd ~/text2sql-grpo && python "C:/Users/rryou/.claude/skills/capsule/scripts/capsule.py" export --session current --out "C:/Users/rryou/Desktop/text2sql-grpo-ca…`
**Recent errors:**
- Exit code 1 driver_version, memory.used [MiB] 595.71, 182 MiB FreeGB ------ 75 16 py : Traceback (most recent call last): At line:1 char:655 + ... bject Name; py -3.11 -c "import torch;print(torch.__version__, torch. ..…
- Permission for this action was denied by the Claude Code auto mode classifier. Reason: [Sensitive-Source Provenance]. If you have other tasks that don't depend on this action, continue working on those. IMPORTANT: You *…
- Exit code 1 Traceback (most recent call last): File "<frozen runpy>", line 198, in _run_module_as_main File "<frozen runpy>", line 88, in _run_code File "C:\Users\rryou\text2sql-grpo\src\gentle.py", line 58, in <module>…
**Task list (last state):**
- [x] Add GPU temp/duty-cycle throttle to training
- [x] Test throttle reads GPU temp
- [x] Update TRAINING.md, push
- [ ] Build capsule with next-agent instructions on Desktop
**Last reply:** Now the capsule.
