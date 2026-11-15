"""Supervised LoRA fine-tuning on Nebius Token Factory (needs NEBIUS_API_KEY in .env).

  prep      write outputs/sft/{train,val}.jsonl in chat format from data/train.jsonl
  start     upload both files and create the job (this spends credit)
  status    print job status, progress and recent events
  download  save every checkpoint's adapter files under outputs/sft/<checkpoint id>/
"""
import argparse
import json
import random
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rewards import build_messages, extract_sql  # noqa: E402

API = "https://api.tokenfactory.nebius.com/v1"
OUT = ROOT / "outputs" / "sft"  # replaced by outputs/<--tag> in main
JOB = OUT / "job.json"
VAL_ROWS = 300


def headers():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("NEBIUS_API_KEY="):
            return {"Authorization": "Bearer " + line.split("=", 1)[1].strip()}
    sys.exit("NEBIUS_API_KEY missing in .env")


def call(method, path, **kw):
    r = requests.request(method, API + path, headers=headers(), timeout=300, **kw)
    if not r.ok:
        sys.exit(f"{method} {path} -> HTTP {r.status_code}: {r.text[:500]}")
    return r


def prep(a):
    rows = [json.loads(l) for l in open(ROOT / a.data, encoding="utf-8")]
    random.Random(0).shuffle(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in (("val", rows[:VAL_ROWS]), ("train", rows[VAL_ROWS:])):
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            kept = 0
            for r in part:
                reply = f"```sql\n{r['answer']}\n```"
                if extract_sql(reply) is None:  # grader rejects it (e.g. ';' inside a string), so don't teach it
                    continue
                kept += 1
                msgs = build_messages(r["schema"], r["question"]) + [{"role": "assistant", "content": reply}]
                f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
        print(name, kept, "rows kept of", len(part))


def start(a):
    if JOB.exists():
        sys.exit(f"{JOB} exists: a job was already started. Delete it to start another.")
    ids = {}
    for name in ("train", "val"):
        with open(OUT / f"{name}.jsonl", "rb") as f:
            ids[name] = call("POST", "/files", files={"file": f}, data={"purpose": "fine-tune"}).json()["id"]
    job = call("POST", "/fine_tuning/jobs", json={
        "model": a.model, "suffix": f"text2sql-{a.tag}", "seed": 42,
        "training_file": ids["train"], "validation_file": ids["val"],
        "hyperparameters": {
            "n_epochs": a.epochs, "batch_size": 16, "learning_rate": a.lr, "warmup_ratio": 0.03,
            "lora": True, "lora_r": a.rank, "lora_alpha": 2 * a.rank, "lora_dropout": 0.05,
            # packing off: ~130-token rows packed into 8192 would leave only a few dozen optimizer steps
            "packing": False, "context_length": 8192,
        },
    }).json()
    JOB.write_text(json.dumps(job, indent=1))
    print("job", job["id"], job["status"])


def job_id():
    return json.loads(JOB.read_text())["id"]


def status(_):
    j = call("GET", f"/fine_tuning/jobs/{job_id()}").json()
    print({k: j.get(k) for k in ("status", "trained_steps", "total_steps", "trained_tokens", "error")})
    for e in call("GET", f"/fine_tuning/jobs/{job_id()}/events", params={"limit": 15}).json()["data"]:
        print(" ", e["level"], e["message"])


def download(_):
    for c in call("GET", f"/fine_tuning/jobs/{job_id()}/checkpoints").json()["data"]:
        d = OUT / c["id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "metrics.json").write_text(json.dumps({"step": c["step_number"], **c["metrics"]}))
        for fid in c["result_files"]:
            name = Path(call("GET", f"/files/{fid}").json()["filename"]).name
            (d / name).write_bytes(call("GET", f"/files/{fid}/content").content)
        print(c["id"], "step", c["step_number"], c["metrics"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "start", "status", "download"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--data", default="data/train.jsonl", help="rows of {question, schema, answer}")
    ap.add_argument("--tag", default="sft", help="run name: files go to outputs/<tag>/")
    a = ap.parse_args()
    OUT = ROOT / "outputs" / a.tag
    JOB = OUT / "job.json"
    {"prep": prep, "start": start, "status": status, "download": download}[a.cmd](a)
