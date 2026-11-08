"""Execution accuracy on a JSONL test set.

Backends:
  --hf <model or lora dir>     transformers/unsloth on the GPU (greedy)
  --ollama <model name>        a local Ollama model over HTTP (stdlib only)
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rewards import build_messages, score  # noqa: E402


def ollama_generate(model, url):
    def gen(batch):
        outs = []
        for msgs in batch:
            body = json.dumps({"model": model, "messages": msgs, "stream": False,
                               "options": {"temperature": 0, "num_predict": 256}}).encode()
            req = urllib.request.Request(f"{url}/api/chat", body, {"Content-Type": "application/json"})
            outs.append(json.load(urllib.request.urlopen(req, timeout=120))["message"]["content"])
        return outs
    return gen


def hf_generate(path):
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(path, max_seq_length=1024, load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    tok.padding_side = "left"

    def gen(batch):
        texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        out = model.generate(**enc, max_new_tokens=256, do_sample=False)
        return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--hf")
    g.add_argument("--ollama")
    ap.add_argument("--ollama_url", default="http://localhost:11434")
    ap.add_argument("--data", default=str(ROOT / "data" / "test.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    rows = rows[: a.limit] if a.limit else rows
    gen = hf_generate(a.hf) if a.hf else ollama_generate(a.ollama, a.ollama_url)

    results, t0 = [], time.time()
    for i in range(0, len(rows), a.batch):
        chunk = rows[i : i + a.batch]
        outs = gen([build_messages(r["schema"], r["question"]) for r in chunk])
        for r, o in zip(chunk, outs):
            s = score(r["schema"], r["answer"], o, r.get("seeds") or (11, 22, 33))
            results.append({**r, "output": o, "correct": s >= 1.0, "runs": s >= 0.1})
        done = len(results)
        print(f"{done}/{len(rows)} acc={sum(x['correct'] for x in results)/done:.3f}", flush=True)

    n = len(results)
    hard = [x for x in results if x.get("hard")]
    summary = {
        "n": n,
        "exec_accuracy": sum(x["correct"] for x in results) / n,
        "hard_accuracy": sum(x["correct"] for x in hard) / max(1, len(hard)),
        "valid_sql_rate": sum(x["runs"] for x in results) / n,
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(summary, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
