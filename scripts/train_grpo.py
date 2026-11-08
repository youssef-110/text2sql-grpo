"""GRPO fine-tune of a small coder model on execution-checked text-to-SQL. Fits an 8 GB GPU (4-bit + LoRA).

    python scripts/train_grpo.py --max_steps 400
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unsloth import FastLanguageModel  # noqa: E402  (must import before trl/transformers)
from datasets import Dataset  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from src.rewards import build_messages, execution_reward, length_penalty  # noqa: E402

BASE = "unsloth/Qwen2.5-Coder-1.5B-Instruct"


def load_rows(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return Dataset.from_list([
        {"prompt": build_messages(r["schema"], r["question"]), "schema": r["schema"],
         "answer": r["answer"], "seeds": r["seeds"]}
        for r in rows
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--train", default=str(ROOT / "data" / "train.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "grpo"))
    ap.add_argument("--max_steps", type=int, default=750)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    model, tok = FastLanguageModel.from_pretrained(a.model, max_seq_length=1024, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=32, lora_alpha=64, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=0,
    )

    cfg = GRPOConfig(
        output_dir=a.out,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        weight_decay=0.0,
        max_grad_norm=0.5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=a.grad_accum,  # 8 x 8 completions = 8 questions per step
        num_generations=8,  # group size: 8 samples per question, rewards compared within the group
        max_prompt_length=600,
        max_completion_length=200,
        temperature=1.0,
        beta=0.0,  # no reference model: saves VRAM
        max_steps=a.max_steps,
        logging_steps=1,
        save_steps=50,
        save_total_limit=4,
        report_to="none",
        bf16=True,
        seed=0,
    )
    trainer = GRPOTrainer(
        model=model, processing_class=tok, args=cfg,
        reward_funcs=[execution_reward, length_penalty],
        train_dataset=load_rows(a.train),
    )
    trainer.train(resume_from_checkpoint=a.resume or None)
    model.save_pretrained(Path(a.out) / "final_lora")
    tok.save_pretrained(Path(a.out) / "final_lora")


if __name__ == "__main__":
    main()
