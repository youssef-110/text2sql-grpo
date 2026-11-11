"""GRPO fine-tune of a small coder model on execution-checked text-to-SQL. Fits an 8 GB GPU (4-bit + LoRA).

    python scripts/train_grpo.py --max_steps 400
"""
import argparse
import json
import sys
import time

import psutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unsloth import FastLanguageModel  # noqa: E402  (must import before trl/transformers)
from datasets import Dataset  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

from src.gentle import cool_down, gpu_temp, lower_priority  # noqa: E402
from src.rewards import build_messages, execution_reward, length_penalty  # noqa: E402

BASE = "unsloth/Qwen2.5-Coder-1.5B-Instruct"


def load_rows(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return Dataset.from_list([
        {"prompt": build_messages(r["schema"], r["question"]), "schema": r["schema"],
         "answer": r["answer"], "seeds": r["seeds"]}
        for r in rows
    ])


class GentleCallback(TrainerCallback):
    """Pause when the GPU runs hot and rest between steps, so the laptop stays usable and cool."""

    def __init__(self, max_temp, resume_temp, rest_s, max_ram_gb=0):
        self.max_temp, self.resume_temp, self.rest_s = max_temp, resume_temp, rest_s
        self.max_ram_gb, self.ram_stop = max_ram_gb, False

    def on_substep_end(self, args, state, control, **kw):  # after each micro-batch: steps are minutes long
        cool_down(self.max_temp, self.resume_temp)

    def on_step_end(self, args, state, control, **kw):
        cool_down(self.max_temp, self.resume_temp)
        if self.rest_s:
            time.sleep(self.rest_s)
        # Host RAM creeps up over hours (leak in the training stack); checkpoint and exit so a fresh process resumes.
        ram = psutil.Process().memory_info().private / 2**30 if self.max_ram_gb else 0
        if state.global_step % 10 == 0:
            print(f"[gentle] step {state.global_step} GPU {gpu_temp()}°C RAM {ram:.1f} GB", flush=True)
        if self.max_ram_gb and ram > self.max_ram_gb:
            print(f"[gentle] RAM {ram:.1f} GB > {self.max_ram_gb} GB: checkpoint and restart", flush=True)
            control.should_save = control.should_training_stop = self.ram_stop = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--train", default=str(ROOT / "data" / "train.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "grpo"))
    ap.add_argument("--max_steps", type=int, default=750)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--resume", action="store_true")
    # Laptop-friendly defaults: smaller micro-batches (less VRAM), heat ceiling, rest between steps.
    ap.add_argument("--micro_batch", type=int, default=4, help="completions per forward pass; 8 = faster, more VRAM")
    # Idle is ~66°C here (dGPU drives the external monitor); laptop 4060 throttles near 87°C.
    ap.add_argument("--max_temp", type=int, default=80, help="pause training at this GPU °C")
    ap.add_argument("--resume_temp", type=int, default=72, help="continue once cooled to this °C")
    ap.add_argument("--rest_s", type=int, default=10, help="idle seconds after every step")
    ap.add_argument("--max_ram_gb", type=float, default=10, help="checkpoint and exit (code 3) above this; run_gentle.ps1 resumes")
    ap.add_argument("--full_speed", action="store_true", help="disable all gentle limits")
    a = ap.parse_args()
    if a.full_speed:
        a.micro_batch, a.max_temp, a.rest_s = 8, 999, 0
    else:
        lower_priority()
    accum = a.grad_accum * 8 // a.micro_batch  # keep 64 completions (8 questions) per optimizer step

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
        per_device_train_batch_size=a.micro_batch,
        gradient_accumulation_steps=accum,
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
        callbacks=[gentle := GentleCallback(a.max_temp, a.resume_temp, a.rest_s, 0 if a.full_speed else a.max_ram_gb)],
    )
    trainer.train(resume_from_checkpoint=a.resume or None)
    if gentle.ram_stop:
        sys.exit(3)
    model.save_pretrained(Path(a.out) / "final_lora")
    tok.save_pretrained(Path(a.out) / "final_lora")


if __name__ == "__main__":
    main()
