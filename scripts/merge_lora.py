"""Bake a LoRA adapter into full-precision base weights, one shard at a time (fits in 16 GB RAM).

  python scripts/merge_lora.py <base snapshot dir> <adapter dir> <out dir>

The out dir is a plain safetensors model that `ollama create --quantize` can import.
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

base, adapter, out = map(Path, sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)

cfg = json.loads((adapter / "adapter_config.json").read_text())
assert not cfg.get("use_rslora") and not cfg.get("use_dora"), "only plain LoRA is handled"
scale = cfg["lora_alpha"] / cfg["r"]

lora = safe_open(str(adapter / "adapter_model.safetensors"), "pt")
# "<...>.q_proj.lora_A.weight" -> base key "<...>.q_proj.weight"
targets = {k.replace(".lora_A.weight", ".weight").removeprefix("base_model.model."): k
           for k in lora.keys() if k.endswith(".lora_A.weight")}

merged = 0
for shard in sorted(base.glob("*.safetensors")):
    tensors = load_file(str(shard))
    for name in tensors.keys() & targets.keys():
        a = lora.get_tensor(targets[name]).float()
        b = lora.get_tensor(targets[name].replace("lora_A", "lora_B")).float()
        w = tensors[name]
        tensors[name] = (w.float() + scale * (b @ a)).to(w.dtype)
        merged += 1
    save_file(tensors, str(out / shard.name), metadata={"format": "pt"})
    print(shard.name, "done", flush=True)
    del tensors

assert merged == len(targets), f"merged {merged} of {len(targets)} LoRA pairs: key names do not match the base"
for f in base.iterdir():
    if f.suffix != ".safetensors" and f.is_file():
        shutil.copy(f, out / f.name)
print(f"merged {merged} LoRA pairs, scale {scale}, into {out}")
