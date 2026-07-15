import os
import torch
from safetensors.torch import save_file

os.makedirs("outputs/adapters/dummy_a", exist_ok=True)
os.makedirs("outputs/adapters/dummy_b", exist_ok=True)

rank = 8
in_dim = 64
out_dim = 64

def make_dummy_adapter(path, seed):
    torch.manual_seed(seed)

    state = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(rank, in_dim) * 0.01,
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(out_dim, rank) * 0.01,
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": torch.randn(rank, in_dim) * 0.01,
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": torch.randn(out_dim, rank) * 0.01,
    }

    save_file(state, os.path.join(path, "adapter_model.safetensors"))

    with open(os.path.join(path, "adapter_config.json"), "w") as f:
        f.write("""{
  "r": 8,
  "lora_alpha": 16,
  "target_modules": ["q_proj", "v_proj"],
  "peft_type": "LORA",
  "task_type": "CAUSAL_LM"
}""")

make_dummy_adapter("outputs/adapters/dummy_a", 1)
make_dummy_adapter("outputs/adapters/dummy_b", 2)

print("Dummy LoRA adapters created.")
