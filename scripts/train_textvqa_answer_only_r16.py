import os
import torch
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model

model_dir = os.environ.get("MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
output_dir = os.environ.get("OUTPUT_DIR", "outputs/adapters/textvqa_answer_only_1000")
os.makedirs(output_dir, exist_ok=True)

split = os.environ.get("SPLIT", "train")
max_samples = int(os.environ.get("MAX_SAMPLES", "1000"))
max_steps = int(os.environ.get("MAX_STEPS", "1000"))

print("Loading TextVQA dataset...")
ds = load_dataset("lmms-lab/textvqa", split=f"{split}[:{max_samples}]")
print(ds)

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)

print("Loading Qwen2.5-VL-7B...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)

model.config.use_cache = False
model.train()

print("Adding LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-5,
)

def normalize_answer(ans):
    if isinstance(ans, list):
        if len(ans) == 0:
            return "unknown"
        return str(ans[0])
    return str(ans)

def make_inputs(image, question, answer):
    user_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": str(question)},
            ],
        }
    ]

    full_messages = [
        user_messages[0],
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": str(answer)}
            ],
        },
    ]

    # prompt: user + assistant generation prefix
    prompt_text = processor.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # full: user + assistant answer
    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    image_inputs, video_inputs = process_vision_info(user_messages)

    prompt_inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    full_inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    labels = full_inputs["input_ids"].clone()

    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels[:, :prompt_len] = -100

    if processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100

    # 如果全被 mask 掉，说明 template 对齐有问题，直接返回 None
    if (labels != -100).sum().item() == 0:
        return None

    full_inputs["labels"] = labels
    return full_inputs

print("Start TextVQA answer-only LoRA training...")

step = 0
loss_history = []

for ex in ds:
    image = ex["image"]
    question = ex["question"]
    answer = normalize_answer(ex["answers"])

    inputs = make_inputs(image, question, answer)
    if inputs is None:
        print(f"WARNING: empty answer labels at step={step}; skip")
        step += 1
        continue

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    optimizer.zero_grad(set_to_none=True)

    outputs = model(**inputs)
    loss = outputs.loss

    if not torch.isfinite(loss):
        print(f"WARNING: non-finite loss at step={step}; skip")
        step += 1
        continue

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    loss_value = float(loss.item())
    loss_history.append(loss_value)

    recent = sum(loss_history[-20:]) / min(20, len(loss_history))

    print(
        f"step={step} loss={loss_value:.4f} recent20={recent:.4f} "
        f"q={str(question)[:50]} ans={str(answer)[:30]}"
    )

    step += 1
    if step >= max_steps:
        break

print(f"Saving adapter to {output_dir}...")
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print("Done.")
