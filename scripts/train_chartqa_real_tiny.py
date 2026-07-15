import os
import torch
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model

model_dir = "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"
output_dir = "outputs/adapters/chartqa_real_tiny"
os.makedirs(output_dir, exist_ok=True)

dataset_name = os.environ.get("DATASET_NAME", "lmms-lab/ChartQA")
split = os.environ.get("SPLIT", "train")
max_samples = int(os.environ.get("MAX_SAMPLES", "30"))
max_steps = int(os.environ.get("MAX_STEPS", "30"))

print("Loading ChartQA dataset:", dataset_name)
ds = load_dataset(dataset_name, split=f"{split}[:{max_samples}]")
print(ds)
print("Columns:", ds.column_names)

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)

print("Loading Qwen2.5-VL-7B...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True,
)

model.config.use_cache = False
model.train()

print("Adding LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
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
    lr=2e-4,
)

def pick(ex, names):
    for n in names:
        if n in ex and ex[n] is not None:
            return ex[n]
    return None

def normalize_answer(ans):
    if isinstance(ans, list):
        if len(ans) == 0:
            return "unknown"
        return str(ans[0])
    return str(ans)

print("Start ChartQA tiny LoRA training...")

step = 0
for ex in ds:
    image = pick(ex, ["image", "decoded_image", "img", "figure"])
    question = pick(ex, ["question", "query", "Question", "input"])
    answer = pick(ex, ["answer", "answers", "label", "Label", "output"])

    if image is None or question is None or answer is None:
        print("Skip example keys:", ex.keys())
        continue

    answer = normalize_answer(answer)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": str(question)},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer}
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels

    optimizer.zero_grad()
    outputs = model(**inputs)
    loss = outputs.loss
    print(f"step={step} loss={loss.item():.4f} q={str(question)[:60]} ans={answer[:40]}")
    loss.backward()
    optimizer.step()

    step += 1
    if step >= max_steps:
        break

print(f"Saving adapter to {output_dir}...")
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print("Done.")
