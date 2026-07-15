import os
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model

model_dir = "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"
output_dir = os.environ.get("OUTPUT_DIR", "outputs/adapters/toy_lora")
question = os.environ.get("QUESTION", "What is in this image? Answer briefly.")
answer = os.environ.get("ANSWER", "The image is blank and white.")

os.makedirs(output_dir, exist_ok=True)

print("Output adapter:", output_dir)
print("Answer target:", answer)

processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True,
)

model.config.use_cache = False
model.train()

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

image = Image.new("RGB", (224, 224), color=(255, 255, 255))

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
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

print("Start toy LoRA training...")
for step in range(5):
    optimizer.zero_grad()
    outputs = model(**inputs)
    loss = outputs.loss
    print(f"step={step} loss={loss.item():.4f}")
    loss.backward()
    optimizer.step()

print("Saving adapter...")
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print("Done.")
