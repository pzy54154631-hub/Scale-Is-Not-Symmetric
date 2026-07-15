import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

base_model_dir = "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"
adapter_dir = "outputs/merged/textvqa_chartqa_real_mirage"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(base_model_dir, local_files_only=True)

print("Loading base model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True,
)

print("Loading merged MIRAGE adapter...")
model = PeftModel.from_pretrained(
    model,
    adapter_dir,
    local_files_only=True,
)
model.eval()

image = Image.new("RGB", (224, 224), color=(255, 255, 255))

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What is in this image? Answer briefly."},
        ],
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
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

print("Generating with merged MIRAGE LoRA...")
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=32,
    )

generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print("MODEL OUTPUT:")
print(output_text[0])
