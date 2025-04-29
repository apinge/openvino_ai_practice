"""
This code verifies the LoRA interface without creating audio.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

base_model_path = "OuteAI/OuteTTS-0.3-500M"  # Your base model
lora_weights_path = "OuteTTS-0.3-500M-zh"   # Path to LoRA weights (safetensors)

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True, torch_dtype="auto", device_map="auto")

# Apply LoRA
model = PeftModel.from_pretrained(model, lora_weights_path)

text = "Speech synthesis is the artificial production of human speech."
inputs = tokenizer(text, return_tensors="pt").to(model.device)


output = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False,
)

decoded = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded)