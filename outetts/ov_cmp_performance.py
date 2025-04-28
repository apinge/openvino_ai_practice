from ov_outetts_helper import InterfaceOV, OVHFModel
import numpy as np
import time
model_dir_quantized = "OuteTTS-0.1-350M-ov_quantized"
model_dir = "OuteTTS-0.1-350M-ov"
"""
Comparing model Performance
"""
import time
import tqdm
from datasets import load_dataset
libritts = load_dataset("parler-tts/libritts_r_filtered", "clean", split="test.clean", streaming=True)

def calculate_inference_time(interface, dataset, limit):
    inference_time = []
    for i, item in tqdm.tqdm(enumerate(dataset), total=limit):
        if i > limit: break
        start = time.perf_counter()
        _ = interface.generate(
            text=item["text_normalized"],
            max_length=256,
            additional_gen_config={
                "pad_token_id": interface.prompt_processor.tokenizer.eos_token_id
            }
        )
        end = time.perf_counter()
        delta = end - start
        inference_time.append(delta)
    return np.median(inference_time)

interface = InterfaceOV(model_dir, "GPU")
limit = 25

fp_inference_time = calculate_inference_time(interface, libritts, limit)
print(f"Original model generate time: {fp_inference_time}")

interface_quantized = InterfaceOV(model_dir_quantized, "CPU")
int_inference_time = calculate_inference_time(interface_quantized, libritts, limit)
print(f"Quantized model generate time: {int_inference_time}")