"""
Load the original model and inference
"""
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import time

device = "cpu"

repo_id = "parler-tts/parler_tts_mini_v0.1"
model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(repo_id)

prompt = "Hey there, you troublemaker! What kind of mischief are you up to today?"
description = "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. She speaks very fast."
#description = "A male speaker with indian accent"
start_time = time.time()
input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
end_time = time.time()

audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out-torch.wav", audio_arr, model.config.sampling_rate)

speech_len =  float(audio_arr.shape[0]) / model.config.sampling_rate
print('yield speech len {} second, rtf {}'.format(speech_len, (end_time - start_time) / speech_len))
print("Torch Done!")