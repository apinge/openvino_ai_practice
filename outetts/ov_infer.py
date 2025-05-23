from ov_outetts_helper import InterfaceOV, OVHFModel
import numpy as np
import time
model_dir = "OuteTTS-0.1-350M-ov"
interface = InterfaceOV(model_dir, "GPU")
start_time = time.time()
#zero-shot infer
tts_output = interface.generate(text="Born no bigger than a thumb within the cup of a flower, \
                                 Thumbelina embarked on a series of unexpected journeys through a world of giants,\
                                 her courage eventually leading her to a true home among winged kin.", temperature=0.1, repetition_penalty=1.1, max_length=4096)

end_time = time.time()
audio_data_float32 = tts_output.audio[0].numpy() # Your raw data, assumed to be float32
sample_rate = tts_output.sr

speech_len =  len(audio_data_float32) / sample_rate
print("specch_len is {}".format(speech_len))
print('yield speech len {} second, rtf {}'.format(speech_len, (end_time - start_time) / speech_len))

# --- Note: WAV files typically store integer types (e.g., int16) ---
# If your data is float32, you need to convert it to int16.
# During conversion, scale it to the int16 range [-32768, 32767].
# Simple scaling method: multiply by the maximum value of int16 and convert to int16 type.
audio_data_int16 = np.int16(audio_data_float32 * (32767 / np.max(np.abs(audio_data_float32))))
# If your float32 data is already in the range [-1.0, 1.0], you can simplify to:
# audio_data_int16 = np.int16(audio_data_float32 * 32767)



wav_filename = "output_tts.wav"
import scipy.io.wavfile

scipy.io.wavfile.write(wav_filename, sample_rate, audio_data_int16)
"""
Text-to-Speech generation with Voice Cloning
"""
from pathlib import Path
file_path = Path("samples","2.wav")

speaker = interface.create_speaker(file_path, "Hello, I can speak pretty well, but sometimes I make some mistakes.")

# Save the speaker to a file
interface.save_speaker(speaker, "speaker.pkl")

# Load the speaker from a file
speaker = interface.load_speaker("speaker.pkl")

# Generate TTS with the custom voice
start_time = time.time()
cloned_output = interface.generate(
    text="This is a cloned voice speaking",
    speaker=speaker,
    temperature=0.1,
    repetition_penalty=1.1,
    max_length=4096,
)
end_time = time.time()
print("specch_len is {}".format(speech_len))
print('yield speech len {} second, rtf {}'.format(speech_len, (end_time - start_time) / speech_len))

# save the output to a file
audio_data_float32 = cloned_output.audio[0].numpy() # Your raw data, assumed to be float32
sample_rate = cloned_output.sr
wav_filename = "output_tts_clone.wav"
audio_data_int16 = np.int16(audio_data_float32 * (32767 / np.max(np.abs(audio_data_float32))))
scipy.io.wavfile.write(wav_filename , sample_rate, audio_data_int16)

"""
Quantization
"""
from datasets import load_dataset
libritts = load_dataset("parler-tts/libritts_r_filtered", "clean", split="test.clean", streaming=True)

import nncf
from functools import partial
import numpy as np

def transform_fn(item, interface):
    text_normalized = item["text_normalized"]
    prompt = interface.prompt_processor.get_completion_prompt(text_normalized, interface.language, None)
    encoded = interface.prompt_processor.tokenizer(prompt, return_tensors="np")

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    position_ids = np.cumsum(attention_mask, axis=1) - 1
    position_ids[attention_mask == 0] = 1
    inputs["position_ids"] = position_ids

    batch_size = input_ids.shape[0]
    inputs["beam_idx"] = np.arange(batch_size, dtype=int)

    return inputs

hf_model = OVHFModel(model_dir,"CPU").model
dataset = nncf.Dataset(libritts, partial(transform_fn, interface=interface))

quantized_model = nncf.quantize(
    hf_model.model,
    dataset,
    preset=nncf.QuantizationPreset.MIXED,
    model_type=nncf.ModelType.TRANSFORMER,
    ignored_scope=nncf.IgnoredScope(
        patterns=[
            # We need to use ignored scope for this pattern to generate the most efficient model
            "__module.model.layers.*.self_attn/aten::scaled_dot_product_attention/ScaledDotProductAttention"
        ]
    )
)

hf_model.model = quantized_model
model_dir_quantized = Path(f"{model_dir}_quantized")
hf_model.save_pretrained(model_dir_quantized)
interface.prompt_processor.tokenizer.save_pretrained(model_dir_quantized)