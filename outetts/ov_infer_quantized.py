from ov_outetts_helper import InterfaceOV, OVHFModel
import numpy as np
import time
model_dir = "OuteTTS-0.1-350M-ov_quantized"
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



wav_filename = "output_tts_q.wav"
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
wav_filename = "output_tts_clone_q.wav"
audio_data_int16 = np.int16(audio_data_float32 * (32767 / np.max(np.abs(audio_data_float32))))
scipy.io.wavfile.write(wav_filename , sample_rate, audio_data_int16)

"""
Comparing model Performance
"""
import time
import tqdm

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