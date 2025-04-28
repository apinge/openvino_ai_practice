from ov_outetts_helper import InterfaceOV, OVHFModel
import numpy as np
import time
model_dir = "OuteTTS-0.1-350M-ov"
interface = InterfaceOV(model_dir, "CPU")
start_time = time.time()
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