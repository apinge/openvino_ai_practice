"""
Compiling models and inference
"""
from collections import namedtuple
import openvino as ov
from pathlib import Path
import torch.nn as nn
import torch
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf
import openvino_tokenizers

EncoderOutput = namedtuple("EncoderOutput", "last_hidden_state")
DecoderOutput = namedtuple("DecoderOutput", ("last_hidden_state", "past_key_values", "hidden_states", "attentions", "cross_attentions"))

core = ov.Core()

device = "CPU"

class TextEncoderModelWrapper(torch.nn.Module):
    def __init__(self, encoder_ir_path, config):
        ov_config = {}
        if "GPU" is device:
            ov_config = {"INFERENCE_PRECISION_HINT": "f32"}
        self.encoder = core.compile_model(encoder_ir_path, device, ov_config)
        self.config = config
        self.dtype = self.config.torch_dtype

    def __call__(self, input_ids, **_):
        last_hidden_state = self.encoder(input_ids)[0]
        return EncoderOutput(torch.from_numpy(last_hidden_state))


class DecoderWrapper(torch.nn.Module):
    def __init__(self, decoder_stage_1_ir_path, decoder_stage_2_ir_path, config):
        super().__init__()
        self.decoder_stage_1 = core.compile_model(decoder_stage_1_ir_path, "CPU")
        self.decoder_stage_2 = core.compile_model(decoder_stage_2_ir_path, "CPU")
        self.config = config
        self.embed_tokens = None
        embed_dim = config.vocab_size + 1  # + 1 for pad token id
        self.embed_tokens = nn.ModuleList([nn.Embedding(embed_dim, config.hidden_size) for _ in range(config.num_codebooks)])

    def __call__(self, input_ids=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, prompt_hidden_states=None, **kwargs):
        inputs = {}
        if input_ids is not None:
            inputs["input_ids"] = input_ids
        if encoder_hidden_states is not None:
            inputs["encoder_hidden_states"] = encoder_hidden_states
        if encoder_attention_mask is not None:
            inputs["encoder_attention_mask"] = encoder_attention_mask
        if prompt_hidden_states is not None:
            inputs["prompt_hidden_states"] = prompt_hidden_states
        if past_key_values is not None:
            past_key_values = tuple(past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer)
            inputs["past_key_values"] = past_key_values
            arguments = (
                input_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                *past_key_values,
            )
            outs = self.decoder_stage_2(arguments)
        else:
            outs = self.decoder_stage_1(inputs)

        outs = [torch.from_numpy(out) for out in outs.values()]
        past_key_values = list(list(outs[i : i + 4]) for i in range(1, len(outs), 4))

        return DecoderOutput(outs[0], past_key_values, None, None, None)

repo_id = "parler-tts/parler_tts_mini_v0.1"
model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id).to("cpu")

TEXT_ENCODER_OV_PATH = Path("models/text_encoder_ir.xml")
DECODER_STAGE_1_OV_PATH = Path("models/decoder_stage_1_ir.xml")
DECODER_STAGE_2_OV_PATH = Path("models/decoder_stage_2_ir.xml")
TOKENIZER_OV_PATH = Path("models/openvino_tokenizer.xml")

model.text_encoder = TextEncoderModelWrapper(TEXT_ENCODER_OV_PATH, model.text_encoder.config)
model.decoder.model.decoder = DecoderWrapper(DECODER_STAGE_1_OV_PATH, DECODER_STAGE_2_OV_PATH, model.decoder.model.decoder.config)
model._supports_cache_class = False
model._supports_static_cache = False
ov_tokenizer = core.compile_model(TOKENIZER_OV_PATH, "CPU")

prompt = "Hey there, you troublemaker! What kind of mischief are you up to today?"
description = "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. She speaks very fast."

input_ids = ov_tokenizer([description])['input_ids']
prompt_input_ids = ov_tokenizer([prompt])['input_ids']

generation = model.generate(input_ids=torch.tensor(input_ids), prompt_input_ids=torch.tensor(prompt_input_ids))
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out_ov.wav", audio_arr, model.config.sampling_rate)

print("OV Done!")
