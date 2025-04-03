import os

os.environ["GIT_CLONE_PROTECTION_ACTIVE"] = "false"

"""
Load the original model and inference
"""
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cpu"

repo_id = "parler-tts/parler_tts_mini_v0.1"
model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(repo_id)

prompt = "Hey, how are you doing today?"
description = "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. She speaks very fast."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)

"""
Convert the model to OpenVINO IR
"""
print("===Convert the model to OpenVINO IR===")

import openvino as ov
from pathlib import Path


def convert(model: torch.nn.Module, xml_path: str, example_input):
    xml_path = Path(xml_path)
    if not xml_path.exists():
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            converted_model = ov.convert_model(model, example_input=example_input)

        ov.save_model(converted_model, xml_path)

        # cleanup memory
        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
        torch.jit._state._clear_class_state()

TEXT_ENCODER_OV_PATH = Path("models/text_encoder_ir.xml")
example_input = {
    "input_ids": torch.ones((1, 39), dtype=torch.int64),
}
text_encoder_ov_model = convert(model.text_encoder, TEXT_ENCODER_OV_PATH, example_input)

DECODER_STAGE_1_OV_PATH = Path("models/decoder_stage_1_ir.xml")


class DecoderStage1Wrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, input_ids=None, encoder_hidden_states=None, encoder_attention_mask=None, prompt_hidden_states=None):
        return self.decoder(
            input_ids=input_ids,
            return_dict=False,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            prompt_hidden_states=prompt_hidden_states,
        )


example_input = {
    "input_ids": torch.ones((9, 1), dtype=torch.int64),
    "encoder_hidden_states": torch.ones((1, 39, 1024), dtype=torch.float32),
    "encoder_attention_mask": torch.ones((1, 39), dtype=torch.int64),
    "prompt_hidden_states": torch.ones((1, 9, 1024), dtype=torch.float32),
}

decoder_1_ov_model = convert(DecoderStage1Wrapper(model.decoder.model.decoder), DECODER_STAGE_1_OV_PATH, example_input)

DECODER_STAGE_2_OV_PATH = Path("models/decoder_stage_2_ir.xml")


class DecoderStage2Wrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, input_ids=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None):
        past_key_values = tuple(tuple(past_key_values[i : i + 4]) for i in range(0, len(past_key_values), 4))
        return self.decoder(
            input_ids=input_ids,
            return_dict=False,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
        )


example_input = {
    "input_ids": torch.ones((9, 1), dtype=torch.int64),
    "encoder_hidden_states": torch.ones((1, 39, 1024), dtype=torch.float32),
    "encoder_attention_mask": torch.ones((1, 39), dtype=torch.int64),
    "past_key_values": (
        (
            torch.ones(1, 16, 10, 64, dtype=torch.float32),
            torch.ones(1, 16, 10, 64, dtype=torch.float32),
            torch.ones(1, 16, 39, 64, dtype=torch.float32),
            torch.ones(1, 16, 39, 64, dtype=torch.float32),
        )
        * 24
    ),
}

decoder_2_ov_model = convert(DecoderStage2Wrapper(model.decoder.model.decoder), DECODER_STAGE_2_OV_PATH, example_input)

