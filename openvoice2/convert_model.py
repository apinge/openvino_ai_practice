import os
import torch
import openvino as ov
import nncf
from pathlib import Path

core = ov.Core()

from openvoice.api import BaseSpeakerTTS, ToneColorConverter, OpenVoiceBaseClass
import openvoice.se_extractor as se_extractor

CKPT_BASE_PATH = Path("./OpenVoiceV2")

en_au_suffix =  "base_speakers/ses/en-au.pth"
en_br_suffix =  "base_speakers/ses/en-br.pth"
en_default_suffix =  "base_speakers/ses/en-default.pth"
en_india_suffix =  "base_speakers/ses/en-india.pth"
en_newest_suffix =  "base_speakers/ses/en-newest.pth"
en_us_suffix = "base_speakers/ses/en-us.pth"
zh_suffix =  "base_speakers/ses/zh.pth"

converter_suffix = CKPT_BASE_PATH / "converter"


IRS_PATH = Path("openvino_irs/")
VOICE_CONVERTER_IR = IRS_PATH / "openvoice_tone_conversion.xml"

class OVOpenVoiceBase(torch.nn.Module):
    """
    Base class for both TTS and voice tone conversion model: constructor is same for both of them.
    """

    def __init__(self, voice_model: OpenVoiceBaseClass):
        super().__init__()
        self.voice_model = voice_model
        for par in voice_model.model.parameters():
            par.requires_grad = False


class OVOpenVoiceTTS(OVOpenVoiceBase):
    """
    Constructor of this class accepts BaseSpeakerTTS object for speech generation and wraps it's 'infer' method with forward.
    """

    def get_example_input(self):
        stn_tst = self.voice_model.get_text("this is original text", self.voice_model.hps, False)
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        speaker_id = torch.LongTensor([1])
        noise_scale = torch.tensor(0.667)
        length_scale = torch.tensor(1.0)
        noise_scale_w = torch.tensor(0.6)
        return (
            x_tst,
            x_tst_lengths,
            speaker_id,
            noise_scale,
            length_scale,
            noise_scale_w,
        )

    def forward(self, x, x_lengths, sid, noise_scale, length_scale, noise_scale_w):
        return self.voice_model.model.infer(x, x_lengths, sid, noise_scale, length_scale, noise_scale_w)


class OVOpenVoiceConverter(OVOpenVoiceBase):
    """
    Constructor of this class accepts ToneColorConverter object for voice tone conversion and wraps it's 'voice_conversion' method with forward.
    """

    def get_example_input(self):
        y = torch.randn([1, 513, 238], dtype=torch.float32)
        y_lengths = torch.LongTensor([y.size(-1)])
        target_se = torch.randn(*(1, 256, 1))
        source_se = torch.randn(*(1, 256, 1))
        tau = torch.tensor(0.3)
        return (y, y_lengths, source_se, target_se, tau)

    def forward(self, y, y_lengths, sid_src, sid_tgt, tau):
        return self.voice_model.model.voice_conversion(y, y_lengths, sid_src, sid_tgt, tau)
pt_device = "cpu"
tone_color_converter = ToneColorConverter(converter_suffix / "config.json", device=pt_device)
tone_color_converter.load_ckpt(converter_suffix / "checkpoint.pth")
paths = [VOICE_CONVERTER_IR]
models = [
    OVOpenVoiceConverter(tone_color_converter),
]
for model, path in zip(models, paths):
    if not path.exists():
        ov_model = ov.convert_model(model, example_input=model.get_example_input())
        ov.save_model(ov_model, path) # save fp16
        ov_quantized_model = nncf.compress_weights(ov_model)
        ov.save_model(ov_quantized_model, IRS_PATH / "openvoice_tone_conversion_quantized.xml")
    else:
        ov_model = core.read_model(path)
    #ov_models.append(ov_model)
print("ok")