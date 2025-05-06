# OpenVoice2
The workflow is based on  https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/openvoice
```
git clone https://github.com/myshell-ai/OpenVoice.git
cp *.py OpenVoice
cd OpenVoice
python setup.py develop
git clone https://huggingface.co/myshell-ai/OpenVoiceV2
python convert_model.py
```