# Parler-TTS

The workflow is based on https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/parler-tts-text-to-speech/parler-tts-text-to-speech.ipynb


```
conda create -m tts_env python=3.10
conda activate tts_env
pip install openvino
pip install  git+https://github.com/huggingface/parler-tts.git "gradio>=4.19" transformers "torch>=2.2" "torchaudio" --extra-index-url https://download.pytorch.org/whl/cpu
```
OR

```
conda create -m tts_env python=3.10
conda activate tts_env
pip install -r requirements.txt
```