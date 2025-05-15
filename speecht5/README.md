# SpeechT5 TTS
https://huggingface.co/microsoft/speecht5_tts

```
optimum-cli export openvino --model microsoft/speecht5_tts speecht5_tts-ov --task text-to-audio-with-past --model-kwargs '{"vocoder": "microsoft/speecht5_hifigan"}'
```
OpenVINO version 2025.2 or higher is required.