# Utilizing LoRa with GenAI LLMPipeline

This is an example that shows the performance and memory testing of a OpenVINO.GenAI LLM pipeline integrated with LoRa.

## Build and Run

Lora safetensors can be download from https://hf-mirror.com/taopanda/test-lora-Qwen2.5-1.5B

```
<OpenVINO_GenAI_DIR>\setupvars.bat
cd genai_lora
mkdir build
.\build\Release\qwen-lora.exe  "C:\\Users\\yourname\\Qwen2.5-1.5B" "C:\\Users\\yourname\\test-lora-Qwen2.5-1.5B\\adapter_model.safetensors" "infer_with_lora_memory"
```