# Sample Integration of OpenVIINO.GenAI with .NET
## Download Model
We can try most model in [LLM IR from OpenVINO huggingface](https://huggingface.co/collections/OpenVINO/llm-6687aaa2abca3bbcec71a9bd)
```
git clone https://huggingface.co/OpenVINO/Phi-3.5-mini-instruct-int8-ov
```


## Live Demo
### Phi-3.5
![phi-demo](phi-demo.gif)


Example of [Phi-3.5-mini-instruct-int8-ov]( https://huggingface.co/OpenVINO/Phi-3.5-mini-instruct-int8-ov)
### DeepSeek-Distill
![deepseek-demo](deepseek-demo.gif)


Example of [DeepSeek-R1-Distill-Qwen-7B-int4-ov](https://huggingface.co/OpenVINO/DeepSeek-R1-Distill-Qwen-7B-int4-ov)


# Deployment

```
curl -O https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/pre-release/2025.1.0.0rc3/openvino_genai_windows_2025.1.0.0rc3_x86_64.zip
tar -xzvf openvino_genai_windows_2025.1.0.0rc3_x86_64.zip
xcopy /y openvino_genai_windows_2025.1.0.0rc3_x86_64\runtime\bin\intel64\Release\*.dll "C:\path\to\ChatSample\"
xcopy /y openvino_genai_windows_2025.1.0.0rc3_x86_64\runtime\3rdparty\tbb\bin\*.dll "C:\path\to\ChatSample\"
```