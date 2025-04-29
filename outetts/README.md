# OuteTTS 
The workflow is based on  https://github.com/openvinotoolkit/openvino_notebooks/tree/38896fbb662cdf4867db68ecf4262736b3705aca/notebooks/outetts-text-to-speech

```bash
pip install openvino openvino_tokenizers
pip install -r requirements.txt
optimum-cli export openvino --model OuteAI/OuteTTS-0.1-350M OuteTTS-0.1-350M-ov --task text-generation-with-past  
optimum-cli export openvino --model OuteAI/Llama-OuteTTS-1.0-1B Llama-OuteTTS-1.0-1B-ov --task text-generation-with-past  
optimum-cli export openvino --model OuteAI/OuteTTS-0.3-1B OuteTTS-0.3-1B-ov --task text-generation-with-past  
```

## Chinese Lora
https://huggingface.co/BricksDisplay/OuteTTS-0.3-500M-zh