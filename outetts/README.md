# OuteTTS 
The workflow is based on  https://github.com/openvinotoolkit/openvino_notebooks/tree/38896fbb662cdf4867db68ecf4262736b3705aca/notebooks/outetts-text-to-speech

```bash
pip install -r requirements.txt
optimum-cli export openvino --model OuteAI/OuteTTS-0.1-350M OuteTTS-0.1-350M-ov --task text-generation-with-past  
```