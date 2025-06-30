import time
from transformers import AutoTokenizer
import torch, os,math
from torch.cuda.amp import autocast
from openvino.runtime import Core
from openvino.preprocess import PrePostProcessor
from openvino import Type
import openvino as ov
import torch, os

def main():
    model_path = "/home/gta/qiu/openvino_ai_practice/bce_reranker/bce-reranker-base_v1/openvino_model.xml"
    tokenizer_name = "maidalun1020/bce-reranker-base_v1"

    # Initialize OpenVINO Core
    core = Core()
    print("OpenVINO version: ", ov.get_version())
    # Load the model
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, "CPU")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Example input
    text = "This is a sample input text."
    inputs = tokenizer(text, return_tensors="pt")

    # Perform inference

    input_ids = inputs["input_ids"].numpy()
    attention_mask = inputs["attention_mask"].numpy()
    result = compiled_model([input_ids, attention_mask])

    print("Inference result:", result)

if __name__ == "__main__":
    main()