# BCE Reranker

## Convert ReRanker 
```bash


optimum-cli export openvino --model maidalun1020/bce-reranker-base_v1  --trust-remote-code  --task text-classification --weight-format fp16 bce-reranker-base_v1-ov

```

## Convert Embedding

```bash


optimum-cli export openvino --model maidalun1020/bce-embedding-base_v1  --trust-remote-code  --task feature-extraction --weight-format fp16 bce-embedding-base_v1-ov --library sentence_transformers
 
```