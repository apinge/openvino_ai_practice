# How to See the Pass Optimization of OV

```
set OV_ENABLE_VISUALIZE_TRACING=1
set OV_VISUALIZE_TREE_IO=1
set OV_VISUALIZE_TREE_OUTPUT_SHAPES=1
set OV_ENABLE_PROFILE_PASS=1 
```
compile_model即可有dot文件
sudo apt-get install graphviz
dot -Tpng input.dot -o output.png
可看
