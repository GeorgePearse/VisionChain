## Detectron2 Deployment

### ONNX Format

* Detectron2 supports Caffe2 ops, Caffe 2 is an old package no longer supported. 

Setup: 
```
pip install nvidia-pyindex
pip install onnx-graphsurgeon
```

An old version of PyTorch supported Caffe2 (this is the easiest way to install it)
https://github.com/facebookresearch/detectron2/issues/4398

This issue has the most detail, and is the most recent https://github.com/NVIDIA/TensorRT/issues/2546

Also potentially only works with MaskRCNN (not FasterRCNN) https://github.com/NVIDIA/TensorRT/issues/2862
