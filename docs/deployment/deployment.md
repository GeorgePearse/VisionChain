# Deployment

This will become the place to document for any steps between training a model, and then 
deploying it, e.g. : 
* How to convert to a different format (ONNX, OpenVino, TensorRT)
* How to run thresholding. 

## Thresholding 

The target metric for the thresholding script is f1_score (harmonic mean of precision and recall).

::: deployment.thresholding.run_thresholding 

Voxel51 plan to implement some features that should simplify these calculations.


### Have you tried running inference with half precision?

Running inference with the Detectron R101 converted to half precision leads 
to this error on CPU or GPU.

```
Traceback of TorchScript, serialized code (most recent call last):
  File "code/__torch__/detectron2/export/flatten.py", line 34, in forward
    batched_imgs = torch.unsqueeze_(_8, 0)
    x0 = torch.contiguous(batched_imgs)
    _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, = (backbone).forward(x0, )
                                                             ~~~~~~~~~~~~~~~~~ <--- HERE
    _20 = (proposal_generator).forward(_9, _10, _11, _12, _13, _14, _15, _16, _17, _18, image_size, )
    _21 = (roi_heads).forward(_9, _20, _10, _11, _19, image_size, )
  File "code/__torch__/detectron2/modeling/backbone/fpn.py", line 28, in forward
    fpn_lateral5 = self.fpn_lateral5
    bottom_up = self.bottom_up
    _0, _1, _2, _3, _4, _5, _6, = (bottom_up).forward(x, )
                                   ~~~~~~~~~~~~~~~~~~ <--- HERE
    _7 = (fpn_lateral5).forward(_0, )
    _8 = (fpn_output5).forward(_7, )
  File "code/__torch__/detectron2/modeling/backbone/resnet.py", line 18, in forward
    res2 = self.res2
    stem = self.stem
    _0 = (res2).forward((stem).forward(x, ), )
                         ~~~~~~~~~~~~~ <--- HERE
    _1 = (res3).forward(_0, )
    _2 = (res4).forward(_1, )
  File "code/__torch__/detectron2/modeling/backbone/resnet.py", line 32, in forward
    x: Tensor) -> Tensor:
    conv1 = self.conv1
    input = torch.relu_((conv1).forward(x, ))
                         ~~~~~~~~~~~~~~ <--- HERE
    x0 = torch.max_pool2d(input, [3, 3], [2, 2], [1, 1], [1, 1])
    return x0
  File "code/__torch__/detectron2/layers/wrappers/___torch_mangle_7.py", line 12, in forward
    norm = self.norm
    weight = self.weight
    input = torch._convolution(x, weight, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1, False, False, True, True)
            ~~~~~~~~~~~~~~~~~~ <--- HERE
    return (norm).forward(input, )
```

Admittedly, I have seen this be caused by either the image, or the model just not being converted to half precision 
correctly, so it may still be worth a double check.

### Attempts to export to ONNX

Be warned, attempting this always ends up being a massive time sink (if there's not a clearly documented approach).
The problem is that there are too many things to try, to many unmerged PRs with techniques that may work, and once 
successfully converted the model still may not perform properly. Any warnings raised by ONNX must be taken seriously 
because they may have completely destroyed the model performance. 

The latest attempt with detectron2's export script with export_method of tracing (and export_format = ONNX) led to a
a serious mismatch of scores:

Torchscript:
```
tensor([0.9643, 0.9571, 0.8457, 0.8289, 0.8041, 0.7650, 0.7378, 0.5687, 0.5448,
         0.5272, 0.3921, 0.3801, 0.3388, 0.3177, 0.3082, 0.2940, 0.2813, 0.2745,
         0.2612, 0.2559, 0.2366, 0.2177, 0.2131, 0.1945, 0.1931, 0.1923, 0.1823,
         0.1790, 0.1627, 0.1561, 0.1525, 0.1292, 0.1282, 0.1233, 0.1228, 0.1194,
         0.1175, 0.1155, 0.1000, 0.0984, 0.0980, 0.0943, 0.0938, 0.0873, 0.0816,
         0.0816, 0.0805, 0.0792, 0.0781, 0.0757, 0.0737, 0.0716, 0.0701, 0.0663,
         0.0636, 0.0625, 0.0625, 0.0625, 0.0617, 0.0584, 0.0582, 0.0575, 0.0536,
         0.0520, 0.0516, 0.0513], device='cuda:0', grad_fn=<IndexBackward0>),
```

ONNX:
```
array([0.0617113 , 0.05936264, 0.05823653, 0.05821782, 0.05821177,
       0.05810352, 0.05809579, 0.05806343, 0.05805939, 0.05794789,
       0.05728726, 0.0565403 , 0.05582548, 0.05562487, 0.05555972,
       0.05546773, 0.05496673, 0.05464637, 0.05424192, 0.05401942,
       0.05399333, 0.05385649, 0.0535058 , 0.05347441, 0.05325294,
       0.05297298, 0.05283824, 0.05270571, 0.05270207, 0.05243513,
       0.05236924, 0.05228456, 0.05226028, 0.05224825, 0.05217861,
       0.05204139, 0.05197912, 0.05180624, 0.05176178, 0.05162115,
       0.05162048, 0.05156277, 0.05142196, 0.05136252, 0.05134707,
       0.05131438, 0.05127087, 0.05121848, 0.05121783, 0.05119659,
       0.05117048, 0.05115191, 0.051135  , 0.05109112, 0.05105976,
       0.05104537, 0.05095825, 0.05086421, 0.05075808, 0.05075512,
       0.05074402, 0.05072736, 0.05062436, 0.05059315, 0.0503956 ,
       0.05036141, 0.05034915, 0.05032273, 0.05010725, 0.05007025,
       0.05005644, 0.05005017, 0.05004704], dtype=float32)
```

Here are the warnings which appear most likely to explain the failure:
```
/home/ubuntu/environments/vision-research/lib/python3.8/site-packages/detectron2/modeling/roi_heads/fast_rcnn.py:155: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if num_bbox_reg_classes == 1:
/home/ubuntu/environments/vision-research/lib/python3.8/site-packages/detectron2/layers/nms.py:15: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert boxes.shape[-1] == 4
/home/ubuntu/environments/vision-research/lib/python3.8/site-packages/torch/onnx/_internal/jit_utils.py:258: UserWarning: The shape inference of prim::Constant type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function. (Triggered internally at ../torch/csrc/jit/passes/onnx/shape_type_inference.cpp:1884.)
  _C._jit_pass_onnx_node_shape_type_inference(node, params_dict, opset_version)
/home/ubuntu/environments/vision-research/lib/python3.8/site-packages/torch/onnx/symbolic_opset9.py:5408: UserWarning: Exporting aten::index operator of advanced indexing in opset 11 is achieved by combination of multiple ONNX operators, including Reshape, Transpose, Concat, and Gather. If indices include negative values, the exported graph will produce incorrect results.
  warnings.warn(
/home/ubuntu/environments/vision-research/lib/python3.8/site-packages/torchvision/ops/_register_onnx_ops.py:31: UserWarning: ROIAlign with aligned=True is not supported in ONNX, but will be supported in opset 16. The workaround is that the user need apply the patch https://github.com/microsoft/onnxruntime/pull/8564 and build ONNXRuntime from source.
```

The ROIAlign one seemed the most likely of these, I tried ONNX opset 16, but it did not resolve the problem. In fact the same 
warning remained. 

This approach appeared promising https://colab.research.google.com/drive/1ZFdkdIAjD0ldhJ9TEhzTL1bndLyqm2Rd?usp=sharing (convert with caffe2_tracing which supports all of the detectron2
operations). Then uses a script from TensorRT (exports to ONNX as an intermediate step). I tried to run it but hit CUDA version mismatches. There is not strict version control, and no testing of the output. So I'm not 
confident it would work. 

