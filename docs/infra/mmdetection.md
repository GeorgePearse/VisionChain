## MMDetection

MMDetection is what I see us using for model training long term. The repo is still under much more active development (than detectron2)
and the contributors are more responsive to questions. They have integrated pytorch 2.0 functionality (torch.compile())

It also has more tooling to help with deployments (see mmdeploy). Many of the models support export to ONNX, 
TensorRT and OpenVino, and the models that do support this functionality are neatly listed here 
<https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/04-supported-codebases/mmdet.md>

Supports training direct from S3, which greatly simplifies the rest of the automated training setup. 
<https://github.com/open-mmlab/mmcv/pull/1545>

It also supports more 'modern' augmentations e.g. Mosaic and MixUp, along with the frameworks to properly 
incorporate them into training, e.g. 2 stage pipelines for aggressive augmentations, then weaker ones. 

The ecosytem is also more rich with a package dedicated to self-supervised learning, 3D Computer Vision and Object Tracking 
<https://mmtracking.readthedocs.io/en/latest/tutorials/config_mot.html>

Make sure you use the latest docs at -> https://mmdetection.readthedocs.io/en/latest/get_started.html

MMDetection is at a slightly weird step in in a major migration. The most likely problem you'll hit is installation of mmdet 2.X 
instead of 3.x. It will refer to registering a Visualiser https://github.com/open-mmlab/mmdetection/issues/9914.

Remedy this with: 
```
mim install mmdet>'3.0.0rc0'
```

Also discussed at 

### Downsides 
* Seems to have higher memory useage than Detectron2, can normally only achieve half the batch size
* Multi-gpu training is less simple to setup 

### How to setup multi gpu training 

MMDet multi-gpu training relies on TorchRun https://pytorch.org/docs/stable/elastic/run.html

nproc_per_nod defines the number of workers on each node. It should equal to the number of GPUs on each node. In this case, 2
https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide


### Where to find docs on Augmentations

For a lot of MMDetection problems you actually need to refer to MMCV 
https://mmcv.readthedocs.io/en/latest/

https://mmcv.readthedocs.io/en/latest/understand_mmcv/data_transform.html#design-of-data-transformation

Some common hooks come from MMCV (e.g. checkpointing and validation)
<https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py>

Below is how you would construct and call a  flip which supported vertical flipping (not yet sure if this 
is supported as default or not).
```
import random
import mmcv
from mmcv.transforms import BaseTransform, TRANSFORMS

@TRANSFORMS.register_module()
class MyFlip(BaseTransform):
    def __init__(self, direction: str):
        super().__init__()
        self.direction = direction

    def transform(self, results: dict) -> dict:
        img = results['img']
        results['img'] = mmcv.imflip(img, direction=self.direction)
        return results
```
And to call in a pipeline:

```
pipeline = [
    ...
    dict(type='MyFlip', direction='horizontal'),
    ...
]
```

Think true implementation of vertical flip is here <https://github.com/open-mmlab/mmcv/blob/main/mmcv/transforms/processing.py>

### How to view the post-augmentation Dataset

This script, runs augmentations and writes them to disk.
<https://github.com/open-mmlab/mmdetection/blob/main/tools/analysis_tools/browse_dataset.py>

Will simplify and move it into vision-research. Detectron2 has an equivalent. 

### What are dynamic intervals? 

<https://mmdetection.readthedocs.io/en/v3.0.0rc0/advanced_guides/customize_runtime.html>

# Before 365001th iteration, we do evaluation every 5000 iterations.
# After 365000th iteration, we do evaluation every 368750 iteraions,
# which means that we do evaluation at the end of training.
```
interval = 5000
max_iters = 368750
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=interval,
    dynamic_intervals=dynamic_intervals)
```

Think it's when you want to start off running eval rarely, but finish running it frequently.

### How to export to torchscript (currently used in model thresholding and eval)

```
python tools/deployment/pytorch2torchscript.py \
    ${CONFIG_FILE} \
    --checkpoint ${CHECKPOINT_FILE} \
    --output-file ${OUTPUT_FILE} \
    --shape ${IMAGE_SHAPE} \
    --verify \
```
where shape is the height and width of input tensor to the model e.g. 1920, 1080
