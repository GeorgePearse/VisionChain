## MMDetection

### How to create and register Custom Checkpoints 

Normally best to keep things that change frequently in the train.py 
script. 

<https://mmdetection.readthedocs.io/en/v3.0.0rc0/advanced_guides/customize_runtime.html>

Taking the below directly from the docs because it tokk a while to find. 
```
@HOOKS.register_module()
class MyHook(Hook):

    def __init__(self, a, b):

    def before_run(self, runner) -> None:

    def after_run(self, runner) -> None:

    def before_train(self, runner) -> None:

    def after_train(self, runner) -> None:

    def before_train_epoch(self, runner) -> None:

    def after_train_epoch(self, runner) -> None:

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None) -> None:

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
```

To register a custom hook:
```
cfg.custom_hooks = [
  dict(type='AWSCloudCheckpoint', interval=checkpoint_and_eval_interval)
]
```

Any of the keys after 'type' in the custom hook are fed through to the 
custom hooks init function, e.g. for the above, 'interval' is provided 
to: 

```
def __init__(self, interval: int): 
  ...
```

Within the vision-research, the recommended approach to add a hook 
is to write it in utils/binit_vision_utils/mmdetection/...

e.g. utils/binit_vision_utils/mmdetection/cloud_checkpoint.py

### How to run Multi-GPU Training

MMDetection relies on torchrun (used to be called torch.distributed). It relies 
on each process being triggered with a different 'local_rank'. 
 
Think the underlying distributed technique is DistributedDataParallel, though 
not yet 100% sure on that. 

```
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash dist_train.sh faster_rcnn_resnet_101.py 4
```

Where dist_train.sh is this script:
```
#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}
```


