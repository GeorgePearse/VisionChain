## Trainer

## How to run training

::: detectron2.detectron2_train.main


Details about the training setup go here.

## How to view the impact of augmentations on the training set 

```
python detectron2_train.py --view-training-dataset
```

## Hooks and Tooling required for any training implementation

* Checkpoint to cloud 
* Workout how to increase MaxDets from 100 -> ~150.
* Aim experiment tracking (normally come with Tensorboard as default)
* Validation at specified iteration, not specified epoch (this can be much too long). Per class. 
* Some form of HParam Optimization 
* Augmentations
* Multi-GPU Training (working with all of the above)
* Export to ONNX (Detectron2 does not support this)
