## ML Frameworks 

So far the frameworks used have gone pytorch-lightning -> detectron2 -> mmdetection. 

Here are some of the reasons for those choices in case anyone gets tempted to go back 
(which may be justified)

### Pytorch Lightning 

* Excellent hooks provided out of the box (cloud checkpointing, experiment tracker compatibility)
* The default torchvision model zoo is tiny, and does not support many deployment formats (ONNX, tensorRT etc.)
* Multi-GPU training not ALWAYS reliable (though this was more a result of our use of FiftyoneDataset)
* Slight pain to get FasterRCNN training on background images out of the box. 
* Could potentially be good to train MMDetection models with Pytorch-Lightning Trainer, but this sort of crossover system is often not simple.

### Detectron2 

* Very poorly maintained (docs could be much better and responses to issues are passive aggressive) 
* Fast training.
* Simple hook system. 
* Almost no useful default hooks.
* Much more detailed loss logging than our initial pytorch-lightning setup (to achieve this would have required 
editting of the loss in torchvision's implementation of the faster_rcnn_loss because it did not track this level of detail). 
* No 'real-time' models, pretty much all of the implementations are built on a FasterRCNN backbone which limits things.

### MMDetection

* Much better support for deployment (ONNX, TensorRT, OpenVINO)
* Extensive model zoo. 
* More advanced augmentations (Mosaic and Mixup)
* Very few hooks inplemented out of the box.
* Part of a larger ecoysystem which includes MMSelfSup and MMTracking. 
* Semi-supervised training supported out of the box. 
* Think there's Model Distillation implemented somewhere.

Should implement something like this <https://mmcv-jm.readthedocs.io/en/latest/_modules/mmcv/fileio/file_client.html>
for different model types.
