# VisionChain
Framework to support common preprocessing and postprocessing steps, along with computer vision heuristics built with opencv, and voting systems ontop of those heuristic-model combos. 

## Tools it would lean on: 

* sklearn.ensemble
* supervision
* dataclasses
* jupyter-scatter ? 
* pandas 
* CocoFrame (pending completion)
* Norfair or an object tracking framework
* human-learn https://koaning.github.io/human-learn/ with UMAP?  -> size based filters + colour based filters. 
* Maybe something like hamilton for clean pandas transformations.
* Definitely hugging-face.
* Top X style queries for Grounding Dino or Sam. e.g. state 3 oranges and retain the top 3 orange predictions.
* QDrant positive - negative 
* Quaterion for similarity learning of the classifier in a CompositeDetector.
* Multiple off-the-shelf-models combined (langchain style, gives the name).
* Simple way to train detector on the feature input of several different architectures
* May raft / optical flow stuff. 
* Edge detectors etc. in kornia, the list in 'feature' could almost all be included https://kornia.readthedocs.io/en/latest/feature.html

## Features 
- [ ] ROI detectors, e.g. an object detector that outpus an area, used as a filter for another detector.
- [ ] Class balanced dataset split.
- [ ] Tooling for combining SAM HQ with object detector
- [ ] Tooling for combining Grounding DINO and SAM, or Grounding DINO and a custom model (Labelling Pipeline).
- [ ] Tooling for analysing dataset of labels to come up with heuristics (colour and size, width / height). 

* Enables you to count objects in a region, even if that region, or the camera, is moving. 
```
object_detector = Model(
    preprocessor=Preprocessor(),
    model_path='model.onnx',
    postprocessor=Postprocessor(
      class_list=class_list,
      thresholds=thresholds,
      roi_filter_class = 'conveyor_belt', # or could be conveyor belt, or could be road etc.
    )
) 
```

Check calmcode tutorials.

Would need to actually integrate models as they came. 
