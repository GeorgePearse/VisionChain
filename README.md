# VisionChain
Framework to support common preprocessing and postprocessing steps, along with computer vision heuristics built with opencv, and voting systems ontop of those heuristic-model combos. 

https://hitla-ml.github.io/

Constraining stupidity and giving AI some hand rails.

## Tools it would lean on: 

* sklearn.ensemble
* supervision
* dataclasses
* jupyter-scatter ? 
* pandas
* pyodi (for a few key functions) 
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
* Maybe raft / optical flow stuff (probably not). 
* Edge detectors etc. in kornia, the list in 'feature' could almost all be included https://kornia.readthedocs.io/en/latest/feature.html

## Features 
- [ ] ROI detectors, e.g. an object detector that outpus an area, used as a filter for another detector.
- [ ] Class balanced dataset split.
- [ ] Tooling for combining SAM HQ with object detector
- [ ] Tooling for combining Grounding DINO and SAM, or Grounding DINO and a custom model (Labelling Pipeline).
- [ ] Tooling for analysing dataset of labels to come up with heuristics (colour and size, width / height).
- [ ] Update labels with replacement, or merge.
- [ ] Functionality to increase sensitivity depending on context, e.g. cluster detection. 
- [ ] SAHI
- [ ] Normalization code (e.g. get normalization values for a dataset). Run some more experiments on this on different datasets and write them up.
- [ ] Package in some simple labelling functionlaity, e.g. 3 of Y, and fix labels.
- [ ] Query based Active Learning, an object detection model that didn't need to be fed coco annotations, but could easily be provided a mix of a COCO dataset and then start asking crop questions from a dataset where it was uncertain. Then use simple copy paste, to place those queries on realistic backgrounds.
- [ ] Sort out the batching to combine a Classifier with an Object Detector efficiently.
- [ ] Function to convert mask to bounding box.
- [ ] Functionality to resize objects to within range (e.g. find smallest and largest true object, and aim to augment to within this scale, would work particularly well with CopyPaste)
- [ ] Auto tuning of heuristics e.g. find a sensible confined range for the width and height of an object, or colour range / set.  + pyodi like visualization, return the code
- [ ] Profiling of speed of each step of the postprocessing.
- [ ] Ability to analyse how each filter impacts the performance, and evaluate different combinations.
- [ ] Support online active learning via heuristic - model disagreement

* Enables you to count objects in a region, even if that region, or the camera, is moving.

```
@dataclass 
class PostprocessingHook: 

    def run(self, predictions: Predictions) -> Predictions:
        pass

```

```
@dataclass
class Thresholding(PostprocessingHook): 
    thresholds: dict 

    def run(self, predictions: Predictions) -> Predictions: 
        """
        Do stuff
        """
        return thresholded_predictions
```

```
Postprocessor([
    Thresholding(thresholds={}),
    ClassAgnosticNMS(nms_threhold=0.8),
    ClassOrderedNMS(preferential_class_list=['person', 'car', 'truck'], iou_threshold), 
    ShapeFilter(width=400, height=400, class='car'),
    ColourFilter(central_colour='XXX', range='XXX'),
    OnlyInsideRegionFilter(region_defining_classes=['conveyor_belt']),
    OnlyOutsideRegionFilter(region_defining_classes=['X']),
    IgnorePredsWhen(ignore_classes=['image_static', 'extreme_blur']),
    BinaryClassificationFixer(suspect_class='truck', model=Model(
        preprocessor=Preprocessor()
        model_path='truck_vs_car.onnx',
        postprocessor=Postprocessor([Thresholding(thresholds={'car': 0.5, 'truck': 0.5}]),
    FallBackModel(trigger=Trigger(), model=Model(model_path='better_model.onnx'))),
])
```


```
object_detector = Model(
    preprocessor=Preprocessor(),
    model_path='model.onnx',
    postprocessor=postprocessor,
) 
```

- [ ] Link this all with a research page / blog: 
      

Check calmcode tutorials.

Would need to actually integrate models as they came. 

Articles I need to get out 
- [ ] Deploying an MMDetection model with Triton (they'd probably also want this in the docs).
- [ ] About the value of heuristics -> there is always some stage in the data-centric loop, where you benefit from heuristics (e.g. very small training dataset) 
