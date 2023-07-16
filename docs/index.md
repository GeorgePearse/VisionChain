## Vision Chain 

Constraining artificial stupidity. 

It's often surprisingly difficult to train out a stubborn false positive. These false positives can appear very stupid to clients. It can be a simple thing like an object of a given colour is NEVER some class, or that an object of a given size is NEVER another. These problems are simple and these mistakes should not be made, and yet neural networks are not good at learning such absolute, deterministic rules.

Good tooling to suggest sensible heuristics, to correct for these false positives does not exist, and this is one problem (among many others), that VisionChain aims to solve. 

This same style of heuristics are also extremely helpful for both labelling data and for online active-learning in resource constrained environments (other techniques include frame-frame jitter, and threshold based sampling).

Similarly there may be a very simple condition for when you should not predict, such as excessive blur, or insufficient exposure. Some business requirements demand that your product must always predict, but many others require high specificity and demand that when your model predicts, it must predict well (potentially due to a high cost automatic intervention). 

A neural network based object detector can be used to improve the rules (via analysis of its predictions), while the rules can be used to improve the neural network (by increasing the size of the dataset).

Most practitioners take inspiration from systems like Tesla's Data Engine, building ever larger dataset in order to teach Neural Networks simple rules, but most practical Machine Learning problems are not this open-ended. Most involve a set of fixed cameras in which object sizes are relatively consistent, and most deployments are aimed at solving a business application, which is not sufficiently addressed by just identifying objects, but instead is resolved by recognising an unexpected combination of items within a certain distance from each other, or the presence of one object in the absense of another etc.

Most applications are composite problems, to build a complete product you must both recognise a region of interest and then detect all objects within it, or flag the presence of one object in the absence of another etc.

The main downside of such approaches is the manual time to develop sensible rules, but with well-designed software, this need not be so. Rules can be suggested by analysis of a COCO dataset, and accepted or rejected by a developer. 

With the recent development of 'foundational models', there will be huge growth in 'training free' deployments, where a model (like grounding dino) is sufficiently accurate to deploy for a problem, but requires some guard rails specific to the dataset at hand. Models like GPT required LangChain, now models like GroundingDino need VisionChain. In an environment where GPU demand looks likely to continue to outstrip supply, such techniques will be needed to continue the democratization of Deep Learning applications. 

Below is a glimpse at the API: 

```
preprocessor = Preprocessor([
  NoPredictFilter(Blur(max_value=0.1)),
  NoPredictFilter(Exposure(max_value=0.7, min_value=0.2)),
])
```

```
postprocessor = Postprocessor([
    Thresholding(thresholds={'person': 0.5, 'car': 0.5, 'truck': 0.5, 'road': 0.5}),
    ClassAgnosticNMS(nms_threhold=0.8),
    ShapeFilter(min_width=400, min_height=400, class='car'),
    ColourFilter(central_colour='XXX', range='XXX', class='car'),
    OnlyPredictInsideRegionFilter(region_defining_classes=['road'])
])
```

```
model = Model(
    preprocessor=preprocessor,
    model_path='model.onnx',
    postprocessor=postprocessor,
    class_list=class_list,
)
```

The demo would be me using this tooling to apply foundation models to a non-coco dataset and business problem in real-time. 


