# VisionChain
Framework to support common preprocessing and postprocessing steps, along with computer vision heuristics built with opencv, and voting systems ontop of those heuristic-model combos. 

Could add a hyperparameter optimizer for things like setting the confidence at which to intervene.

Constraining artificial stupidity and giving AI some hand rails.

To Do: 
- [ ] Create some templates, e.g. cascade model
- [ ] Support VQA models on crops or similar
- [ ] Support NN Classifiers based on the embeddings of multiple models.
- [ ] Support Segmentation models
- [ ] Support more from https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main
- [ ] Integrate TaskMatrix a bit https://github.com/microsoft/TaskMatrix


```python
model = vc.ModelChain(
    [
        vc.ConditionalDetector(
            model=yolov8,
        ),
        vc.ConditionalDetector(
            model=grounded_sam, 
            frame_level_condition = lambda predictions: any([score < 0.5 for score in predictions.scores]),
            prediction_level_condition = lambda pred: 'cat' in pred.label,
        ),
        vc.ConditionalClassifier(
            model=nn_classifier,
            prediction_level_condition = lambda pred: 'dog' in pred.label,
        ),
    ],
    log_level = 'verbose',
)
```
