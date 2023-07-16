## Ensembles 

Ensembles for object detection are rarely deployed in production, however it can be very useful to use them when labelling. 

### 
```python
model = Ensemble(
  models=[
    Model(),
    Model()
  ],
  aggregation=[],
)
```

### Class List Ensemble 

This setup enables you to share the preprocessing actions between each of the models in the ensemble. 
But also to cleanly combine the outputs of multiple models.

```python
model = Ensemble(
  preprocessor=Preprocessor(...),
  models=[
    Model(
      model_path='model_one.onnx',
      postprocessor=postprocessor(...),
      class_list=['truck','car'],
    ),
    Model(
      model_path='model_two.onnx',
      postprocessor=postprocessor(...),
      class_list=['person'],
    )
  ],
  aggregation=NMS(...),
  postprocessor=Postprocessor([
    
  ]),
)
