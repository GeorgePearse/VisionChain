from dataclasses import dataclass
from model import Model
from resolution import Resolution
from predictions import Predictions
from typing import List
from transformation import Transformation

# TODO: 
# Integrate https://machinelearningmastery.com/voting-ensembles-with-python/
# sklearn with IoU for object detection

# NB this is all complete pseudo code, just mapping out thev value the 
# package could deliver


@dataclass
class Ensemble:
  resolution: Callable
  models: List[Model]
    
  def predict(file_path: str) -> Predictions: 
    predictions = []
    
    for model in self.models:
      prediction = model.predict(file_path)
      predictions.append(prediction)
      
    self.resolution(predictions)
    
    
@dataclass
class CompositeDetector:
  """
  Main valud of this function would be appropriately batching 
  the inference of a 'cropper' - object detection model.
  
  And feeding those crops to a classifier.

  Potentially particularly valuable in the context of models 
  like GroundingDino and SAM
  """
  cropper: ObjectDetectionModel
  classifier: ClassificationModel
    
  def predict(file_path: str) -> Predictions: 
    crops = self.cropper(file_path)
    predictions = []
    for crop in crops:
      prediction = self.classifier(crop)
      predictions.append(prediction)
      
     return Predictions(predictions)
  
  
@dataclass
class Heuristic: 
  """
  Not sure how this would be / is truly different 
  from a model.
  """
  
  def predict(file_path: str) -> Predictions:
    pass
  
  
@dataclass
class Transformation:
  
  # use beartype to validate input and output
  @beartype
  def run(predictions: Predictions) -> Predictions:
    ...
  
  
@dataclass
class Postprocess: 
  """
  Can have a few traits
  Filters. 
  """
  transformations: List[Transformation[
    
  def run(predictions: Predictions) -> Predictions:
    for transformation in transformations:
      predictions = transformation.run(predictions)
    
    return predictions
