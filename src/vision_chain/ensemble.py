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
class TargetRegionSegmentation:
  """
  Design a pipeline to help use a segmentation model to define 
  what regions it's okay to keep predictions within. 
  
  For moving targets or moving cameras.
  """
  pass


@dataclass
class FeatureLevelEnsemble:
  feature_extractors: List[FeatureExtractor]
    
  def predict(file_path: str) -> Predictions:
    for feature_extractor in self.feature_extractors
      features = feature_extractor.run(file_path)
      ...
  
  
@dataclass 
class ProximityBasedPredictionFilter(Transformation):
  """
  Type of filter dedicated to determining the class of an object 
  based on the proximity of other objects. 
  
  e.g. only a Z if both X and Y present. 
  
  A if only 1 present within certain radius, 
  B if only 2 present within certain radius,
  C if both 1 and 2 present within certain radius
  """
  min_dist: float
  max_dist: float
  
  def get_pairwise_distances():
    pass
  
  def get_iou(): 
    pass 
  
  def run():
    pass
    

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
