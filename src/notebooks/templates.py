from dataclasses import dataclass
from PIL import Image
import supervision as sv

@dataclass
class Image: 
    """
    To prevent the system from having to open the 
    file multiple times. 

    Pass this object around, 
    """
    file_path: str
    image: PIL.Image = None

    def open(self): 
        """
        To prevent models in the same pipeline
        having to reopen the same file. 

        Wherever possible.
        """
        if self.image is not None: 
            return image

        else: 
            self.image = Image.open(file_path)
            return self.image

@dataclass
class Model: 
    path_to_weights: str

    def predict(self, Detections) -> Detections: 
        ...

@dataclass
class ObjectCropper:
    path_to_weights: str
    threshold: float

    """
    Just a model that's good at finding objects 
    something like SAM or GroundingDino.
    """
    def predict() -> Detections:
        ...


@dataclass
class ObjectClassifier:
    path_to_weights: str
    thresholds: dict

    def predict(image_path, preds: Detections) -> Detections:
        for pred in preds: 
           

@dataclass
class TwoStageDetector: 
    """
    Model consisting of one cropper and one 
    classifier.
    """
    object_cropper: ObjectCropper
    object_classifier: ObjectClassifier

    def predict(image_path: str) -> Detections:
        

@datclass 
class RegionOfInterestFilterer: 
    """
    Only count the detections within 
    this space 

    (use supervision to achieve that).

    Can be run before or after the Detector.
    """
    path_to_weights: str 
    threshold: float 

    def predict(image, preds: Detections) -> Detections: 
        mask = model(image)
        preds = 
        return preds

@dataclass
class NearestNeighbourDetector: 
    """
    Uses a VectorDatabase like QDrant to 
    classify the contents of bounding boxes. 

    This enables 0 training model improvement. 
    """
    cropper: Cropper
    backend: str = "QDrant"

    def predict(self, file_path: str):
        ...
   

@dataclass
class CompositeModel: 
    """
    A chain of different models.
    As long as each has the 
    ability to take in predictions,
    and an image, and outputs 
    predictions, they can be arbitrarily 
    chained together.
    """
    components: List[Model]
