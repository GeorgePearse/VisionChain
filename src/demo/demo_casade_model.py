from autodistill_grounded_sam import GroundedSAM
from super_gradients.training import models
from super_gradients.common.object_names import Models
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8

import super_gradients
import typer
from tqdm import tqdm
import os
from PIL import Image
import fiftyone as fo
from typing import List
from dataclasses import dataclass
from ultralytics import YOLO
import supervision as sv
from typing import Tuple



@dataclass
class Model:
    """
    Just a thing which predicts
    """

    def predict(self, file_path: str) -> sv.Detections:
        pass


@dataclass
class SpeculativePrediction:
    """
    Didnt want to forget the name
    https://arxiv.org/abs/2302.01318

    This is a request, and will only 
    be fulfilled if the CascadeModel
    contains the components to do so.

    Should be aiming for a matching system. 
    Where they all pass through everything
    """
    chain_start: bool 

    confident_detections: sv.Detections

    # hit if there's a downstream classifier
    reclassify_detections: sv.Detections

    # hit if there's a downstream Detector
    repredict_whole_frame: bool

    # for Mixture of Expert models
    # only really works if first model 
    # is a classifier
    # request_model: Model


@dataclass
class Detector:
    """
    Should better generalise this.
    """

    model_name: str = "fastest"

    def __post_init__(self):
        self.model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

    def predict(self, file_paths: List[str]) -> sv.Detections:
        # ultralytics format
        predictions = self.model(file_paths, stream=True)
        return sv.Detections.from_ultralytics(predictions)


@dataclass
class Condition:
    def evaluate(detections: sv.Detections) -> Tuple[sv.Detections, sv.Detections]:
        return (
            accepted,
            rejected,
        )

@dataclass
class UncertaintyRejection:
    confidence_trigger: float

    def evaluate(detections: sv.Detections) -> bool:
        filtered_detections = detections[
            detections.confidence > self.confidence_trigger
        ]
        if len(filtered_detections) == 0: 
            return False
        else: 
            return True


@dataclass
class ConditionalModel:
    model: Model

    def match(speculative_prediction: SpeculativePrediction) -> SpeculativePrediction:
        """
        When the model should be called.
        In terms of attributes of the speculative 
        prediction.
        """
        pass

    def speculate(speculative_prediction: SpeculativePrediction) -> SpeculativePrediction:
        """
        Run inference and flag anything that should be reconsidered
        by other models
        """
        pass



@dataclass
class CascadeModel(Model):
    conditional_models: List[ConditionalModel]

    def predict(self, file_path: str) -> sv.Detections:
        """
        Should the triggers be attached to the curr model, 
        or the next model? 

        Which creates the cleaner API
        """
        speculative_prediction = SpeculativePrediction(
            chain_start=True,
        )

        for conditional_model in self.conditional_models: 
            speculative_predictions = conditional_model.speculate(
                speculative_predictions
            )

        return speculative_predictions.confident_detections


 
@dataclass
class FastBase(ConditionalModel):
    model: Model

    def match(speculative_prediction: SpeculativePrediction) -> bool:
        return True

    def speculate(self, file_path: str, speculative_prediction: SpeculativePrediction) -> SpeculativePrediction:
        """
        Run inference, return speculative prediction
        """
        detections = self.model.predict(file_path)
        
        repredict_whole_frame=UncertaintyRejection(
            confidence_trigger=0.5
        ).evaluate()

        return SpeculativePrediction(
            confident_detections=detections,
            reclassify_detections=sv.Detections(), # empty detection
            repredict_whole_frame=repredict_whole_frame,
        )


def main(
    limit: int = None,
):
    """
    GroundedSAM to crop then DINO + QDrant to classify!
    """

    fast_base = Detector(model_name="yolo_nas_s")

    dir_path = "/home/ubuntu/VisionChain/src/demo/bottles_dataset/data"
    file_paths = [
        os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path)
    ]

    if limit is not None:
        file_paths = file_paths[:limit]

    model = CascadeModel([
        FastBase(model=fast_base),
        AccurateFallback(model=grounded_sam)
    ])
    
    dataset = get_predictions(dataset, model)

    session = fo.launch_app(dataset, remote=True, address="0.0.0.0", desktop=True)
    session.wait()


if __name__ == "__main__":
    typer.run(main)
