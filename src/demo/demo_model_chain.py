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
from dataclasses import dataclass, field
from ultralytics import YOLO, RTDETR, NAS
import supervision as sv
from typing import Tuple, Dict, List
from get_predictions import get_predictions
from rich import print
import time


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

    chain_start: bool = False

    confident_detections: sv.Detections = None

    # hit if there's a downstream classifier
    reclassify_detections: sv.Detections = None

    # hit if there's a downstream Detector
    repredict_whole_frame: bool = False

    # just whether the last model ran or not
    triggered: bool = None


@dataclass
class UltralyticsDetector(Model):
    """
    Should better generalise this.
    """
    model_family: str
    model_weights: str 

    def __post_init__(self):
        model_family_from_string = {
            'RTDETR': RTDETR,
            'YOLO': YOLO,
            'NAS': NAS,
        }
        yolo_models = [
            'yolov8n.pt',
            'yolov8s.pt',
            'yolov8m.pt',
            'yolov8l.pt',
            'yolov8x.pt',
        ]
        rtdetr_models = [
            'rtdetr-l.pt',
            'rtdetr-x.pt',
        ]
        nas_models = [
            'yolo_nas_s.pt',
            'yolo_nas_m.pt',
            'yolo_nas_l.pt'
        ]

        supported_models = yolo_models + rtdetr_models + nas_models

        assert self.model_weights in supported_models, (
            f'Requested model {self.model_weights} not supported'
        )
        self.model = model_family_from_string[
            self.model_family
        ](self.model_weights)

    def predict(self, file_paths: List[str]) -> sv.Detections:
        # ultralytics format
        # Use to have stream=False
        # but changed after code complained about
        # a generator
        predictions = self.model(file_paths, stream=False)
        detections = sv.Detections.from_ultralytics(predictions[0])
        print("SUCCESSFUL CONVERSION")
        return detections


@dataclass
class Condition:
    def evaluate(detections: sv.Detections) -> bool:
        pass


@dataclass
class UncertaintyRejection(Condition):
    confidence_trigger: float

    def evaluate(self, detections: sv.Detections) -> bool:
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
    name: str
    condition_triggered: bool = field(default=False, init=False)

    def match(
        self, speculative_prediction: SpeculativePrediction
    ) -> SpeculativePrediction:
        """
        When the model should be called.
        In terms of attributes of the speculative
        prediction.
        """
        pass

    def speculate(
        speculative_prediction: SpeculativePrediction,
    ) -> SpeculativePrediction:
        """
        Run inference and flag anything that should be reconsidered
        by other models
        """
        pass


@dataclass
class ModelChain(Model):
    conditional_models: List[ConditionalModel]
    log_level: str

    def predict(self, file_path: str) -> sv.Detections:
        """
        Should the triggers be attached to the curr model,
        or the next model?

        Which creates the cleaner API
        """
        speculative_predictions = SpeculativePrediction(
            chain_start=True,
        )

        for conditional_model in self.conditional_models:
            start = time.time()
            speculative_predictions = conditional_model.speculate(
                file_path,
                speculative_predictions,
            )
            end = time.time()
            elapsed = end - start
            if self.log_level == "verbose":
                detail = {
                    'model_name': conditional_model.name,
                    'condition_triggered': conditional_model.condition_triggered,
                    'time_taken': elapsed,
                }
                print(detail)
                print(
                    f"Preds with {conditional_model.name} took {elapsed}. Condition triggered: {conditional_model.condition_triggered}"
                )

        return speculative_predictions.confident_detections


@dataclass
class GroundedSamDetector:
    ontology: Dict[str, str]

    def __post_init__(self):
        self.model = GroundedSAM(ontology=CaptionOntology(self.ontology))

    def predict(self, file_path: str) -> sv.Detections:
        return self.model.predict(file_path)


@dataclass
class FastBase(ConditionalModel):
    def match(self, speculative_prediction: SpeculativePrediction) -> bool:
        self.condition_triggered = True

    def speculate(
        self, file_path: str, speculative_prediction: SpeculativePrediction
    ) -> SpeculativePrediction:
        """
        Run inference, return speculative prediction
        """
        detections = self.model.predict(file_path)

        # bit of a fake / meaningless line
        self.match(detections)

        print("Got preds from yolo")

        repredict_whole_frame = UncertaintyRejection(
            confidence_trigger=0.1,
        ).evaluate(detections)

        print("got reprediction result")

        return SpeculativePrediction(
            confident_detections=detections,
            reclassify_detections=None,  # empty detection
            repredict_whole_frame=repredict_whole_frame,
        )


@dataclass
class AccurateFallback(ConditionalModel):
    def match(self, speculative_prediction: SpeculativePrediction) -> bool:
        if speculative_prediction.repredict_whole_frame:
            self.condition_triggered = True

    def speculate(
        self, file_path: str, speculative_prediction: SpeculativePrediction
    ) -> SpeculativePrediction:
        """ """
        self.match(speculative_prediction)

        if self.condition_triggered:
            detections = self.model.predict(file_path)

            return SpeculativePrediction(
                confident_detections=detections,
                reclassify_detections=None,
                repredict_whole_frame=False,
            )
        else:
            return speculative_prediction


def main(
    limit: int = 100,
):
    """
    GroundedSAM to crop then DINO + QDrant to classify!
    """

    fast_base = UltralyticsDetector(
        model_family='YOLO',
        model_weights='yolov8n.pt',
        #model_family='NAS', # didn't work
        #model_weights='yolo_nas_s.pt', # didn't work
        #model_family='RTDETR', # worked 
        #model_weights='rtdetr-l.pt', #worked
    )
    grounded_sam = GroundedSamDetector(
        ontology={
            "plastic bottle": "bottle",
            "glass bottle": "bottle",
            "wine bottle": "bottle",
            "person": "person",
            "grapes": "fruit",
            "banana": "fruit",
            "bird": "bird",
            "glove": "glove",
            "basket": "basket",
            "trophy": "trophy",
            "traffic lights": "traffic lights",
            "orange": "orange",
            "lemon": "lemon",
            "cow": "cow",
            "laptop": "laptop",
            "dog": "dog",
        }
    )

    dir_path = "/home/ubuntu/VisionChain/src/demo/bottles_dataset/data"
    file_paths = [
        os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path)
    ]

    if limit:
        file_paths = file_paths[:limit]

    # Could also call this a model chain???
    model = ModelChain(
        [
            FastBase(model=fast_base, name="fast_base"),
            AccurateFallback(model=grounded_sam, name="grounded_sam"),
        ],
        log_level="verbose",
    )

    dataset = fo.Dataset.from_images(file_paths)
    dataset = get_predictions(dataset, model)

    session = fo.launch_app(dataset, remote=True, address="0.0.0.0", desktop=True)
    session.wait()


if __name__ == "__main__":
    typer.run(main)
