import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Union
from abc import ABC, abstractmethod

import fiftyone as fo
import pandas as pd
import super_gradients
import supervision as sv
import torch
import typer
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
from autodistill_yolov8 import YOLOv8
from PIL import Image
from qdrant_client import QdrantClient, models
from rich import print
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel
from ultralytics import NAS, RTDETR, YOLO
from get_predictions import get_predictions

@dataclass
class Prediction:
    score: float
    box: List[float]
    label: str


@dataclass
class Predictions:
    """
    Predictions for one image.

    Arguments
        scores: e.g. [0.4] floats 0 -> 1.
        boxes: [[x1, y1, x2, y2]], where all are absolute values (VOC format) e.g 0 < x < 1920.
        labels: e.g. string representation to keep you sane!
    """

    scores: List[float]
    boxes: List[List[float]]
    labels: List[str]

    def __post_init__(self):
        """
        If no coordinates are > 1, it's almost guaranteed
        that the box format is wrong.

        Can add further sanity checks.

        e.g. if x2 < x1 or same for y
        """
        # set to 1 for [[]] e.g. for no boxes, len(boxes) == 1
        if len(self.boxes) > 1:
            suspect_boxes_normalized = True
            for box in self.boxes:
                for coord in box:
                    if coord > 1:
                        suspect_boxes_normalized = False

            if suspect_boxes_normalized:
                print("WARNING: looks like boxes are normalized")

            if len(self.boxes[0]) != 4:
                raise Exception(
                    f"""
                    Input boxes are not of length 4.
                    Example box length is {len(self.boxes[0])}
                """
                )

    def __len__(self) -> int:
        return len(self.labels)

    def with_nms(self, class_agnostic: bool):
        """
        TODO: Convert this into generalised 'supervision'
        wrapper
        """
        label_to_id = {class_name: id for class_name in list(set(self.labels))}
        id_to_label = {v: k for k,v in label_to_id.items()}

        detections = sv.Detections(
            xyxy=np.array(self.boxes),
            labels=[label_to_id[label] for label in self.labels],
            scores=np.array(self.scores),
        ).with_nms(class_agnostic = class_agnostic)

        return Predictions(
            boxes=detections.xyxy.tolist(),
            labels=[id_to_label[class_id] for class_id in detections],
            scores=detections.confidence.tolist(),
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "labels": self.labels,
                "boxes": self.boxes,
                "scores": self.scores,
            }
        )

    @staticmethod
    def from_dataframe(df: pd.DataFrame):
        if len(df) == 0: 
            return Predictions(
                labels=[],
                boxes=[[]],
                scores=[],
            )

        else:
            return Predictions(
                labels=df["labels"].tolist(),
                boxes=df["boxes"].tolist(),
                scores=df["scores"].tolist(),
            )

    @staticmethod
    def from_list_of_preds(predictions: List[Prediction]):
        return Predictions(
            labels=[pred.label for pred in predictions],
            boxes=[pred.box for pred in predictions],
            scores=[pred.score for pred in predictions],
        )

    def to_list_of_preds(self) -> List[Prediction]:
        return [
            Prediction(
                label=self.labels[idx],
                box=self.boxes[idx],
                score=self.scores[idx],
            )
            for idx, _ in enumerate(self.labels)
        ]

    def from_supervision(detections: sv.Detections, class_list: List[str]):
        labels = [class_list[x] for x in detections.class_id.tolist()]
        return Predictions(
            boxes=detections.xyxy.tolist(),
            labels=labels,
            scores=detections.confidence.tolist(),
        )
    

empty_predictions = Predictions(
    scores = [],
    boxes = [[]],
    labels = [],
)

@dataclass
class ClassificationPrediction:
    label: str
    score: float


class Model(ABC):
    """
    Just a thing which predicts
    """

    @abstractmethod
    def predict(self, file_path: str) -> Predictions:
        raise Exception("Prediction method not implemented")


@dataclass
class UltralyticsDetector(Model):
    """
    Should better generalise this.
    """

    model_family: str
    model_weights: str
    name: str

    def __post_init__(self):
        model_family_from_string = {
            "RTDETR": RTDETR,
            "YOLO": YOLO,
            "NAS": NAS,
        }
        yolo_models = [
            "yolov8n.pt",
            "yolov8s.pt",
            "yolov8m.pt",
            "yolov8l.pt",
            "yolov8x.pt",
        ]
        rtdetr_models = [
            "rtdetr-l.pt",
            "rtdetr-x.pt",
        ]
        nas_models = ["yolo_nas_s.pt", "yolo_nas_m.pt", "yolo_nas_l.pt"]

        assert self.model_weights not in nas_models, (
            "NAS model weights havent worked when tested"
        )

        supported_models = yolo_models + rtdetr_models + nas_models
       
        assert self.model_weights in supported_models, (
            f"Requested model {self.model_weights} not supported"
        )
        self.model = model_family_from_string[self.model_family](self.model_weights)

    def predict(self, file_paths: List[str]) -> Predictions:
        predictions = self.model(file_paths, stream=False)
        detections = sv.Detections.from_ultralytics(predictions[0])
        class_list = list(predictions[0].names.values())
        class_list = [f'{self.name}: {class_name}' for class_name in class_list]
        predictions = Predictions.from_supervision(
            detections,
            class_list,
        )
        return predictions



def confidence_trigger(
        predictions: Predictions,
        confidence_trigger: float,
    ) -> bool: 
    predictions_df = predictions.to_dataframe()
    condition = predictions_df.scores < confidence_trigger
    filtered_df = predictions_df[condition]
    filtered_predictions = Predictions.from_dataframe(filtered_df)

    if len(filtered_predictions) == 0:
        return False
    else:
        return True


@dataclass
class ConditionalDetector:
    # should be detector not Model
    model: Model

    # If you created these as instances 
    # you could just check the instance type
    # and act as appropriate
    frame_level_condition: Callable = None
    prediction_level_condition: Callable = None

    def predict(
        self,
        file_path: str,
        predictions: Predictions,
        iteration: int,
    ) -> Predictions:
        """
        Run inference and flag anything that should be reconsidered
        by other models
        """
        if iteration == 0: 
            return self.model.predict(file_path)

        if (self.prediction_level_condition is None) and (self.frame_level_condition is None):
            return self.model.predict(file_path)

        if self.prediction_level_condition is not None:
            for prediction in predictions.to_list_of_preds():
                
                if self.prediction_level_condition(prediction):
                    return self.model.predict(file_path)

        if self.frame_level_condition is not None: 
            if self.frame_level_condition(predictions):
                return self.model.predict(file_path)

        return predictions


@dataclass
class ConditionalClassifier:
    """
    This could at least now take multiply underlying 
    NN classifiers and work together.
    """
    # Should be classifier not Model
    model: Model

    # This is a prediction level 
    # Condition
    prediction_level_condition: Callable

    def predict(
            self, 
            file_path: str, 
            predictions: Predictions, 
            iteration: int
        ) -> Predictions:

        if iteration == 0: 
            raise NotImplementedError("""
                Starting the chain with a classifier 
                is not yet supported
            """)


        output_preds = []
        list_of_preds = predictions.to_list_of_preds()

        for prediction in list_of_preds:
            if self.prediction_level_condition(prediction): 
                image = Image.open(file_path)
                cropped_image = image.crop(prediction.box)
                classification_prediction = self.model.predict('', image=cropped_image)

                # TODO: give Prediction or ClassificationPrediction a method to 
                # tidy this up
                prediction.label = f'{self.model.name}: {classification_prediction.label}'
                prediction.score = classification_prediction.score
             
            # either the label is editted or it's the 
            # original prediction
            output_preds.append(prediction)

        return Predictions.from_list_of_preds(output_preds)
            

@dataclass
class ModelChain(Model):
    conditional_models: List[Union[ConditionalDetector, ConditionalClassifier]]
    log_level: str

    def predict(self, file_path: str) -> Predictions:
        """
        Should the triggers be attached to the curr model,
        or the next model?

        Which creates the cleaner API
        """
        predictions = empty_predictions

        for iteration, conditional_model in enumerate(self.conditional_models):
            start = time.time()
            predictions = conditional_model.predict(
                file_path,
                predictions,
                iteration,
            )
            end = time.time()
            elapsed = end - start
            if self.log_level == "verbose":
                # TODO: This should use the name of the 'conditional model', not the underlying model
                print(
                    f"Preds with {conditional_model.model.name} took {elapsed}"
                )

        return predictions 


@dataclass
class GroundedSamDetector(Model):
    ontology: Dict[str, str]
    name: str

    def __post_init__(self):
        self.model = GroundedSAM(ontology=CaptionOntology(self.ontology))

    def predict(self, file_path: str) -> Predictions:
        detections = self.model.predict(file_path)
        # Quick patch fix
        detections = detections.with_nms(class_agnostic=True)
        class_list = [f'{self.name}: {class_name}' for class_name in list(self.ontology.values())]
        return Predictions.from_supervision(detections, class_list)


# TODO: Should probably use ABC and abstractmethod
@dataclass
class Embedder:
    def embed(self, file_path: str):
        raise NotImplementedError('Need to implement embed()')


@dataclass
class HFEmbedder(Embedder):
    preprocessor_name: str
    model_name: str
    device: str

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = ViTImageProcessor.from_pretrained(self.preprocessor_name)
        self.model = ViTModel.from_pretrained(self.model_name).to(self.device)

    def embed(self, file_path: str = None, image = None) -> List[float]:
        # TODO, make this neater
        if image is None:
            image = Image.open(file_path)

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state[0][0].tolist()

        return embeddings


@dataclass
class NNClassifier:
    collection_name: str
    client: QdrantClient
    embedder: Embedder
    name: str

    # shouldn't need to define this upfront
    class_list: List[str]

    def __post_init__(self):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=384, distance=models.Distance.COSINE
            ),
        )

    def train(self, train_dataset: fo.Dataset):
        """
        Just needs to run inference over all of the objects.
        Make it very easy to plug and play different models
        into this.
        """
        embeddings = []
        for file_path in tqdm(train_dataset.values("filepath")):
            embedding = self.embedder.embed(file_path=file_path)
            embeddings.append(embedding)

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=idx,
                    vector=embedding,
                    payload={
                        "file_path": file_path,
                        "label": label,
                    },
                )
                for idx, (vector, file_path, label) in enumerate(
                    zip(
                        embeddings,
                        train_dataset.values("filepath"),
                        train_dataset.values("ground_truth"),
                    )
                )
            ],
        )

    def predict(
        self, query_image_path: str, image = None,  num_nearest_neighbours: int = 5
    ) -> ClassificationPrediction:
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=self.embedder.embed(image=image),
            limit=num_nearest_neighbours,
        )

        predictions = []
        for hit in hits:
            prediction = ClassificationPrediction(
                label=hit.payload["label"]["label"],
                score=hit.score,
            )
            predictions.append(prediction)

        return self.aggregate(predictions)

    @staticmethod
    def aggregate(
        predictions: List[ClassificationPrediction], method="majority"
    ) -> ClassificationPrediction:
        if method == "majority":
            neighbour_labels = [prediction.label for prediction in predictions]
            label =  max(set(neighbour_labels), key=neighbour_labels.count)
            score = len([x for x in neighbour_labels if x == label]) / len(neighbour_labels)
            return ClassificationPrediction(
                label=label,
                score=score,
            )

        if method == "weighted":
            raise NotImplementedError("Weighted aggregate not yet implemented")

