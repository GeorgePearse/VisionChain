import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

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
    class_list: List[str]

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

    def from_supervision(detections: sv.Detections, class_list: List[str]):
        labels = [class_list[x] for x in detections.class_id.tolist()]
        return Predictions(
            boxes=detections.xyxy.tolist(),
            labels=labels,
            scores=detections.confidence.tolist(),
            class_list=class_list,
        )

    def to_supervision(self) -> sv.Detections:
        class_id = np.array([self.class_list.index(x) for x in self.labels])
        return sv.Detections(
            xyxy=np.array(self.boxes),
            class_id=class_id,
            confidence=np.array(self.scores),
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "labels": self.labels,
                "boxes": self.boxes,
                "scores": self.scores,
                "class_list": [self.class_list] * len(self.labels),
            }
        )

    @staticmethod
    def from_dataframe(df: pd.DataFrame):
        if len(df) == 0: 
            return Predictions(
                labels=[],
                boxes=[[]],
                scores=[],
                class_list=[]
            )

        else:
            return Predictions(
                labels=df["labels"].tolist(),
                boxes=df["boxes"].tolist(),
                scores=df["scores"].tolist(),
                class_list=df["class_list"].tolist()[0],
            )


@dataclass
class ClassificationPrediction:
    name: str
    score: float


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

    detections: Optional[Predictions] = None

    # hit if there's a downstream Detector
    repredict_whole_frame: bool = False

    # just whether the last model ran or not
    triggered: Optional[bool] = None


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


@dataclass
class Condition:
    def evaluate(detections: sv.Detections) -> bool:
        raise Exception(
            """"
            Need to implement the method to 
            evaluate the specified condition.
        """
        )


@dataclass
class UncertaintyRejection(Condition):
    confidence_trigger: float = 1

    def evaluate(self, predictions: Predictions) -> bool:
        predictions_df = predictions.to_dataframe()
        condition = predictions_df.scores < self.confidence_trigger
        filtered_df = predictions_df[condition]
        filtered_predictions = Predictions.from_dataframe(filtered_df)

        if len(filtered_predictions) == 0:
            return False
        else:
            return True


@dataclass
class ConditionalModel:
    model: Model
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
        self,
        file_path: str,
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

    def predict(self, file_path: str) -> Predictions:
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
                print(
                    f"Preds with {conditional_model.model.name} took {elapsed}. Condition triggered: {conditional_model.condition_triggered}"
                )

        return speculative_predictions.detections


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


@dataclass
class FastBase(ConditionalModel):
    confidence_trigger: float

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

        repredict_whole_frame = UncertaintyRejection(
            confidence_trigger=self.confidence_trigger,
        ).evaluate(detections)

        return SpeculativePrediction(
            detections=detections,
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
                detections=detections,
                repredict_whole_frame=False,
            )
        else:
            return speculative_prediction


@dataclass
class Embedder:
    def embed(self, file_path: str):
        pass


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
            #embeddings = self.model(**inputs) 
            #print('successful embedding generation')

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
                name=hit.payload["label"]["label"],
                score=hit.score,
            )
            predictions.append(prediction)

        return self.aggregate(predictions)

    @staticmethod
    def aggregate(
        predictions: List[ClassificationPrediction], method="majority"
    ) -> str:
        if method == "majority":
            neighbour_labels = [prediction.name for prediction in predictions]
            return max(set(neighbour_labels), key=neighbour_labels.count)

        if method == "weighted":
            raise Exception("Weighted aggregate not yet implemented")


@dataclass
class ConditionalNNClassifier(ConditionalModel):
    model: NNClassifier
    condition: Callable

    def match(self, speculative_prediction: SpeculativePrediction) -> bool:
        """
        Biggest weakness is prediction level conditions
        """
        pass

    def speculate(self, file_path: str, speculative_prediction: SpeculativePrediction) -> SpeculativePrediction:

        # does not edit the boxes
        output_labels = []
        output_scores = []
        output_boxes = []

        # prioristise fixing this, should be easy
        for label, box, score in zip(
            speculative_prediction.detections.labels,
            speculative_prediction.detections.boxes,
            speculative_prediction.detections.scores,
        ):
            # simplifies use of lambda for condition
            prediction = Prediction(
                label=label,
                box=box,
                score=score,
            )

            if self.condition(prediction): 
                speculative_prediction.condition_triggered = True
                image = Image.open(file_path)
                cropped_image = image.crop(prediction.box)
                cropped_image.save('crop.jpeg')
                print('successfully cropped image')
                new_label = self.model.predict('', image=cropped_image)
                output_labels.append(f'{self.model.name}: {new_label}')
                output_scores.append(0.5)
                output_boxes.append(box)
            else: 
                output_labels.append(label)
                output_scores.append(score)
                output_boxes.append(box)

        speculative_prediction.detections = Predictions(
            labels=output_labels,
            scores=output_scores,
            boxes=output_boxes,
            class_list=self.model.class_list,
        )
        return speculative_prediction

            

