from dataclasses import dataclass
import pandas as pd
import supervision as sv
from typing import List
import numpy as np


@dataclass
class Predictions:
    """
    Predictions for one image.

    Arguments
        scores: e.g. [0.4] floats 0 -> 1.
        boxes: [[x1, y1, x2, y2]], where all are absolute values (VOC format) e.g 0 < x < 1920.
        labels: e.g. [0] integer represenatations of classes.
    """

    scores: List[float]
    boxes: List[List[float]]
    labels: List[int]

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

    def to_supervision(self) -> sv.Detections:
        return sv.Detections(
            xyxy=np.array(self.boxes),
            class_id=np.array(self.labels),
            confidence=np.array(self.scores),
        )

    def from_supervision(detections: sv.Detections) -> "Predictions":
        return Predictions(
            boxes=detections.xyxy.tolist(),
            labels=detections.class_id.tolist(),
            scores=detections.confidence.tolist(),
        )

    def from_inference_output(inference_output) -> "Predictions":
        labels = inference_output["labels"][0].tolist()
        scores = inference_output["dets"][0][:, -1].tolist()
        boxes = inference_output["dets"][0][:, :-1].tolist()

        return Predictions(
            boxes=boxes,
            labels=labels,
            scores=scores,
        )

    def to_detections(self) -> List[dict]:
        """
        Just matching the old format for now.
        """
        detections = []
        for label, box, score in zip(self.labels, self.boxes, self.scores):
            detections.append(
                {
                    "label": label,
                    "box": box,
                    "score": score,
                }
            )

        return detections

    def from_detections(detections: list) -> "Predictions":
        """
        Out of the old format.
        """
        return Predictions(
            labels=[x["label"] for x in detections],
            boxes=[x["box"] for x in detections],
            scores=[x["score"] for x in detections],
        )

    def to_dataframe(self) -> pd.core.frame.DataFrame:
        return pd.DataFrame(
            {
                "labels": self.labels,
                "boxes": self.boxes,
                "scores": self.scores,
            }
        )

    def from_dataframe(df: pd.core.frame.DataFrame) -> "Predictions":
        return Predictions(
            labels=df["labels"].tolist(),
            boxes=df["boxes"].tolist(),
            scores=df["scores"].tolist(),
        )


EmptyPredictions = Predictions(
    scores = [],
    boxes = [[]],
    labels = [],
)
