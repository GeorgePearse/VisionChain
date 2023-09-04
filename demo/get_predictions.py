import inspect
import logging
from typing import List, Tuple

import fiftyone as fo
import torch
from icecream import ic
from PIL import Image
from torchvision import transforms
from vision.tools.predictions import EmptyPredictions, Predictions

def get_predictions(
    dataset: fo.Dataset,
    model, 
    output_key: str = "predictions",
):
    """
    Run inference over a Voxel51
    dataset
    """
    with fo.ProgressBar() as pb:
        for sample in dataset:
            preds = model.predict(sample.filepath)
            image = Image.open(sample.filepath)
            w, h = image.size

            # Convert detections to FiftyOne format
            detections = []
            for label, score, box in zip(
                preds.labels,
                preds.scores,
                preds.boxes,
            ):
                # Convert to [top-left-x, top-left-y, width, height]
                # in relative coordinates in [0, 1] x [0, 1]
                x1, y1, x2, y2 = box
                rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

                detection = fo.Detection(
                    label=label,
                    bounding_box=rel_box,
                    confidence=score,
                )

                detections.append(detection)

            # Save predictions to dataset
            sample[output_key] = fo.Detections(detections=detections)
            sample.save()

    return dataset
