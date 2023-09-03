import inspect
import torch
from PIL import Image
from torchvision import transforms
import logging
import fiftyone as fo
from PIL import Image
from icecream import ic
from vision.tools.predictions import (
    Predictions, 
    EmptyPredictions,
)
from typing import List, Tuple


def get_predictions(
        model,
        dataset: fo.Dataset,
        output_key = 'predictions',
    ):
    """
    Run inference over a Voxel51 
    dataset
    """
    with fo.ProgressBar() as pb:

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
                label=class_name,
                bounding_box=rel_box,
                confidence=score,
            )

            detections.append(detection)

         # Save predictions to dataset
        sample[output_key] = fo.Detections(detections=detections)
        sample.save()

    return dataset
