from autodistill_grounded_sam import GroundedSAM
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


@dataclass
class Cropper:
    """
    Should better generalise this.
    """
    model_name: str = 'fastest' 

    def __post_init__(self):
        assert self.model_name in ['yolo_nas_l', 'yolo_nas_m', 'yolo_nas_s'], (
            'Selected model is not available'
        ) 

        self.model = super_gradients.training.models.get(
            "yolo_nas_l", pretrained_weights="coco"
        ).cuda()

    def predict(self, file_path: str) -> List[fo.Detection]:
        predictions = self.model.predict(file_path)

        class_names = predictions[0].class_names
        predictions = predictions[0].prediction

        im = Image.open(file_path)
        width, height = im.size

        detections = []

        for bbox, score, label in zip(
            predictions.bboxes_xyxy,
            predictions.confidence.astype(float),
            predictions.labels.astype(int),
        ):

            rel_box = [
                bbox[0] / width,
                bbox[1] / height,
                (bbox[2] - bbox[0]) / width,
                (bbox[3] - bbox[1]) / height,
            ]
 
            detection = fo.Detection(
                label=class_names[label],
                bounding_box=rel_box,
                confidence=score,
            )
            detections.append(detection)

        return detections



def main(
    limit: int = None,
):
    """
    GroundedSAM to crop then DINO + QDrant to classify!
    """

    cropper = Cropper(model_name='yolo_nas_s')

    dir_path = "/home/ubuntu/VisionChain/src/demo/bottles_dataset/data"
    file_paths = [
        os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path)
    ]

    dataset = fo.Dataset()  # .from_image_dir(dir_path)

    if limit is not None:
        file_paths = file_paths[:limit]

    for file_path in tqdm(file_paths):
        detections = []
        sample = fo.Sample(os.path.abspath(file_path))
        dataset.add_sample(sample)

        detections = cropper.predict(file_path)

        # Save predictions to dataset
        sample["predictions"] = fo.Detections(detections=detections)
        sample.save()

    session = fo.launch_app(dataset, remote=True, address="0.0.0.0", desktop=True)
    session.wait()


if __name__ == "__main__":
    typer.run(main)
