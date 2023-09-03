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

from colour_model import ColourModel


def main(
    dir_path: str = "/home/ubuntu/VisionChain/src/demo/bottles_dataset/data",
    limit: int = None,
):
    """
    GroundedSAM to crop then DINO + QDrant to classify!
    """

    model = ColourModel()

    file_paths = [
        os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path)
    ]

    if limit is not None:
        file_paths = file_paths[:limit]

    dataset = model.predict(file_paths) 

    session = fo.launch_app(dataset, remote=True, address="0.0.0.0", desktop=True)
    session.wait()


if __name__ == "__main__":
    typer.run(main)
