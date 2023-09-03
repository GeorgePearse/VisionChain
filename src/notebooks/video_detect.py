import torch
import fiftyone as fo

from fiftyone import ViewField as F
import inquirer 
import shutil
import os
import typer
import json
import pandas as pd
import numpy as np
from rich import print as print
import time
import json
import cv2

from datetime import datetime
from pandas.core.frame import DataFrame
import fiftyone.zoo as foz
from dataclasses import dataclass
from typing import List
from get_predictions import get_predictions

from heuristic_model import HeuristicModel
from colour_model import ColourModel

def delete_voxel_datasets():
    for dataset_name in fo.list_datasets():
        print(f'Deleting voxel dataset = {dataset_name}')
        dataset = fo.load_dataset(dataset_name)
        dataset.delete()
    
def main(
        video_path: str = 'bottle_counter/bottles.mp4',
        view_dataset: bool = True,
        limit: int = None,
        class_name='wood',
        colour='green',
    ):
    """
    Use a heuristic model on a the cooper dataset.
    """
    assert os.path.exists(video_path), 'Path does not exist'

    os.system('sudo fuser -k 5151/tcp')

    # sometimes voxdl51 breaks, this is the easiest way to make sure 
    # it will be working
    delete_voxel_datasets()

    model = ColourModel(
        min_width=5,
        min_height=5,
    )

    # not sure about the to_frames() bit 
    dataset = fo.Dataset.from_videos(
        [video_path],
    ).shuffle()

    dataset = dataset.to_frames(sample_frames=True).clone()

    if limit: 
        val_per_belt_dataset = val_per_belt_dataset[:limit].clone()

    # use_nms should normally be set to False for thresholding.
    labelled_dataset = get_predictions(
        model, 
        dataset, 
    )

    if view_dataset:
        session = fo.launch_app(labelled_dataset, remote=True, address="0.0.0.0", desktop=True)
        session.wait()


if __name__ == '__main__': 
    typer.run(main)
