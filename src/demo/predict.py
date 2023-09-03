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
        inference_path: str = 'datasets/quickstart', 
        view_dataset: bool = True,
        limit: int = None,
        class_name='wood',
        colour='green',
    ):
    """
    Use a heuristic model on a the cooper dataset.
    """
    assert os.path.exists(inference_path), 'Path does not exist'

    # sometimes voxdl51 breaks, this is the easiest way to make sure 
    # it will be working
    os.system('sudo fuser -k 5151/tcp')
    delete_voxel_datasets()

    model = ColourModel()

    val_per_belt_dataset = fo.Dataset.from_images_dir(
        inference_path,
    ).shuffle()

    if limit: 
        val_per_belt_dataset = val_per_belt_dataset[:limit].clone()

    # use_nms should normally be set to False for thresholding.
    dataset = get_predictions(
        model, 
        val_per_belt_dataset, 
    )

    if view_dataset:
        session = fo.launch_app(dataset, remote=True, address="0.0.0.0", desktop=True)
        session.wait()


if __name__ == '__main__': 
    typer.run(main)
