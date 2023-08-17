import torch
import fiftyone as fo

from fiftyone import ViewField as F
import inquirer 

from vision.tools.inference import (
    get_predictions,
)

from vision.tools.pipelines import batched_send_to_cvat
from vision.tools.model import (
    Model,
    Ensemble,
    YoloModel,
    HeuristicModel,
    colour_store,
)
from vision.tools import config
from vision.tools.postprocessor import Postprocessor
from vision.tools.preprocessor import Preprocessor
from vision.tools.predictions import Predictions
#from vision.tools.postprocessors import ClientPostprocessors
import shutil
import os
import typer
import json
import pandas as pd
import numpy as np
from rich import print as print
import time
import json

from datetime import datetime
from pandas.core.frame import DataFrame
import fiftyone.zoo as foz
from dataclasses import dataclass
from typing import List

def delete_voxel_datasets():
    for dataset_name in fo.list_datasets():
        print(f'Deleting voxel dataset = {dataset_name}')
        dataset = fo.load_dataset(dataset_name)
        dataset.delete()
    
def main(
        inference_path: str = '../../vision-research/data/construction_and_demolition/data',
        view_dataset: bool = True,
        limit: int = None,
    ):
    """
    Use a heuristic model on a the cooper dataset.
    """
    # sometimes voxdl51 breaks, this is the easiest way to make sure 
    # it will be working
    os.system('sudo fuser -k 5151/tcp')
    delete_voxel_datasets()

    model = HeuristicModel(
        class_name='wood',
        colour='brown',
    )

    val_per_belt_dataset = fo.Dataset.from_images_dir(
        inference_path,
    ).shuffle()

    if limit: 
        val_per_belt_dataset = val_per_belt_dataset[:limit].clone()

    print(f'Length is {val_per_belt_dataset}')

    if string_path_filter: 
        val_per_belt_dataset = val_per_belt_dataset.match(F('filepath').contains_str(string_path_filter)).clone()

    print(f'Length is {val_per_belt_dataset}')
    # so that you can threshold in Voxel 51 

    time_before_predictions = time.time()
    
    output_dir = 'send_to_cvat_evterra'

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # use_nms should normally be set to False for thresholding.
    dataset = get_predictions(
        model, 
        val_per_belt_dataset, 
        class_list=class_list,
        segmentation=segmentation,
    )

    if view_dataset:
        session = fo.launch_app(dataset, remote=True, address="0.0.0.0", desktop=True)
        session.wait()


if __name__ == '__main__': 
    typer.run(main)
