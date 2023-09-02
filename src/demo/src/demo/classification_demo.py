import typer
from transformers import ViTImageProcessor, ViTModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from datasets import load_dataset
import numpy as np
import torch
import fiftyone as fo
from PIL import Image
from tqdm import tqdm
from qdrant_client.models import PointStruct
import shutil
import os
from dataclasses import dataclass
from typing import List

import fiftyone.zoo as foz

from embedder import Embedder, HFEmbedder
from classifier import Classifier

def main(
        mode: str = 'local',
        limit: int = 1000,
    ):

    datasets = foz.list_zoo_datasets()
    import pdb; pdb.set_trace()

    if os.path.exists('qdrant'):
        shutil.rmtree('qdrant')

    if MODE == 'remote':
        client = QdrantClient(host="localhost", port=6333)

    if MODE == 'dev':
        client = QdrantClient(path="qdrant") 

    split = 'train'
    train_dataset = fo.Dataset.from_dir(
        data_path=f'data/splits_final_deblurred/{split}/data',
        labels_path=f'data/splits_final_deblurred/{split}/labels.json',
        dataset_type=fo.types.COCODetectionDataset,
        max_samples=LIMIT,
    )

    split = 'test'
    test_dataset = fo.Dataset.from_dir(
        data_path=f'data/splits_final_deblurred/{split}/data',
        labels_path=f'data/splits_final_deblurred/{split}/labels.json',
        dataset_type=fo.types.COCODetectionDataset,
        max_samples=LIMIT,
    )

    train_dataset.to_patches('detections').export(
        export_dir='train_objects',
        dataset_type=fo.types.ImageClassificationDirectoryTree,
        overwrite=True,
    )

    test_dataset.to_patches('detections').export(
        export_dir='test_objects',
        dataset_type=fo.types.ImageClassificationDirectoryTree,
        overwrite=True,
    )

    train_object_dataset = fo.Dataset.from_dir(
        dataset_dir='train_objects',
        dataset_type=fo.types.ImageClassificationDirectoryTree,
    )

    test_object_dataset = fo.Dataset.from_dir(
        dataset_dir='test_objects',
        dataset_type=fo.types.ImageClassificationDirectoryTree,
    )
    
    hf_embedder = HFEmbedder(
        preprocessor_name='facebook/dino-vits16',
        model_name='facebook/dino-vits16',
    ) 

    classifier = Classifier(
        collection_name='detector',
        client=client, 
        embedder=hf_embedder,
    )
        
    for file_path in test_dataset.values('filepath'):
        prediction = classifier.predict(file_path)
        print (file_path)
    


if __name__ == '__main__':
    typer.run(main)

