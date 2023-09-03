import os

from PIL import Image

from transformers import ViTImageProcessor, ViTModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from tqdm import tqdm
import torch
import typer
import fiftyone as fo
from grounding_dino import GroundingDinoDetector



def get_embeddings(batch):
    """
    Taken from the demo notebook.
    """
    inputs = processor(images=batch['image'], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
    batch['embeddings'] = outputs
    return batch


def main(
    collection_name: str = 'detector',
    device: str = 'cuda',
    data_path: str = 'data/splits_final_deblurred/train/data',
    limit: int = 20,
):
    assert data_path is not None, 'Need to specify a path to images'

    client = QdrantClient(host="localhost", port=6333)
    
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )
    
    #dataset = fo.Dataset.from_dir(
    #data_path='data/splits_final_deblurred/train/data',
    #labels_path='data/splits_final_deblurred/train/labels.json',
    # dataset_type=fo.types.COCODetectionDataset,
    # )

    # Would probably work better with SAM 
    # because you're just cropping things out 
    # SAM tends to 'over segment though'
    # and chop things up
    detector = GroundingDinoDetector(
        text_prompt='paper,cardboard',
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
    model = ViTModel.from_pretrained('facebook/dino-vits16').to(device)

    embeddings = {}
    file_names = os.listdir(data_path)

    for file_name in tqdm(file_names[:limit]):
        file_path = os.path.join(data_path, file_name)
        detections = detector.predict(file_path)
        image = Image.open(file_path)
        image_width, image_height = image.size

        embeddings[file_name] = {}
        for detection in detections:
            
            # these are all relative 
            x, y, w, h = detection
            absolute_box =  [
                x * image_width, 
                y * image_height, 
                (x + w) * image_width, 
                (y + h) * image_height,
            ]
            #rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]
            cropped_object = image.crop(absolute_box)
            inputs = processor(images=cropped_object, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()

            embeddings[file_name][str(absolute_box)] = outputs

    np.save("vectors", np.array(embeddings), allow_pickle=True)
    
    batch_size = 1000
    print('Uploading embeddings to Qrant')
    for i in range(0, len(embeddings), batch_size):

        low_idx = min(i+batch_size, len(embeddings))

        batch_of_filenames = file_names[i: low_idx]
        batch_of_embs = embeddings[i: low_idx]
        batch_of_payloads = payloads[i: low_idx]

        client.upsert(
            collection_name=my_collection,
            points=models.Batch(
                ids=batch_of_filenames,
                vectors=batch_of_embs,
                #payloads=batch_of_payloads
            )
        )

if __name__ == '__main__':
    typer.run(main)
