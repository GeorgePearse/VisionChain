from transformers import ViTImageProcessor, ViTModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import torch
import typer
import fiftyone as fo
from templates import ObjectCropper

object_cropper = ObjectCropper(

)

client = QdrantClient(host="localhost", port=6333)

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
    data_path: str
):

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )
    
    #dataset = fo.Dataset.from_dir(
    #data_path='data/splits_final_deblurred/train/data',
    #labels_path='data/splits_final_deblurred/train/labels.json',
    # dataset_type=fo.types.COCODetectionDataset,
    # )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
    model = ViTModel.from_pretrained('facebook/dino-vits16').to(device)

    for file_name in tqdm(os.listdir(data_path)):
        file_path = os.path.join(data_path, file_name)
        detections = detector.predict(file_path)

    
        for detection in detections: 
            cropped_object = Image.crop()
            inputs = processor(images=cropped_object, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()

    np.save("vectors", np.array(outputs), allow_pickle=False)
    

if __name__ == '__main__':
    typer.run(main)
