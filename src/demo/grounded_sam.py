from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology

import typer
from tqdm import tqdm
import os
from PIL import Image
import fiftyone as fo


def main(
        limit: int = None,
    ):
    """
    GroundedSAM to crop then DINO + QDrant to classify!
    """
    # define an ontology to map class names to our GroundingDINO prompt
    # the ontology dictionary has the format {caption: class}
    # where caption is the prompt sent to the base model, and class is the label that will
    # be saved for that caption in the generated annotations
    ONTOLOGY = {
        "plastic bottle": "bottle",
        'glass bottle': 'bottle',
        'wine bottle': 'bottle',
        'person': 'person',
        'grapes': 'fruit',
        'banana': 'fruit', 
        'bird': 'bird',
        'glove': 'glove',
        'basket': 'basket',
        'trophy': 'trophy',
        'traffic lights': 'traffic lights',
        'orange': 'orange',
        'lemon': 'lemon',
        'cow': 'cow',
        'laptop': 'laptop',
        'dog': 'dog',
    }

    base_model = GroundedSAM(ontology=CaptionOntology(ONTOLOGY))
    #class_list = list(set(ONTOLOGY.keys()))
    class_list = list(ONTOLOGY.keys())

    dir_path = "/home/ubuntu/VisionChain/src/demo/bottles_dataset/data"
    file_paths = [os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path)]

    dataset = fo.Dataset()  #.from_image_dir(dir_path)
    
    if limit is not None:
        file_paths = file_paths[:limit]

    for file_path in tqdm(file_paths):
        
        detections = []
        sample = fo.Sample(os.path.abspath(file_path))
        dataset.add_sample(sample)

        prediction = base_model.predict(file_path)

        prediction = prediction.with_nms(
            threshold=0.8, 
            class_agnostic=True,
        )
        im = Image.open(file_path)
        width, height = im.size

        for bbox, score, label, mask in zip(
            list(prediction.xyxy),
            list(prediction.confidence),
            list(prediction.class_id),
            list(prediction.mask),
        ):
            rel_box = [
                bbox[0] / width,
                bbox[1] / height,
                (bbox[2] - bbox[0]) / width,
                (bbox[3] - bbox[1]) / height,
            ]
            print(rel_box)
            #print(bbox, score, label, mask)

            detection = fo.Detection(
                label=class_list[label],
                bounding_box=rel_box,
                confidence=score,
            )
            
            detections.append(detection)

        # Save predictions to dataset
        sample['predictions'] = fo.Detections(detections=detections)
        sample.save()

    session = fo.launch_app(dataset, remote=True, address="0.0.0.0", desktop=True)
    session.wait()

if __name__ == '__main__':
    typer.run(main)
