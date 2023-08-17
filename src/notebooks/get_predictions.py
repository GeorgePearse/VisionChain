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
        dataset,
        class_list: list = None,
        output_key = 'predictions',
        device: str ='cpu',
        w: int = 1920, 
        h: int = 1080,
    ):
    
    logging.info(f"""
        Output key is {output_key}, WARNING: getting this wrong leads 
        to horrible downstream bugs with missing labels and out of sync 
        datasets.
        
        For cropping it should be set to detections so that the output
        of the model can be treated the same as GT labels.
        
        For evaluation and thresholding it should be set to predictions, 
        so that the output of the model can be compared to the GT
        labels.
    """)
    assert class_list is not None, 'Need to specify class list'

    with fo.ProgressBar() as pb:
        for sample in pb(dataset):
            try:
                preds = model.predict(sample.filepath)

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
                    
                    assert int(label) <= (len(class_list) - 1), f'Index {label} is out of class list range'
                    class_name = class_list[label]
                    
                    detection = fo.Detection(
                        label=class_name,
                        bounding_box=rel_box,
                        confidence=score,
                    )

                    detections.append(detection)

                 # Save predictions to dataset
                sample[output_key] = fo.Detections(detections=detections)
                sample.save()
            except Exception as e:
                #raise e
                # need to be more specific with exception handling
                sample[output_key] = fo.Detections(detections=[])
                sample.save()

                print(f'Exception hit == {e}')
                print(f"""
                    WARNING: Possible there are no objects in the frames.
                    Or the model has been trained on more classes than are 
                    included in the provided class_list.
                """)
                logging.info(e)

    return dataset