import fiftyone as fo
import fiftyone.zoo as foz
import typer

def main():
    """
    Load and export dataset for a given class.
    """
    dataset = foz.datasets.load_zoo_dataset(
        'coco-2017',
        classes=['bottle'],
        max_samples=100,
    )

    size_subsample = fo.Dataset()
    
    for sample in dataset:
        detections = sample['ground_truth']['detections']
        for detection in detections:
            print(detection['bounding_box'])


    dataset.export(
        'bottles_dataset',
        dataset_type=fo.types.COCODetectionDataset,
    )

    session = fo.launch_app(dataset, remote=True, address="0.0.0.0", desktop=True)
    session.wait()


if __name__ == '__main__':
    typer.run(main)
