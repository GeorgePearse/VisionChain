import fiftyone.zoo as foz
import fiftyone as fo
from rich import print

dataset_name = 'quickstart'

# List available datasets
print(foz.list_zoo_datasets())

dataset = foz.load_zoo_dataset(dataset_name)

# Export images and ground truth labels to disk
dataset.export(
    export_dir=f'datasets/{dataset_name}',
    dataset_type=fo.types.COCODetectionDataset,
    label_field="ground_truth",
)
