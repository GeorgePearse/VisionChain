import typer
from qdrant_client import QdrantClient
import os
import vision_chain as vc
import fiftyone as fo
from get_predictions import get_predictions

def main(
    limit: int = 10,
    test_path: str = "../data/test_dataset/data",
    nn_training_data: str = '../data/nn_training_data/',
):
    """
    GroundedSAM to crop then DINO + QDrant to classify!
    """

    hf_embedder = vc.HFEmbedder(
        model_name="facebook/dino-vits16", 
        preprocessor_name="facebook/dino-vits16",
        device='cuda',
    )

    nn_classifier = vc.NNClassifier(
        embedder=hf_embedder,
        client=QdrantClient(path="qdrant.db"),
        collection_name="vision_chain_classifier",
        name='nn',
        class_list=['husky', 'labrador'],
    )

    training_dataset = fo.Dataset.from_dir(
        dataset_dir=nn_training_data,
        dataset_type=fo.types.ImageClassificationDirectoryTree,
    )

    nn_classifier.train(training_dataset)

    yolov8 = vc.UltralyticsDetector(
        model_family="YOLO",
        model_weights="yolov8n.pt",
        name='yolov8n',
    )

    grounded_sam = vc.GroundedSamDetector(
        ontology={
            "plastic bottle": "bottle",
            "glass bottle": "bottle",
            "wine bottle": "bottle",
            "person": "person",
            "grapes": "fruit",
            "banana": "fruit",
            "bird": "bird",
            "glove": "glove",
            "basket": "basket",
            "trophy": "trophy",
            "traffic lights": "traffic lights",
            "orange": "orange",
            "lemon": "lemon",
            "cow": "cow",
            "laptop": "laptop",
            "dog": "dog",
        },
        name='grounded_sam',
    )

    file_paths = [
        os.path.join(test_path, file_name) for file_name in os.listdir(test_path)
    ]

    if limit:
        file_paths = file_paths[:limit]

    model = vc.ModelChain(
        [
            vc.ConditionalDetector(
                model=yolov8,
            ),
            vc.ConditionalDetector(
                model=grounded_sam, 
                frame_level_condition = lambda predictions: any([score < 0.5 for score in predictions.scores]),
                prediction_level_condition = lambda pred: 'cat' in pred.label,
            ),
            vc.ConditionalClassifier(
                model=nn_classifier,
                prediction_level_condition = lambda pred: 'dog' in pred.label,
            ),
        ],
        log_level = 'verbose',
    )

    dataset = fo.Dataset.from_images(file_paths)
    dataset = get_predictions(dataset, model)

    session = fo.launch_app(dataset, remote=True, address="0.0.0.0", desktop=True)
    session.wait()


if __name__ == "__main__":
    typer.run(main)
