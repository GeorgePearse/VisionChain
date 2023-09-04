import os

import typer
from qdrant_client import QdrantClient
import fiftyone as fo
import vision_chain as vc
from get_predictions import get_predictions


def main(
    limit: int = 10,
    training_set_path = '../data/nn_training_data/',
    test_dataset = '../data/test_dataset/data',
):
    """
    GroundedSAM to crop then DINO + QDrant to classify!
    """

    hf_embedder = vc.HFEmbedder(
        model_name="facebook/dino-vits16",
        preprocessor_name="facebook/dino-vits16",
        device="cuda",
    )

    train_dataset = fo.Dataset.from_dir(
        dataset_dir=training_set_path,
        dataset_type=fo.types.ImageClassificationDirectoryTree,
    )

    nn_classifier = vc.NNClassifier(
        embedder=hf_embedder,
        client=QdrantClient(path="qdrant.db"),
        collection_name="vision_chain_classifier",
        name='nn_classifier',
        class_list=['husky', 'labrador']
    )
    nn_classifier.train(train_dataset)

    fast_base = vc.UltralyticsDetector(
        model_family="YOLO",
        model_weights="yolov8n.pt",
        name='yolov8',
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
        os.path.join(test_dataset, file_name) for file_name in os.listdir(test_dataset)
    ]

    if limit:
        file_paths = file_paths[:limit]

    model = vc.ModelChain(
        [
            vc.FastBase(model=fast_base, confidence_trigger=0.5),
            vc.AccurateFallback(model=grounded_sam),
            vc.ConditionalNNClassifier(
                model = nn_classifier,
                condition = lambda pred: ('dog' in pred.label) or ('cat' in pred.label),
            )
        ],
        log_level="verbose",
    )

    dataset = fo.Dataset.from_images(file_paths)
    dataset = get_predictions(dataset, model)

    session = fo.launch_app(dataset, remote=True, address="0.0.0.0", desktop=True)
    session.wait()


if __name__ == "__main__":
    typer.run(main)
