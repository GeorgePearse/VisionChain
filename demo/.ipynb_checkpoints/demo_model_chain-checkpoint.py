import vision_chain as vc


def main(
    limit: int = 100,
):
    """
    GroundedSAM to crop then DINO + QDrant to classify!
    """
    qdrant_client = QDrantClient('qdrant.db')

    hf_embedder = vc.HFEmbedder(
        model_name = 'facebook/dino-vits16',
        preprocessor_name = 'facebook/dino-vits16'
    )

    classifier = vc.Classifier(
        embedder=hf_embedder,
        client=qdrant_client,
        name='vision_chain_classifier',
    )

    fast_base = vc.UltralyticsDetector(
        model_family="YOLO",
        model_weights="yolov8n.pt",
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
        }
    )

    dir_path = "/home/ubuntu/VisionChain/src/demo/bottles_dataset/data"
    file_paths = [
        os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path)
    ]

    if limit:
        file_paths = file_paths[:limit]

    # Could also call this a model chain???
    model = vc.ModelChain(
        [
            FastBase(model=fast_base, name="fast_base"),
            AccurateFallback(model=grounded_sam, name="grounded_sam"),
            Classifier()
        ],
        log_level="verbose",
    )

    dataset = fo.Dataset.from_images(file_paths)
    dataset = get_predictions(dataset, model)

    session = fo.launch_app(dataset, remote=True, address="0.0.0.0", desktop=True)
    session.wait()


if __name__ == "__main__":
    typer.run(main)
