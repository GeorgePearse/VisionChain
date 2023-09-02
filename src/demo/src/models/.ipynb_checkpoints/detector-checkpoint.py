from dataclasses import dataclass

@dataclass
class Detector:
    cropper: Cropper
    classifier: Classifier

    def predict(self: file_path) -> List[DetectionPrediction]:
        detection_predictions = []
        
        crops = self.cropper(file_path)
        for crop in crops:
            classification_prediction = self.classifier(crop)
            detection_prediction = DetectionPrediction(
                name=classification_prediction.name
                score=classification_prediction.score,
                bbox=crop.bbox,
            )
            detection-predictions.append(detection_predictions)

        return detection_predictions


@dataclass
class EnsembleDetector:
    """
    Possible there should be an EnsembleClassifier 
    too? 
    """
    detectors: List[Detector]

    def train(self, train_dataset: fo.Dataset):
        for detector in self.detectors:
            detector.train(train_dataset)

    def predict(self, query_image_path: str) -> Prediction:
        predictions = []
        for detector in self.detectors:
            prediction = detector.predict(query_image_path)

        return self.aggregate(predictions)

    @staticmethod
    def aggregate(predictions: List[Prediction], method: str = 'majority') -> Prediction:
        if method == 'majority':
            neighbour_labels = [prediction.name for prediction in predictions]
            return max(set(neighbour_labels), key=neighbour_labels.count)

        if method == 'weighted':                 
            raise Exception('Weighted aggregate not yet implemented')

    @staticmethod
    def select(predictions: List[Prediction], method: str) -> str: 
        """
        Active Learning by disagreement.

        How many of the neighbours are the same? 
        """
        # makes no sense because sometimes none of the neighbours 
        # will be the same, but all will be of the same 
        # class
        if method == 'count_neighbour_overlap': 


    def test(self, test_dataset: fo.Dataset):
        """
        Just highjack Voxel51's eval. 
        """
        for file_path in tqdm(test_dataset.values('filepath')):
            prediction = self.predict(file_path)