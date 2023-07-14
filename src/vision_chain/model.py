from dataclasses import dataclass
import onnxruntime as ort
import numpy as np

from vision.tools import (
    Preprocessor,
    Predictions,
    Postprocessor,
)

@dataclass
class Model: 
    model_path: str = '../deployment/exports/onnx_static/end2end.onnx'
    inference_session = None
    preprocessor = Preprocessor()
    postprocessor: Postprocessor = None
    provider = 'CUDAExecutionProvider'

    def __post_init__(self):

        if self.inference_session is None:
            # providers must be specified since ORT 1.9
            self.inference_session = ort.InferenceSession(
                self.model_path,
                providers=[self.provider],
            )

    def predict(self, file_path: str) -> Predictions:
        
        image_array = self.preprocessor.run(file_path)
        batch_onnx_output = self.inference_session.run(None, {'input': image_array})
        predictions = Predictions(
            labels = batch_onnx_output[1].tolist(),
            scores = batch_onnx_output[0][:,-1].tolist(),
            boxes = batch_onnx_output[0][:,:-1].tolist(),
        )
        predictions = self.postprocessor.run(predictions)
                
        return predictionss
