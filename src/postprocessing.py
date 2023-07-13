from vision_chain import (
    Predictions,
)

@dataclass
class PostprocessingHook: 
    
    def run(self, predictions: Predictions) -> Predictions:
        pass


@dataclass
class Thresholding(PostprocessingHook): 

    def run(self, predictions: Predictions) -> Predictions:

