from vision_chain import (
    Predictions,
)

@dataclass
class PostprocessingHook: 
    
    def run(self, predictions: Predictions) -> Predictions:
        pass


@dataclass
class Thresholding(PostprocessingHook): 
    """
    """
    def run(self, predictions: Predictions) -> Predictions:


@dataclass
class ShapeFilter(PostprocessingHook): 
    class_name: str
    min_width: int = None
    max_width: int = None 
    min_height: int = None 
    max_height: int = None

    def run(self, predictions: Predictions) -> Predictions: 
        


@dataclass 
class Postprocessor: 
    hooks: List[PostprocessingHook] = None
    
    def run(self, predictions: Predictions) -> Predictions:
        
        for hook in self.hooks: 
            predictions = hook.run(predictions)

        return predictions
