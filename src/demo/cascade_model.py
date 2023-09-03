from model import (
    Model,
    ConditionalModel,
)
from dataclasses import dataclass
import os 

VISION_CHAIN_TRACKING = os.environ.get('VISION_CHAIN_TRACKING', True)


@dataclass
class SpeculativePrediction:
    """
    Didnt want to forget the name
    https://arxiv.org/abs/2302.01318

    This is a request, and will only 
    be fulfilled if the CascadeModel
    contains the components to do so.

    Should be aiming for a matching system. 
    Where they all pass through everything
    """
    chain_start: bool 

    confident_detections: sv.Detections

    # hit if there's a downstream classifier
    reclassify_detections: sv.Detections

    # hit if there's a downstream Detector
    repredict_whole_frame: bool

    # for Mixture of Expert models
    # only really works if first model 
    # is a classifier
    # request_model: Model

    

@dataclass
class CascadeModel(Model):
    conditional_models: List[ConditionalModel]

    def predict(self: file_path: str) -> sv.Detections:
        """
        Should the triggers be attached to the curr model, 
        or the next model? 

        Which creates the cleaner API
        """
        speculative_prediction = SpeculativePrediction(
            chain_start=True,
        )

        for conditional_model in self.conditional_models: 
            speculative_predictions = conditional_model.speculate(
                speculative_predictions
            )

        return speculative_predictions.confident_detections
                
