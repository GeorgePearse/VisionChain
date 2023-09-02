from dataclasses import dataclass
from typing import List

@dataclass
class ClassificationPrediction: 
    name: str
    score: float


@dataclass
class DetectionPrediction: 
    """
    Could have separate localisation 
    and classification score?
    """
    name: str
    score: float
    bbox: List[int]


@dataclass
class SegmentationnPrediction: 
    """
    Could have separate localisation 
    and classification score?
    """
    name: str
    score: float
    segmentation: List[int]
    
