from dataclasses import dataclass
from typing import List, Tuple
from rich import print
import cv2
import numpy as np
from typing import List

@dataclass
class Predictions: 
    labels: List[str]
    scores: List[float]
    boxes: List[List[int]]

EmptyPredictions = Predictions(
    scores = [],
    boxes = [[]],
    labels = [],
)


@dataclass
class Colour:
    upper: Tuple[int]
    lower: Tuple[int] 

@dataclass
class ColourStore: 
    black: Colour
    white: Colour
    blue: Colour
    red: Colour
    blue: Colour
    red: Colour
    green: Colour
    yellow: Colour
    brown: Colour


colour_store = ColourStore(
    black = Colour(lower=(0, 0, 0), upper=(180, 255, 30)), # https://stackoverflow.com/questions/25398188/black-color-object-detection-hsv-range-in-opencv#:~:text=For%20black%20and%20white%20colors,200%20to%20255%20for%20white.
    white = Colour(lower=(0, 0, 200), upper=(180, 255, 255)), # https://stackoverflow.com/questions/25398188/black-color-object-detection-hsv-range-in-opencv#:~:text=For%20black%20and%20white%20colors,200%20to%20255%20for%20white.
    blue = Colour(lower=(100,150,0), upper=(140,255,255)),
    red = Colour(lower=(155,25,0), upper=(179,255,255)),
    green = Colour(lower=(36, 25, 25), upper=(70, 255,255)), # https://stackoverflow.com/questions/47483951/how-to-define-a-threshold-value-to-detect-only-green-colour-objects-in-an-image
    yellow = Colour(lower=(22, 93, 0), upper=(45, 255, 255)), 
    brown = Colour(lower=(0, 100, 20), upper=(10, 255, 255)), # https://stackoverflow.com/questions/31760302/detect-brown-colour-object-using-opencv 
)

@dataclass
class ColourModel:
    """
    Way to create heuristics based on colour range.
    """
    min_width: int = 40
    min_height: int = 40

    def predict(self, image_path: str):
        try:
            # Load the image
            image = cv2.imread(image_path)

            # Convert the image to the HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            boxes = []
            labels = []
            scores = []
            for colour_name in list(colour_store.__dict__.keys()):

                colour = getattr(colour_store, colour_name)
                # Define the lower and upper bounds for white color
                lower_value = np.array(colour.lower)
                upper_value = np.array(colour.upper)

                # Create a mask for white regions
                mask = cv2.inRange(hsv, lower_value, upper_value)

                # Find contours in the mask
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # Draw bounding rectangles around white regions
                bounding_boxes = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if (w > 20) and (h > 20):
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        if (w > self.min_width) or (h > self.min_height):
                            bounding_box = [x, y, x + w, y + h]
                            bounding_boxes.append(bounding_box)

                            boxes.append(bounding_box)
                            labels.append(colour_name)
                            scores.append(1)

                
            if len(bounding_boxes) != 0:
                return Predictions(
                    scores=scores,
                    labels=labels,
                    boxes=boxes,
                )
            else:
                return EmptyPredictions

        except Exception as e:
            print(e)
            return EmptyPredictions
