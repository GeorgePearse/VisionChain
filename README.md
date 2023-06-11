# VisionChain
Framework to support common preprocessing and postprocessing steps, along with computer vision heuristics built with opencv, and voting systems ontop of those heuristic-model combos. 

## Tools it would lean on: 

* sklearn.ensemble
* supervision
* dataclasses
* jupyter-scatter ? 
* pandas 
* CocoFrame (pending completion)
* Norfair or an object tracking framework
* human-learn https://koaning.github.io/human-learn/ with UMAP?  -> size based filters, colour based filters. 
* Maybe something like hamilton for clean pandas transformations.
* Definitely hugging-face.
* Top X style queries for Grounding Dino or Sam. e.g. state 3 oranges and retain the top 3 orange predictions.
* QDrant positive - negative 
* Quaterion for similarity learning of the classifier in a CompositeDetector

Check calmcode tutorials.

Would need to actually integrate models as they came. 
