## Active Learning 

Active learning is the problem of choosing what datapoints to label in order to most improve model performance. In most cases it boils down to trying to 
find cases where the model is likely to fail, without requiring human input. 

Most research is focused on offline approaches, taking a large sample of unlabelled images, and ranking them, but often the most valuable 
application of Active Learning is in edge deployments, with limited internet bandwidth to send data back to the cloud ready for re-training,
and limited computational resource to run the algorithm on. 

### Online Approaches

* Frame-Frame Jitter.
* Disagreement between constituents of an ensemble.
* Uncertainty based (range of scores close to the class threshold).


