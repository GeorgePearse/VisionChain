## Automation

Directory to begin work on automation with Dagster:

Automation of a Data Engine has 3 core components

1. Pull predictions from prod and send them to CVAT (automation/filter_and_send_to_cvat.py).
2. Update labels in S3 (e.g. pull reviewed predictions from CVAT, merge them with pyodi and return) (automation/update_training_labels.py).
3. Trigger training (evaluation via MLTest will run as part of the same pipeline).
     - This will execute an AWS Batch job.
     - This job will have several parts (training, checkpoint selection, Lakera's MLTest evaluation)

These steps need to run as an independent loop for each model (municipals, CnD, Covant, ENS) The development of pipelines automated with Dagster will also help to increase the visibility into the ML / Data Engineering processes required to improve our models.

Additional Potential Components:

* Queries to seek the most likely labelling errors WITHIN the existing training set. This is the hardest part to operationalise.


Dagster can be developed with:

```
dagster dev -f repo.py -h 0.0.0.0
```

Next step up to leave a functional, but crude deployment behind is to run the above in a tmux session so that it continues.

## 1. Pull predictions from BigQuery and send them to CVAT. 

Current problems: 

1. Does not include empty frames because it pulls from a table of objects.
2. Can be a pain to develop because of 51's insistence on unique naming (even when the other dataset has been deleted).

Steps to resolve: 

1. Additional function and SQL query to pull frame_uris which contain no objects (potentially just an independent pipeline).
2. Remove Voxel51 from the send_to_cvat function in FrameBatch.
3. Solidify the concept of a 'filter' and the relationship between filters, cameras, models and clients (lots of dataclasses). Potential filters: 
    1. Filter by size of detection and class name (e.g. "Other PET" over a certain prediction size).
    2. Overlap between predictions of two different classes (if NMS is not applied in postprocessing).
4. How to run for multiple clients / models. 

## 2. Update training labels. 

This ones fairly simple. Pattern matching for CVAT tasks -> S3 directory is achieved via a SQL query with duckdb over task names.
It is all at the model level, e.g. 'municipals'. 

Pipeline just needs to know how to filter the tasks, and where to send the data. The same sensor can easily be responsible for 
any number of models (can just iterate over them checking if the corresponding SQL query over CVAT tasks returns any that are completed).

## 3. Triggering training with AWS Batch

### Dev Instance

AMI ami-02681d26eb2d77ef7 from instance i-0aa487d2d6ab15e03 <- pending
ami-0925355a3c8b114ee <- successfully built before

have to setup a 'compute environment'

https://docs.aws.amazon.com/batch/latest/userguide/create-compute-environment.html#create-compute-environment-managed-ec2


### Prod Instance (4x v100)
