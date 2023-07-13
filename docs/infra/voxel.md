# Fiftyone Tips, Tricks and Advice 

Fiftyone is excellent for the fact that it provides you with some structure to build other tooling on top of. 
But it's very slow for what it's 'needed' for. This is any case where you chuck it in a pipeline to talk 
to CVAT. The CVAT API is a real pain for simple problems like creating a task, Voxel51 vastly simplifies this 
but you want to already be running it on a small dataset (e.g. run the filters before you reach Voxel51).

It should be used for the interactive UI in notebooks from time to time (e.g. fixing a known problem in the training dataset).
But wherever possible we should build our own tooling to replace it.

### Why not train on top of a Voxel51 dataset and why not use Voxel more? 

1. Consistently hit problems with 'duplicate key' when trying to run distibuted training. They didn't have any 
real answers when I asked about this in the slack channel. Distributed training is absolutely needed to train 
our models in a reasonable time frame.

2. We change our datasets very often, this requires loading from disk and merging it in with a previous persistent
voxel51 dataset saved in Mongo. This is a very slow process, and to need to do this before training everytime 
makes no sense.

3. Introduces 'state' to pipelines in a way which confuses instead of simplifying things, requires a lot more 'cleanup'
to make a pipeline idempotent than it does without Voxel51 as an intermediate. An example, I was working on the pipeline
to send data from prod BigQuery table to CVAT (via some dataframe filters). This had failed a few times so there were
some voxel51 datasets with the same names that needed to be created. I deleted those. When I went to repeat the pipeline 
I hit:
```
ValueError: Dataset 'mun_cardboard_2023-04-19' is deleted
```
This was happening when voxel51 tried to create a dataset with that name (not access it), that is to say, it stores 
the state of deleted datasets. CVAT names are derived from thet names of Voxel51 datasets so in order to get the pipeline
working I had to remove and restart the docker container in which the MongoDB instance for Voxel51 runs. 

4. We could have 'leant in' to voxel51, but it costs ~ 1200K per month, and does not offer that amount of value.

### Filter for data from a given day

```python 
view = dataset.match(F("filepath").contains_str("2023-04-05"))
```

### Replace object labels in a subset

```python 
# add new labels to dataset
print(dataset.count_values('detections.detections.label'))

for sample in tqdm(dataset.select(sorted_dataset.values('id'))):
    for detection in sample.detections.detections: 
        if detection['label'] == 'Ferromagnetic Metals':
            detection['label'] = 'Aluminum Can'
    sample.save()
    
print(dataset.count_values('detections.detections.label'))
```

### Add an ID to monitor progress (when scrolling through images). 

```python 
sorted_dataset.set_values('int_id', list(range(0, len(sorted_dataset))))
```

### Sort by purity of baler data

```python 
dataset.add_sample_field("num_predictions", fo.IntField)
view = dataset.set_field("num_predictions", F("detections.detections").length())

sorted_dataset = view\
    .match(F("detections.detections").length() > 3)\
    .filter_labels("detections", F("label").is_in(['Colored Paper']))\
    .sort_by(F("detections.detections").length() / F('num_predictions'), reverse=True)
```

### Retrieve the frames which contain a given class over a given area. 

```python
dataset.add_sample_field("num_predictions", fo.IntField)
view = dataset.set_field("num_predictions", F("detections.detections").length())

sorted_dataset = view\
    .match(F("detections.detections").length() > 3)\
    .filter_labels("detections", F("label").is_in(['Colored Paper']))\
    .sort_by(F("detections.detections").length() / F('num_predictions'), reverse=True)
```

### Remove all objects from selected frames (for improving foreground - background detection)

```python
# add new labels to dataset
for sample in dataset.select(session.selected):
    sample['detections'] = fo.Detections(detections=[])
    sample.save()
```

### How to add thumbnails to improve loading speeds of the UI

```python
import fiftyone as fo
import fiftyone.utils.image as foui
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")

# View the dataset's current App config
print(dataset.app_config)

# Generate some thumbnail images
foui.transform_images(
    dataset,
    size=(480, 270),
    output_field="thumbnail_path",
    output_dir="/tmp/thumbnails",
)

# Modify the dataset's App config
dataset.app_config.media_fields = ["filepath", "thumbnail_path"]
dataset.app_config.grid_media_field = "thumbnail_path"
dataset.save()  # must save after edits

session = fo.launch_app(dataset)
```

