import typer 
import json
import os
import shutil

import funcy
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import numpy as np


def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def filter_images(images, annotations):

    annotation_ids = funcy.lmap(lambda i: int(i['image_id']), annotations)

    return funcy.lfilter(lambda a: int(a['id']) in annotation_ids, images)


def main(
        annotations_path: str = 'data.json',
        train: str = 'coco_split/train.json',
        test: str = 'coco_split/val.json',
        split: float = 0.8,
    ):
    """
    Just splits a coco file according to a split fraction
    """
    if os.path.exists('coco_split'):
        shutil.rmtree('coco_split')

    os.mkdir('coco_split')

    with open(annotations_path, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco['info']
        licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        number_of_images = len(images)

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        X_train, X_test = train_test_split(images, train_size=split)

        anns_train = filter_annotations(annotations, X_train)
        anns_test=filter_annotations(annotations, X_test)

        save_coco(train, info, licenses, X_train, anns_train, categories)
        save_coco(test, info, licenses, X_test, anns_test, categories)

        print("Saved {} entries in {} and {} in {}".format(len(anns_train), train, len(anns_test), test))
        


if __name__ == "__main__":
    typer.run(main)
