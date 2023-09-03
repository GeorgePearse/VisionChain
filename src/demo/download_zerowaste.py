import os
import os.path as osp
import fiftyone as fo
import typer

def filter_dataset(ds):
    F = fo.ViewField
    bbox_area = F("bounding_box")[2] * F("bounding_box")[3]
    clean_view = ds.filter_labels("ground_truth", bbox_area > 0)
    clean_dataset = clean_view.clone()
    print("loaded {} valid samples".format(len(clean_dataset)))
    return clean_dataset

def load_dataset(dataset_dir, dataset_type):
    dataset = fo.Dataset.from_dir(dataset_dir=dataset_dir, dataset_type=dataset_type)
    return filter_dataset(dataset)

def download_zerowaste():
    data_dir = 'data'
    folder_dir = 'splits_final_deblurred'
    data_folder_dir = osp.join(data_dir, folder_dir)
    train_dir = osp.join(data_folder_dir, 'train')
    val_dir = osp.join(data_folder_dir, 'val')
    test_dir = osp.join(data_folder_dir, 'test')

    # make sure data directory is set up
    if not osp.exists(data_dir):
        os.makedirs(data_dir)

    # download the zipped file if necessary
    zipped_file = 'zerowaste-f-final.zip'
    zipped_file_dir = osp.join(data_dir, zipped_file)
    if not osp.exists(zipped_file_dir):
        print('downloading zipped file...')
        url = 'https://zenodo.org/record/6412647/files/zerowaste-f-final.zip'
        os.system("wget -nv " + url + ' -P ' + data_dir)

    # unzip the zipped file if necessary
    if not osp.exists(data_folder_dir):
        print('unzipping files...')
        os.system('unzip -q ' + zipped_file_dir + ' -d ' + data_dir)

    assert osp.exists(train_dir) and osp.exists(val_dir) and osp.exists(test_dir)
    print('data preparation ready!!!')

    return train_dir, val_dir, test_dir


def fetch():
    train_dir, val_dir, test_dir = download_zerowaste()
    classnames = ["background", "rigid_plastic", "cardboard", "metal", "soft_plastic"]

    dataset_type = fo.types.COCODetectionDataset

    train_dataset = load_dataset(dataset_dir=train_dir, dataset_type=dataset_type)
    val_dataset = load_dataset(dataset_dir=val_dir, dataset_type=dataset_type)
    test_dataset = load_dataset(dataset_dir=test_dir, dataset_type=dataset_type)
    
    train_dataset.add_collection(val_dataset)
    train_dataset.add_collection(test_dataset)
    
    train_dataset.export(
        export_dir='zerowaste-demo',
        dataset_type=fo.types.COCODetectionDataset,
    )

if __name__ == '__main__':
    typer.run(fetch)
