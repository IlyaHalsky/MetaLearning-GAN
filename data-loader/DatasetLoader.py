import openml
import pandas as pd
import csv
import shutil
import os

from openml import OpenMLDataset


def list_datasets():
    return openml.datasets.list_datasets()


def load_dataset(no):
    return openml.datasets.get_dataset(no)


if __name__ == '__main__':
    openml_list = list_datasets()
    with open('./datasets/datasets.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(
            ['DS_ID', 'NumberOfInstances', 'NumberOfFeatures', 'NumberOfClasses', 'Target_Feature', 'DS_URL'])
        for (i, dsd) in openml_list.items():
            try:
                print(i)
                did = dsd['did']
                ds: OpenMLDataset = load_dataset(did)
                file = ds.data_file
                filename, file_extension = os.path.splitext(file)
                shutil.copy2(file, f'D:\DataSets/{did}{file_extension}')
                features = ds.features
                target_name = ds.default_target_attribute
                if target_name is not None:
                    target_id = (-1, '')
                    for (_, f) in features.items():
                        if f.name == target_name:
                            target_id = (f.index, f.data_type)
                    writer.writerow(
                        [dsd['did'], dsd['NumberOfInstances'], dsd['NumberOfFeatures'], dsd['NumberOfClasses'],
                         target_id[0],
                         ds.url])
                else:
                    print("Skipped", dsd['did'])
            except:
                print("error")
