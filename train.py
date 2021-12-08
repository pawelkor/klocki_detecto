from detecto.core import Model, Dataset
from detecto.visualize import show_labeled_image
from detecto.utils import xml_to_csv
from object_detection.utils import label_map_util  # part of tensorflow?
import pandas as pd
import numpy as np
import os, sys
klocki_home = os.path.expanduser('~/klocki/')
expt_name = sys.argv[1]
expt_dir = klocki_home + 'experiments/' + expt_name
if os.path.isdir(expt_dir):
    if '-o' in sys.argv:
        os.rmdir(expt_dir)
    else:
        print("Experiment already exists.")
        exit(1)
os.mkdir(expt_dir)
dataset_dir = klocki_home + 'data/' + sys.argv[2] + '/' + sys.argv[3]
ann_file = dataset_dir + '.annotations.csv'
f_file = dataset_dir + '.files.csv'
label_file = klocki_home + 'data/label_map.pbtxt'
# end of 'params'

label_map = label_map_util.get_label_map_dict(label_map_util.load_labelmap(label_file))
# xml_to_csv(dataset_dir, ann_file)
df = pd.read_csv(ann_file).astype(str)
classes = [*label_map]
boolean_series = df['class'].isin(classes)
filtered_df = df[boolean_series]
df = filtered_df.iloc[:, :-1]
df['image_id'] = np.arange(len(filtered_df))
df['class'] = df['class'].apply(lambda x: 'c' + x)
classes = ['c' + c for c in classes]
df.insert(1, 'width', 0)
df.insert(2, 'height', 0)
files_df = pd.read_csv(f_file, index_col='filename')
# duplications in 20K_cut_photos_no_Brick XMLs -> 20K_cut_photos_no_Brick.files.csv -> here
# files_df.drop_duplicates(subset=['filename'], inplace=True)
files_df = files_df[~files_df.index.duplicated()]
files_df.sort_index(inplace=True)
for i, ann in df.iterrows():
    print(ann['filename'])
    print(files_df.loc[ann['filename'], 'width'])
    df.at[i, 'width'] = files_df.loc[ann['filename'], 'width']
    print(files_df.loc[ann['filename'], 'width'])
    df.at[i, 'height'] = files_df.loc[ann['filename'], 'height']

df.to_csv(expt_dir + '/mylabels.csv', index=False)

dataset = Dataset(os.path.abspath(expt_dir + '/mylabels.csv'), os.path.abspath(dataset_dir))
# image, targets = dataset[1]
# show_labeled_image(image, targets['boxes'], targets['labels'])
# model = Model([*label_map], device=torch.device('cpu'))
model = Model(classes)
print(classes)
# model.fit(dataset, verbose=True)
model.fit(dataset, verbose=False)
model.save(expt_dir + '/' + expt_name + '.pth')

