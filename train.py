import torch
from detecto.core import Model, Dataset
from detecto.visualize import show_labeled_image
from detecto.utils import xml_to_csv
from object_detection.utils import label_map_util  # part of tensorflow?
import pandas as pd
import numpy as np
import os, sys, json
from experiments import Experiments
import new_experiment
klocki_home = os.getcwd()  # os.path.expanduser('~/klocki/')
#expt_name = sys.argv[1]
# CONFIG_FILE = 'config.json'
# f = open(CONFIG_FILE, 'r')
# config = json.load(f)
exps = Experiments()



# *** actual train ***
# expt_name
# *** end of "params"
def run(expt_name):
    expt_dir = klocki_home + 'experiments/' + expt_name
    # end of 'params'
    label_file = expt_dir + '/label_map.pbtxt'
    label_map = label_map_util.get_label_map_dict(label_map_util.load_labelmap(label_file))
    classes = [*label_map]
    batch, pp, template = pd.read_csv(expt_dir + 'pipeline.csv')['params'].tolist()
    datasets = []
    for source in pd.read_csv('data/batch_sources.csv')['source']:
        dataset_dir = klocki_home + 'data/' + pp + '/' + source
        ann_file = dataset_dir + '.annotations.csv'
        f_file = dataset_dir + '.files.csv'

        # *** convert labels ***
        # xml_to_csv(dataset_dir, ann_file)
        df = pd.read_csv(ann_file).astype(str)
        filtered_df = df[df['class'].isin(classes)]
        df = filtered_df.iloc[:, :-1]
        df['image_id'] = np.arange(len(filtered_df))
        df['class'] = df['class'].apply(lambda x: 'c' + x)
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
        df.to_csv(expt_dir + '/' + source + '.csv', index=False)

        # *** get dataset ***
        d = Dataset(os.path.abspath(expt_dir + '/' + source + '.csv'), os.path.abspath(dataset_dir))
        datasets.append(d)
    dataset = torch.utils.data.ConcatDataset(datasets)
    # image, targets = dataset[1]
    # show_labeled_image(image, targets['boxes'], targets['labels'])
    # model = Model([*label_map], device=torch.device('cpu'))
    classes = ['c' + c for c in classes]
    model = Model(classes)  #, model_name='fasterrcnn_mobilenet_v3_large_fpn')
    print(classes)
    model.fit(dataset, verbose=False)
    model.save(expt_dir + '/model.pth')
    return 0


if __name__ == '__main__':
    batch, pp, template = sys.argv[1], sys.argv[2], sys.argv[3]
    expt_name = '_'.join([batch, pp, template])
    new_experiment.create(exps, expt_name, batch, pp, template)
    run(expt_name)