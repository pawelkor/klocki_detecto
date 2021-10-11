import os
import sys
import pandas as pd
from utils import *
from tabulate import tabulate
from collections import namedtuple
from pathlib import Path
from metrics import match_results
import json
from object_detection.utils import config_util

def get_gt(directory, sources):
    ann_df = pd.DataFrame()
    file_df = pd.DataFrame()
    source_list = os.listdir(directory)
    for source in source_list:
        source_path = directory + source
        if not os.path.isdir(source_path) or not source in sources:
              continue
        f = pd.read_csv(source_path + '.files.csv', index_col='filename', dtype={ 'for_training': pd.Int64Dtype() }) #.head()
        f["source"] = source
        file_df = file_df.append(f)
        a = pd.read_csv(source_path + '.annotations.csv', dtype={ 'class': pd.StringDtype() })
        ann_df = ann_df.append(a)
    return file_df, ann_df

def find_display_name(name, label_map_path):
    label_map = label_map_util.load_labelmap(label_map_path)
    for i in label_map.item:
        if i.name == name:
            return i.display_name
    return name


def main(args):
    files_df, ann_df = get_gt(args.gt_dir, args.sources)
    if files_df.empty or ann_df.empty:
        print("No data")
        exit()
    raw_gt_df = pd.merge(ann_df, files_df, on=['filename'])
    trained_gt_df = raw_gt_df[
        raw_gt_df['source'].isin(args.trained_sources)
        & raw_gt_df['for_training'] == 1
    ]
    eval_gt_df = raw_gt_df[raw_gt_df['for_training'] == 0]
    raw_pred_df = pd.read_csv(args.prediction_file, dtype={ 'class': pd.StringDtype() })
    field = None
    values = []
    if args.constraint:
        [field, values] = args.constraint.split('=')
        values = values.split(',')
        eval_gt_df = eval_gt_df[eval_gt_df[field].isin(values)]
        raw_pred_df = raw_pred_df[raw_pred_df[field].isin(values)]
        trained_gt_df = trained_gt_df[trained_gt_df[field].isin(values)]

    results = match_results(eval_gt_df, raw_pred_df, args.max_score_th, args.max_iou_th)

    if not args.constraint:
        to_add_df = pd.DataFrame(results, columns=['filename', 'source', 'class', 'iou', 'tp', 'fp', 'fn'])
        to_add_df['pp'] = params['pp']
        to_add_df['batch'] = params['batch']
        to_add_df['exp'] = params['exp']
        global_results_file = 'global_stats/results.csv'
        if (os.path.isfile(global_results_file)):
            global_results = pd.read_csv(global_results_file)
            global_results = global_results[global_results['exp'] != params['exp']]
        else:
            global_results = pd.DataFrame()
        global_results = global_results.append(to_add_df)
        global_results.to_csv(global_results_file, index=False)
    checked_pred_df = pd.DataFrame(results, columns=['filename', 'source', 'class', 'iou', 'tp', 'fp', 'fn'])
    to_pivot = ['class', 'source']
    for pivot in to_pivot:
        metrics_df = pd.pivot_table(checked_pred_df, index=[pivot],
                                 values=['tp', 'fp', 'fn'], aggfunc=np.sum, fill_value=0)
        train_metrics_df = pd.pivot_table(trained_gt_df, index=[pivot],
                                          values=['for_training'], aggfunc=np.sum,
                                          fill_value=0)
        if train_metrics_df.empty:
            train_metrics_df['for_training'] = None
        joined_df = metrics_df.join(train_metrics_df, how='outer')
        joined_df = joined_df.fillna(0)
        joined_df = joined_df.astype({'fn': 'int64', 'fp': 'int64', 'tp': 'int64', 'for_training': 'int64'})
        if not args.constraint:
            global_summary_file = f'global_stats/{pivot}.csv'
            if(os.path.isfile(global_summary_file)):
                global_summary_df = pd.read_csv(global_summary_file)
                global_summary_df = global_summary_df[global_summary_df['exp'] != params['exp']]
            else:
                global_summary_df = pd.DataFrame()

            to_add_df = joined_df.copy()
            to_add_df['pp'] = params['pp']
            to_add_df['batch'] = params['batch']
            to_add_df['exp'] = params['exp']
            to_add_df[pivot] = to_add_df.index
            global_summary_df = global_summary_df.append(to_add_df)
            global_summary_df.to_csv(global_summary_file, index=False)

        summary = pd.Series(data={
            'for_training': joined_df['for_training'].sum(),
            'fn': joined_df['fn'].sum(),
            'fp': joined_df['fp'].sum(),
            'tp': joined_df['tp'].sum()
        }, name='ALL')
        joined_df = joined_df.append(summary)
        joined_df['precision'] = joined_df['tp']/(joined_df['tp']+joined_df['fp'])
        joined_df['recall'] = joined_df['tp']/(joined_df['tp']+joined_df['fn'])
        if pivot == 'class':
            joined_df['display'] = joined_df.index.to_series().apply(lambda x: find_display_name(x, args.label_map_path))
            display_columns = [ 'display', 'for_training', 'fn', 'fp', 'tp', 'precision', 'recall' ]
        else:
            display_columns = [ 'for_training', 'fn', 'fp', 'tp', 'precision', 'recall' ]
        print(joined_df.to_string(columns=display_columns))
        print("\n")

params = {}

klocki_home = os.path.expanduser('~/klocki/')

if(not len(sys.argv) > 1):
    print("No experiment name")
    exit()

expt_name = sys.argv[1]  # relative
pp_dir = json.load(open(klocki_home + '/config.json'))['PP_DIR'] + '/'
gt_dir = klocki_home + '/data/' + pp_dir
params['gt_dir'] = gt_dir

if (len(sys.argv) > 2):
    params['constraint'] = sys.argv[2]
    print(f"Stats only for {params['constraint']=}")
else:
   params['constraint'] = None

f = klocki_home + '/experiments/' + expt_name + '/model/pipeline.config'
if(not os.path.isfile(f)):
    print("Pipeline config not found:", f)
    exit()
configs = config_util.get_configs_from_pipeline_file(f)
#f = configs['eval_input_config'].tf_record_input_reader.input_path[0]
#params['gt_dir'] = os.path.dirname(os.path.dirname(f)) + '/'

f = configs['train_input_config'].tf_record_input_reader.input_path[0]
batch_name = os.path.basename(f)[:-13]
# batch_name = Path(expt_name).parts[-1]

params['batch'] = batch_name
batch_df = pd.read_csv(klocki_home + 'data/batch_sources.csv')
trained_sources = batch_df[batch_df['batch_name'] == batch_name].source.unique()
params['trained_sources'] = trained_sources
if trained_sources.size == 0:
    print(f"No sources found for batch {batch_name}")
    exit()
params['prediction_file'] = 'experiments/' + expt_name + '/predictions.csv'
params['max_score_th'] = 0.2
params['max_iou_th'] = 0.5
params['label_map_path'] = klocki_home + 'data/label_map.pbtxt'

sources_df = pd.read_csv(klocki_home + 'data/sources.csv')
params['sources'] = sources_df.source_name.to_numpy()
params['pp'] = pp_dir
params['exp'] = expt_name

args = namedtuple("ObjectName", params.keys())(*params.values())

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.float_format', lambda x: '%.2f' % x)

if not os.path.isdir('global_stats'):
    os.mkdir('global_stats')

if __name__ == '__main__':
    main(args)

