import json 
import numpy as np
import pandas as pd
from utils import *
import pprint
from collections import Counter
from mean_average_precision import MetricBuilder


def match_results(gt_df, pred_df, max_score_th, max_iou_th):
    pred_df['gt_i'] = np.nan
    results = []
    results_disc = []
    files = {}  # annotated files
    for i, gt in gt_df.iterrows():
        found = False
        files[gt['filename']] = True
        for j, pred in pred_df[
                (pred_df['filename'] == gt['filename'])
                &(pred_df['class'] == gt['class'])
                &(pred_df['score'] >= max_score_th)
                ].iterrows():
            gt_bbox = [gt['xmin'], gt['ymin'], gt['xmax'], gt['ymax']];
            pred_bbox = [pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax']];
            iou_value = iou(gt_bbox, pred_bbox)
            if iou_value > max_iou_th :
                found = True
                pred_df.at[j, 'gt_i'] = j
                results.append({
                    'filename': pred['filename'],
                    'source': pred['source'],
                    'class': pred['class'],
                    'iou': iou_value,
                    'tp': 1
                    })
                break
        if not found:  # existing brick not found in pred - not predicted, or are < thresholds (todo: analyze those that are below threshold, keep them here - they are true TF)
            # existing brick not found in pred - not predicted, or are < thresholds
            results.append({
                'filename': gt['filename'],
                'source': gt['source'],
                'class': gt['class'],
                'fn': 1
                })

    # here below we take only those predicted that were not used above, not from proper class or are < thresholds (due to in pred_df.at[j, 'gt_i'] = j)
    # We need distinction here: 
    # 1. Those that are < threshold are not FP, they are not predictions, they do not exist (in real application we discard them, we do not show them to the user) - we can store them for stats, but not base our model optimisation on them
    # 2. Those that are > threshold and are wrong class - those are real FP, and we want to lower the value
    # By manipulating on the thresholds, we will then be able to balance the numbers of TP and FP:
    # thresholds up -> TP down but FP also down! (since TP and FP are both dependant of thresholds)
    for i, pred in pred_df[
            pred_df.gt_i.isnull()
            &(pred_df['score'] >= max_score_th)
            ].iterrows():
        if pred['filename'] in files: # files = names existing in gt, data annotations
            # leave alone files without annotations
            results.append({
                'filename': pred['filename'],
                'source': pred['source'],
                'class': pred['class'],
                'fp': 1
                })

    return results

def map(ann_df, pred_df):
    cat_to_id = {
    "BRICK_1X2" : 0,
    "BRICK_1X4" : 1,
    "BRICK_2X2" : 2,
    "BRICK_2X4" : 3,
    "BRICK_2X6" : 4,	# mean_average_precision requires classes to start at
    "PLATE_1X2" : 5,	# 0 so bricks correspond to numbers 1 lower than usual
    "PLATE_1X4" : 6,
    "PLATE_2X2" : 7,
    "PLATE_2X4" : 8,
    "PLATE_2X6" : 9
    }

    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=10)

    filenames = pred_df.filename.unique()

    for filename in filenames:
        ann_slice = ann_df[ann_df['filename'] == filename]
        ann_slice = ann_slice.replace({'class' : cat_to_id})
        ann_slice.drop(['filename', 'width', 'height'], axis='columns', inplace=True)	
        cols = ['xmin', 'ymin', 'xmax', 'ymax', 'class']
        ann_slice = ann_slice.reindex(columns=cols)
        ann_slice['difficult'] = 0
        ann_slice['crowd'] = 0
        ann_slice = ann_slice.to_numpy()
	
        pred_slice = pred_df[pred_df['filename'] == filename]
        pred_slice = pred_slice.replace({'class' : cat_to_id})
        pred_slice.drop('filename', axis='columns', inplace=True)
        pred_slice = pred_slice.to_numpy()
		
        metric_fn.add(pred_slice, ann_slice)
	
    return metric_fn


