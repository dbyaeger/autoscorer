#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 21:37:36 2019

@author: danielyaeger
"""
from autoscorer.autoscorer import Autoscorer
from autoscorer.evaluator import Evaluator
import pickle

def make_evaluation_file():
    path_to_IDs = '/Users/danielyaeger/Documents/Modules/sleep-research-ml/data/supplemental/test_predictions_cnn_multi_channel_window_20.p'
    results = {}                   
    # Open file to find unique IDs
    with open(path_to_IDs, 'rb') as fh:
        data = pickle.load(fh)
    unique_IDs = set([x.split('_')[0] for x in data.keys()])
    for ID in unique_IDs:
        scorer = Autoscorer(ID = ID, t_amplitude_threshold = 1,t_continuity_threshold = 20, p_mode = 'quantile',
                            p_amplitude_threshold = 4, p_quantile = 0.5, p_continuity_threshold = 5, return_seq = True,
                            return_concat = True, return_multilabel_track = True, 
                            return_matrix_event_track = True, return_tuple = False)
        predictions = scorer.score_REM()
        annotations = scorer.get_annotations()
        for subseq in predictions.keys():
            num = subseq.split('_')[-1]
            file_name = ID + '_' + num
            if file_name in data.keys():
                results[file_name] = {}
                results[file_name]['targets'] = annotations[subseq]
                results[file_name]['predictions'] = predictions[subseq]
                evaluator = Evaluator(predictions = predictions[subseq], annotations = annotations[subseq],
                                      sequence = True, single_ID = True, single_subseq = True)
                results[file_name]['evaluation'] = {}
                results[file_name]['evaluation']['balanced_accuracy'] = evaluator.balanced_accuracy_signals()
                results[file_name]['evaluation']['confusion_matrix'] = evaluator.confusion_matrix_signals()
                print(results[file_name])
    with open('/Users/danielyaeger/Documents/Modules/sleep-research-ml/data/supplemental/test_predictions_rule_based_scorer.p', 'wb') as fout:
        pickle.dump(obj = results, file = fout)
    return results
    