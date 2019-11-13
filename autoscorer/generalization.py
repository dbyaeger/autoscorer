#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 08:49:51 2019

@author: danielyaeger
"""
import re
from autoscorer.autoscorer import Autoscorer
from autoscorer.evaluator import Evaluator
import pickle
from pathlib import Path
import numpy as np

def generalization(data_path = Path('/Users/danielyaeger/Documents/processed_data/processed'),
                 partition_file_name = 'data_partition.p', partition_mode = "test"):
    balanced_accuracies = []
    interepoch_agreements = []
    diagnosis_accuracy = []
    with data_path.joinpath(partition_file_name).open('rb') as fh:
        ID_list = list(pickle.load(fh)[partition_mode])
        ID_list = [x for x in ID_list if len(re.findall('[0-9A-Z]', x)) > 0]
    ID_list = [s.split('_')[0] for s in ID_list]
    ID_list = list(set(ID_list))
    for ID in ID_list:
        scorer = Autoscorer(ID = ID, data_path = data_path,
                 f_s = 10,
                 t_amplitude_threshold = 4,
                 t_continuity_threshold = 30, 
                 p_mode = 'quantile', 
                 p_amplitude_threshold = 2, 
                 p_quantile = 0.9,
                 p_continuity_threshold = 8, 
                 p_baseline_length = 120, 
                 ignore_hypoxics_duration = 15,
                 return_seq = True, return_concat = True, 
                 return_tuple = False, 
                 phasic_start_time_only = True,
                 return_multilabel_track = True,
                 verbose = False)
        predictions = scorer.score_REM()
        annotations = scorer.get_annotations()
        for seq in predictions:
            evaluator = Evaluator(predictions = predictions[seq], annotations = annotations[seq], 
                 sequence = True, single_ID = True, single_subseq = True, verbose = False)
            balanced_accuracies.append(evaluator.balanced_accuracy_signals())
            interepoch_agreements.append(evaluator.cohen_kappa_epoch())
            diagnosis_accuracy.append(evaluator.accuracy_score_diagnosis())
            print(f"For ID: {ID}\tsubseq: {seq}\tbalanced_accuracy: {balanced_accuracies[-1]}\tinter_epoch_agreement: {interepoch_agreements[-1]}\tdiagnosis_accuracy: {diagnosis_accuracy[-1]}")
    print(f'Mean balanced accuracy: {np.mean(np.array(balanced_accuracies))}')
    print(f"Standard deviation balanced accuracy: {np.std(np.array(balanced_accuracies))}")
    print(f"Mean Inter-epoch agreement: {np.mean(np.array(interepoch_agreements))}")
    print(f"Standard deviation inter-epoch agreement: {np.std(np.array(interepoch_agreements))}")
    print(f"Mean diagnostic accuracy: {np.mean(np.array(diagnosis_accuracy))}")
    print(f"Standard deviation diagnostic accuracy: {np.std(np.array(diagnosis_accuracy))}")
    return np.array(balanced_accuracies), np.array(interepoch_agreements), np.array(diagnosis_accuracy)       
        