#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:55:52 2019

@author: danielyaeger
"""

import numpy as np
from pathlib import Path
import pickle
import re
from autoscorer.Score_All import All_Scorer
from autoscorer.evaluator import Evaluator

def greedy_grid(data_path = Path('/Users/yaeger/Documents/processed_data/processed'),
                parameters = {'t_continuity_threshold': [10,20,30,40,50], 'p_quantile': [0.3,0.5,0.7,0.9], 
                  'p_amplitude_threshold': [2,4,6,8],'p_continuity_threshold': [1,3,5,8],
                  't_amplitude_threshold': [1,2,3,4]},
                default_parameters = {'t_continuity_threshold': 10, 'p_amplitude_threshold': 4,
                  'p_continuity_threshold':1, 't_amplitude_threshold':1, 'p_quantile': 0.5}, folds = 3):
    """Uses a greedy approach to optimize the parameters of autoscorer"""
        
    # Open partition file and extract cv and train
    with data_path.joinpath('data_partition.p').open('rb') as fh:
        old_partitions = pickle.load(fh)
        
    # Convert from format ID_n.p to ID
    partitions = {}
    for partition in old_partitions:
        ID_list = list(old_partitions[partition])
        ID_list = [x for x in ID_list if len(re.findall('[0-9A-Z]', x)) > 0]
        ID_list = [s.split('_')[0] for s in ID_list]
        partitions[partition] = list(set(ID_list))
    
    # Combine train and cv into a single list
    train = partitions['train'] + partitions['cv']
    np.random.shuffle(train)
    test = partitions['test']
    
    # Compute number of sleepers in each cv fold
    fold_size = len(train)//folds
    
    # shuffle order in which parameters are optimized
    parameter_names = list(parameters.keys())
    np.random.shuffle(parameter_names)
    
    best_params = {}
    
    # Perform k-fold cross-validation to choose best parameter in greedy fashion
    for parameter in parameter_names:
        print(f'Testing {parameter}')
        best_value = 0
        best_bal_acc = 0
        for value in parameters[parameter]:
            bal_accuracy = 0
            print(f'Testing value: {value}')
            # Fill param_dict to use as parameters for Score_All
            param_dict = {parameter: value}
            for param in parameters:
                if param in best_params:
                    param_dict[param] = best_params[param]
                elif param not in best_params and param not in param_dict:
                    param_dict[param] = default_parameters[param]
            # Perform model tuning
            for fold in range(folds):
                print(f'Fold: {fold}')
                if fold < folds - 1:
                    cv = train[fold*fold_size:fold*fold_size + fold_size]
                else:
                    cv = train[fold*fold_size:]
                scorer = All_Scorer(data_path = data_path,
                     f_s = 10, t_amplitude_threshold = param_dict['t_amplitude_threshold'],
                     t_continuity_threshold = param_dict['t_continuity_threshold'], 
                     p_mode = 'quantile', p_amplitude_threshold = param_dict['p_amplitude_threshold'], 
                     p_quantile = param_dict['p_quantile'],
                     p_continuity_threshold = param_dict['p_continuity_threshold'], 
                     p_baseline_length = 120,
                     ignore_hypoxics_duration = 15, return_seq = True, 
                     return_concat = True, return_tuple = False, 
                     phasic_start_time_only = False,
                     return_multilabel_track = True,
                     use_muliprocessors = True,
                     num_processors = 4,
                     score_all_files_in_dir = False,
                     use_partition = False,
                     ID_list = cv,
                     verbose = False)
                predictions = scorer.score_all()
                annotations = scorer.get_annotations()
                evaluator = Evaluator(predictions = predictions, annotations = annotations, verbose = False)
                bal_accuracy += evaluator.balanced_accuracy_signals()
            bal_accuracy = bal_accuracy/folds
            print(f'For {parameter} with value {value}, balanced accuracy: {bal_accuracy}')
            if bal_accuracy > best_bal_acc:
                best_value = value
                best_bal_acc = bal_accuracy
        best_params[parameter] = best_value
        print(f'For parameter {parameter}, best value: {best_value}')
    # Evaluate best parameters on test set
#    scorer = All_Scorer(data_path = data_path,
#                     f_s = 10, t_amplitude_threshold = be,
#                     t_continuity_threshold = best_params['t_continuity_threshold'], 
#                     p_mode = 'quantile', p_amplitude_threshold = best_params['p_amplitude_threshold'], 
#                     p_quantile = 0.9,
#                     p_continuity_threshold = best_params['p_continuity_threshold'], 
#                     p_baseline_length = 120,
#                     ignore_hypoxics_duration = 15, return_seq = True, 
#                     return_concat = True, return_tuple = False, 
#                     phasic_start_time_only = False,
#                     return_multilabel_track = True,
#                     verbose = True,
#                     use_muliprocessors = True,
#                     num_processors = 4,
#                     score_all_files_in_dir = False,
#                     use_partition = False,
#                     ID_list = test)
#    predictions = scorer.score_all()
#    annotations = scorer.get_annotations()
#    evaluator = Evaluator(predictions = predictions, annotations = annotations, verbose = False)
#    bal_accuracy = evaluator.balanced_accuracy_signals()
#    cohen_kappa_epoch = evaluator.cohen_kappa_epoch()
#    accuracy_diagnosis = evaluator.accuracy_score_diagnosis()
#    print(f"Balanced accuracy: {bal_accuracy}")
#    print(f"Cohen_kappa_epoch: {cohen_kappa_epoch}")
#    print(f"Accuracy_diagnosis: {accuracy_diagnosis}")
    return best_params
                
            
        
        
        
        
