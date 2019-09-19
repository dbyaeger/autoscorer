#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 11:48:39 2019

@author: danielyaeger
"""
from itertools import product
from autoscorer.Score_All import All_Scorer
from pathlib import Path
import pandas as pd


def grid_search(params_dict: dict, 
                path_to_csv = '/Users/danielyaeger/Documents/autoscorer/results/results.csv'):
    """Performs a grid search using parameters. Writes to a csv file and reuses
    csv file if the file exists"""
        
    if type(path_to_csv) == str: path_to_csv = Path(path_to_csv)
        
    # Create experiment dictionary of combinations of parameters
    keys, values = zip(*params_dict.items())
    experiments = [dict(zip(keys, v)) for v in product(*values)]
    
    columns = ['t_amplitude_threshold', 't_continuity_threshold', 'p_mode',
               'p_amplitude_threshold','p_quantile', 'p_continuity_threshold',
               'p_baseline_length','ignore_hypoxics_duration', 'balanced_accuracy_event',
               'Cohen_kappa_epoch', 'Cohen_kappa_diagnosis', 'balanced_accuracy_diagnosis',
               'collisions']
    
    if path_to_csv.exists():
        results_df = pd.read_csv(path_to_csv)
        print("Existing results file found!")
    else:
        results_df = pd.DataFrame(columns = columns)
    
    
    for experiment in experiments:
        all_scorer = All_Scorer(data_path = '/Users/danielyaeger/Documents/processed_data/processed',
                 f_s = 10, t_amplitude_threshold = experiment['t_amplitude_threshold'],
                 t_continuity_threshold = experiment['t_continuity_threshold'], 
                 p_mode = experiment['p_mode'], 
                 p_amplitude_threshold = experiment['p_amplitude_threshold'], 
                 p_quantile = experiment['p_quantile'],
                 p_continuity_threshold = experiment['p_continuity_threshold'], 
                 p_baseline_length = experiment['p_baseline_length'],
                 ignore_hypoxics_duration = experiment['ignore_hypoxics_duration'], 
                 return_seq = True, 
                 return_concat = True, return_tuple = False, 
                 phasic_start_time_only = True,
                 return_multilabel_track = True,
                 verbose = True,
                 use_muliprocessors = False,
                 num_processors = 2,
                 use_partition = True,
                 partition_file_name = 'data_partition.p',
                 partition_mode = "cv")
        results_dict = all_scorer.score_all()
        experiment['balanced_accuracy_event'] = all_scorer.balanced_accuracy_signals()
        experiment['collisions'] = all_scorer.get_collisions()
        experiment['Cohen_kappa_epoch'] = all_scorer.cohen_kappa_epoch()
        experiment['Cohen_kappa_diagnosis'] = all_scorer.cohen_kappa_diagnosis()
        experiment['balanced_accuracy_diagnosis'] = all_scorer.cohen_kappa_diagnosis()
        experiment_df = pd.DataFrame.from_dict(listify_dict(experiment))
        results_df = results_df.append(experiment_df)
        results_df.to_csv(str(path_to_csv), index=False)

def listify_dict(d: dict) -> dict:
    """Pandas pd.DataFrame.from_dict throws error if a dictionary of scalar values
    is passed (see https://eulertech.wordpress.com/2017/11/28/pandas-valueerror-if-using-all-scalar-values-you-must-pass-an-index/).
    This function takes a dictionary with scalar values and returns a dictionary
    with lists"""
    list_dict = {}
    for key in d:
        if type(d[key]) != list: list_dict[key] = [d[key]]
    return list_dict

if __name__ == "__main__":
    
    params = {'t_amplitude_threshold': [15, 20, 30],
          'p_mode': ['quantile'], 'p_quantile': [0.50, 0.67, 0.85],
          't_continuity_threshold': [10], 'p_amplitude_threshold': [4],
          'p_continuity_threshold': [1], 'p_baseline_length': [120],
          'ignore_hypoxics_duration': [15]}
    
    grid_search(params_dict = params)



