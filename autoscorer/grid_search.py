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
               'p_baseline_length','ignore_hypoxics_duration', 'balanced_accuracy',
               'Cohen_kappa_epoch', 'collisions']
    
    if path_to_csv.exists():
        results_df = pd.read_csv(path_to_csv)
    else:
        results_df = pd.DataFrame(columns = columns)
    
    
    for experiment in experiments:
        print(experiment)
        
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
                 num_processors = 4,
                 use_partition = True,
                 partition_file_name = 'data_partition.p',
                 partition_mode = "train")
        results = all_scorer.score_all()
        experiment['balanced_accuracy'] = all_scorer.balanced_accuracy()
        experiment['collisions'] = all_scorer.get_collisions()
        experiment['Cohen_kappa_epoch'] = all_scorer.cohen_kappa_epoch()
        experiment_df = pd.DataFrame.from_dict(experiment)
        results_df.append(experiment_df, ignore_index=True)
        results_df.to_csv(str(path_to_csv), index=False)

if __name__ == "__main__":
    
    params = {'t_amplitude_threshold': [1, 3, 10],
          'p_mode': ['quantile'], 'p_quantile': [0.50, 0.67, 0.85],
          't_continuity_threshold': [10], 'p_amplitude_threshold': [4],
          'p_continuity_threshold': [1], 'p_baseline_length': [120],
          'ignore_hypoxics_duration': [15]}
    
    grid_search(params_dict = params)
    
    

    
    
    


