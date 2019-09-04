#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:34:54 2019

@author: danielyaeger
"""
import re
import autoscorer.autoscorer
from pathlib import Path
import pickle
import pandas
import numpy as np


def confusionMatrix(predictions: dict, annotations: dict, combine_t_and_p: True):
    """ Input: predictions dictionary, organized by ID, and an annotations (ground
    truth) dictionary, organized by ID. It is assumed that the data in both
    the predictions and annotations are in the format of sequences.
    
    Parameters:
        combine_t_and_p: if selected, the T and P events are combined into a
        single sequence.
    
    Output: A dataframe with true positives, true negatives, false positives,
    and false negatives organized by ID"""
    
    def calc_conf_matrix(predictions: np.ndarray, actual: np.ndarray,
                         label_dict: dict = {'TP': 'TP', 'FP': 'FP', 
                                            'TN': 'TN', 'FN': 'FN'}) -> dict:
        TPs, FPs, TNs, FNs = [], [], [], []
        TPs.append(sum((predictions == 1) & (actual == 1)))
        FPs.append(sum((predictions == 1) & (actual == 0)))
        TNs.append(sum((predictions == 0) & (actual == 0)))
        FNs.append(sum((predictions == 0) & (actual == 1)))
        return {label_dict['TP']: TPs, 
                label_dict['FP']: FPs,
                label_dict['TN']: TNs,
                label_dict['FN']: FNs}
        
        
    IDs = list(predictions.keys())
    for ID in IDs:
        for i,subseq in enumerate(sorted(predictions[ID]['RSWA_T'].keys())):
            if i == 0:
                predicted = np.vstack((predictions[ID]['RSWA_T'][subseq], predictions[ID]['RSWA_P'][subseq]))
                actual = np.vstack((annotations[ID]['RSWA_T'][subseq], annotations[ID]['RSWA_P'][subseq]))
            else:
                predicted = np.hstack((predicted, 
                                       np.vstack((predictions[ID]['RSWA_T'][subseq], predictions[ID]['RSWA_P'][subseq]))))
                actual = np.hstack((actual,
                                    np.vstack((annotations[ID]['RSWA_T'][subseq], annotations[ID]['RSWA_P'][subseq]))))
    if combine_t_and_p:
        combined_predictions, combined_actual = np.zeros(predicted.shape[0]), np.zeros(actual.shape[0])
        combined_predictions[np.nonzero(predicted)[1]], combined_actual[np.nonzero(actual)[1]] = 1, 1
        data_dict = calc_conf_matrix(predictions = combined_predictions, actual = combined_actual,
                     label_dict = {'TP': 'TP', 'FP': 'FP', 'TN': 'TN', 'FN': 'FN'})
        data_dict['ID'] = IDs
    else:
        # Top row of predicted and actual are tonic events
        data_dict_tonic = calc_conf_matrix(predictions = predicted[0,:], actual = actual[0,:],
                     label_dict = {'TP': 'tonic_TP', 'FP': 'tonic_FP', 'TN': 'tonic_TN', 'FN': 'tonic_FN'})
        # Bottom row of predicted and actual are phasic events
        data_dict_phasic = calc_conf_matrix(predictions = predicted[1,:], actual = actual[1,:],
                     label_dict = {'TP': 'phasic_TP', 'FP': 'phasic_FP', 'TN': 'phasic_TN', 'FN': 'phasic_FN'})
        data_dict = {**data_dict_tonic, **data_dict_phasic}
        data_dict['ID'] = IDs
    return pandas.DataFrame(data = data_dict)

def scoreAll(data_path = '/Users/danielyaeger/Documents/processed_data/Rectified_and_Resampled',
                 f_s = 10, t_amplitude_threshold = 0.05,
                 t_continuity_threshold = 300, p_mode = 'mean', 
                 p_amplitude_threshold = 1, p_quantile = 0.99,
                 p_continuity_threshold = 1, p_baseline_length = 120,
                 ignore_hypoxics_duration = 0, return_seq = True, 
                 return_concat = True, return_tuple = True, 
                 phasic_start_time_only = True,
                 verbose = True):

    """ Wrapper method for rule_based_scorer's score() method that will iterate
    through all files in the data_path and call the score method on each file.
    See rule_based_scorer for information on parameters.
    
    INPUT: data_path, path to directory of sleep study .p files
    
    OUTPUT: dictionary of dictionaries with keys that are sleeper IDs, and
    values that are the output of rule_based_scorer's score() method (i.e. tonic
    and phasic signal-level event annotations)"""
    
    if type(data_path) == str:
            data_path = Path(data_path)
    
    results_dict = {'params': {'t_amplitude_threshold': t_amplitude_threshold,
                               't_continuity_threshold': t_continuity_threshold,
                               'p_mode': p_mode, 'p_amplitude_threshold': p_amplitude_threshold,
                               'p_quantile': p_quantile, 'p_continuity_threshold': p_continuity_threshold,
                               'p_baseline_length': p_baseline_length, 
                               'ignore_hypoxics_duration': ignore_hypoxics_duration}, 
                    'results': {}}
    annotations_dict = {}
    ID_set = set()
    
    # Generate list of unique IDs
    for file in list(data_path.glob('**/*.p')):
        if len(re.findall('[0-9A-Z]', file.stem)) > 0:
            ID_set.add(file.stem.split('_')[0])
    
    for ID in list(ID_set):
        try:
            scorer = autoscorer.Autoscorer(ID = ID, data_path = data_path,
                     f_s = f_s,
                     t_amplitude_threshold = t_amplitude_threshold,
                     t_continuity_threshold = t_continuity_threshold, 
                     p_mode = p_mode, 
                     p_amplitude_threshold = p_amplitude_threshold, 
                     p_quantile = p_quantile,
                     p_continuity_threshold = p_continuity_threshold, 
                     p_baseline_length = p_baseline_length, 
                     ignore_hypoxics_duration = ignore_hypoxics_duration,
                     return_seq = return_seq, return_concat = return_concat, 
                     return_tuple = return_tuple, 
                     phasic_start_time_only = phasic_start_time_only,
                     verbose = verbose)
            results_dict['results'][ID] = scorer.score_REM()
            annotations_dict[ID] = scorer.get_annotations()
            
        except AssertionError as e:
            print(e)
    
    return results_dict, annotations_dict

if __name__ == "__main__":
    master_dict = scoreAll()    
    out_path = Path('/Users/danielyaeger/Documents/My_sleep_research_ml')
    save_dir_name = 'rule_based_scorer_results'
    if not out_path.joinpath(save_dir_name).exists():
        out_path.joinpath(save_dir_name).mkdir()
    save_path = out_path.joinpath(save_dir_name)
    with open(str(save_path.joinpath('Results_Dict')), 'wb') as f_out:
        pickle.dump(master_dict,f_out)