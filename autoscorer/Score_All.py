#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:34:54 2019

@author: danielyaeger
"""
import re
from autoscorer.autoscorer import Autoscorer
from pathlib import Path
#import pickle
import pandas as pd
import numpy as np

def collapse_confusion_matrix(conf_matrix: pd.DataFrame, separate_events: True):
    if not separate_events:
        c_mat = np.array([[sum(conf_matrix['TP'].values), sum(conf_matrix['FP'].values)],
                   [(conf_matrix['FN'].values), sum(conf_matrix['TN'].values)]])
        rows = pd.Index(['Predicted True', 'Predicted False'], name = 'rows')
        cols = pd.Index(['Actual True', 'Actual False'], name = 'columns')
        return pd.DataFrame(data = c_mat, index = rows, columns = cols)
    else:
        rows = pd.Index(['Predicted True', 'Predicted False'], name = 'rows')
        cols = pd.Index(['Actual True', 'Actual False'], name = 'columns')
        tonic_mat = np.array([[sum(conf_matrix['tonic_TP'].values), sum(conf_matrix['tonic_FP'].values)],
                   [sum(conf_matrix['tonic_FN'].values), sum(conf_matrix['tonic_TN'].values)]])
        phasic_mat = np.array([[sum(conf_matrix['phasic_TP'].values), sum(conf_matrix['phasic_FP'].values)],
                   [sum(conf_matrix['phasic_FN'].values), sum(conf_matrix['phasic_TN'].values)]])
        return pd.DataFrame(data = tonic_mat, index = rows, columns = cols), pd.DataFrame(data = phasic_mat, index = rows, columns = cols)


def confusionMatrix(predictions: dict, annotations: dict, combine_t_and_p: True):
    """ Input: predictions dictionary, organized by ID, and an annotations (ground
    truth) dictionary, organized by ID. It is assumed that the data in both
    the predictions and annotations are in the format of sequences.
    
    Parameters:
        combine_t_and_p: if selected, the T and P events are combined into a
        single sequence.
    
    Output: A dataframe with true positives, true negatives, false positives,
    and false negatives organized by ID"""
    
    IDs = list(predictions.keys())
    if combine_t_and_p:
        TPs, FPs, TNs, FNs = [], [], [], []
    else:
        tTPs, tFPs, tTNs, tFNs = [], [], [], []
        fTPs, fFPs, fTNs, fFNs = [], [], [], []
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
            TPs.append(sum((combined_predictions == 1) & (combined_actual == 1)))
            FPs.append(sum((combined_predictions == 1) & (combined_actual == 0)))
            TNs.append(sum((combined_predictions == 0) & (combined_actual == 0)))
            FNs.append(sum((combined_predictions == 0) & (combined_actual == 1)))
        else:
            # Top row of predicted and actual are tonic events
            tTPs.append(sum((predicted[0,:] == 1) & (actual[0,:] == 1)))
            tFPs.append(sum((predicted[0,:] == 1) & (actual[0,:] == 0)))
            tTNs.append(sum((predicted[0,:] == 0) & (actual[0,:] == 0)))
            tFNs.append(sum((predicted[0,:] == 0) & (actual[0,:] == 1)))
            # Bottom row of predicted and actual are phasic events
            fTPs.append(sum((predicted[1,:] == 1) & (actual[1,:] == 1)))
            fFPs.append(sum((predicted[1,:] == 1) & (actual[1,:] == 0)))
            fTNs.append(sum((predicted[1,:] == 0) & (actual[1,:] == 0)))
            fFNs.append(sum((predicted[1,:] == 0) & (actual[1,:] == 1)))
    if combine_t_and_p: 
        data_dict = {'ID': IDs, 'TP': TPs, 'FP': FPs, 'TN': TNs, 'FN': FNs}
    else:
        data_dict = {'ID': IDs, 'tonic_TP': tTPs, 'tonic_FP': tFPs, 
                     'tonic_TN': tTNs, 'tonic_FN': tFNs, 'phasic_TP': fTPs,
                     'phasic_FP': fFPs, 'phasic_TN': fTNs, 'phasic_FN': fFNs}
    return pd.DataFrame(data = data_dict)

def scoreAll(data_path = '/Users/danielyaeger/Documents/processed_data/Rectified_and_Resampled',
                 f_s = 10, t_amplitude_threshold = 1,
                 t_continuity_threshold = 10, p_mode = 'mean', 
                 p_amplitude_threshold = 4, p_quantile = 0.99,
                 p_continuity_threshold = 1, p_baseline_length = 120,
                 ignore_hypoxics_duration = 15, return_seq = True, 
                 return_concat = True, return_tuple = False, 
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
        #try:
        scorer = Autoscorer(ID = ID, data_path = data_path,
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
            
        #except AssertionError as e:
            #print(e)
    
    return results_dict, annotations_dict

if __name__ == "__main__":
     data_path = '/Users/danielyaeger/Documents/processed_data/processed'
     results_dict, annotations_dict = scoreAll(data_path = data_path)
     df = confusionMatrix(predictions = results_dict['results'], annotations = annotations_dict, combine_t_and_p = False)
     df2, df3 = collapse_confusion_matrix(conf_matrix = df, separate_events = True)
     df2.to_csv('tonic_conf_matrix.csv', index = None, header = True)
     df3.to_csv('phasic_conf_matrix.csv', index = None, header = True)
#    master_dict = scoreAll()    
#    out_path = Path('/Users/danielyaeger/Documents/My_sleep_research_ml')
#    save_dir_name = 'rule_based_scorer_results'
#    if not out_path.joinpath(save_dir_name).exists():
#        out_path.joinpath(save_dir_name).mkdir()
#    save_path = out_path.joinpath(save_dir_name)
#    with open(str(save_path.joinpath('Results_Dict')), 'wb') as f_out:
#        pickle.dump(master_dict,f_out)