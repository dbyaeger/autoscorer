#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for 'autoscorer' package
"""
import pytest
import numpy as np
#from autoscorer.autoscorer_helpers.autoscorer_helpers import get_data, make_event_idx, convert_to_rem_idx, tuple_builder, sequence_builder
from autoscorer import autoscorer


def set_scorer_baseline_dict(scorer: autoscorer, rem_subsequences = 2,
                             channels = ['Chin', 'L Leg', 'R Leg']) -> None:
    baseline_dict = {'RSWA_P': {}, 'RSWA_T': {}}
    for i in range(rem_subsequences):
        baseline_dict['RSWA_P'][f'REM_{i}'] = {}
        baseline_dict['RSWA_T'][f'REM_{i}'] = {}
        for channel in channels:
            baseline_dict['RSWA_T'][f'REM_{i}'][channel] = {}
    scorer.baseline_dict = baseline_dict
    

def set_scorer_times(scorer: autoscorer, nrem_start_time = 4940,
                          nrem_end_time = 5000, rem_start_time = 5000,
                          rem_end_time = 5060) -> None:
    scorer.nrem_start_time = nrem_start_time
    scorer.nrem_end_time = nrem_end_time
    scorer.rem_start_time = rem_start_time
    scorer.rem_end_time = rem_end_time
    scorer.baseline_dict = {'RSWA_P': {'REM_0': {}, 'REM_1':{}}, 'RSWA_T': {}}

def make_fake_data(mode = 'same', f_s = 100, baseline_length = 60, 
                   REM_length = 60, baseline_type = 'same', channels = ['Chin', 'L Leg', 'R Leg']) -> dict:
    assert mode in ['same', 'normal', 'peaky'], f"Mode must be same, normal, or peaky, not {mode}!"
    assert baseline_type in ['same', 'low', 'high'], f"Baseline mode must same, low, or high, not {baseline_type}!"
    
    data = {}
    
    if mode == 'same':
        for channel in channels:
            data[channel] = np.ones((REM_length,f_s))
    
    if baseline_type == 'same':
        for channel in channels:
            data[channel] = np.concatenate(data[channel][0:baseline_length],data[channel])
    
    return data


def all_same():
    scorer = autoscorer.autoscorer(t_amplitude_threshold = 3,
                 t_continuity_threshold = 100, p_mode = 'stdev',
                 p_amplitude_threshold = 5, p_quantile = 0.99,
                 p_continuity_threshold = 10, p_baseline_length = 120,
                 ignore_hypoxics_duration = 15, return_seq = True)
    set_scorer_times(scorer)
    set_scorer_baseline_dict(scorer)
    scorer.rem_subseq = 0
    data = make_fake_data(mode = 'same',  baseline_type = 'same')
    results = scorer.score_Tonics(data)
    return results

def test_all_same():
    assert sum(results) == 0
    
    
    