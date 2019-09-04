#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:05:48 2019

@author: danielyaeger
"""

from autoscorer import autoscorer

def set_scorer_attributes(scorer: autoscorer.Autoscorer, rem_subsequences: int = 2, 
    channels: list = ['Chin', 'L Leg', 'R Leg'], nrem_start_time: int = 4940, 
    nrem_end_time: int = 5000, rem_start_time: int = 5000, rem_end_time: int = 5060,
    a_and_h_overlap: float = 0, f_s: int = 100) -> None:
    
    set_scorer_baseline_dict(scorer = scorer, rem_subsequences = rem_subsequences,
                             channels = channels)
    set_scorer_times(scorer = scorer, nrem_start_time = nrem_start_time,
                          nrem_end_time = nrem_end_time, rem_start_time = rem_start_time,
                          rem_end_time = rem_end_time)
    set_scorer_a_and_h(scorer = scorer, a_and_h_overlap = a_and_h_overlap, 
                REM_length = rem_start_time - rem_end_time, f_s = f_s)
    
        
                          
                          
                          

def set_scorer_baseline_dict(scorer: autoscorer.Autoscorer, rem_subsequences: int = 2,
                             channels: list = ['Chin', 'L Leg', 'R Leg']) -> None:
    baseline_dict = {'RSWA_P': {}, 'RSWA_T': {}}
    for i in range(rem_subsequences):
        baseline_dict['RSWA_P'][f'REM_{i}'] = {}
        baseline_dict['RSWA_T'][f'REM_{i}'] = {}
        for channel in channels:
            baseline_dict['RSWA_T'][f'REM_{i}'][channel] = {}
    scorer.baseline_dict = baseline_dict
    

def set_scorer_times(scorer: autoscorer.Autoscorer, nrem_start_time: int = 4940,
                          nrem_end_time: int = 5000, rem_start_time: int = 5000,
                          rem_end_time: int = 5060) -> None:
    scorer.nrem_start_time = nrem_start_time
    scorer.nrem_end_time = nrem_end_time
    scorer.rem_start_time = rem_start_time
    scorer.rem_end_time = rem_end_time
    scorer.baseline_dict = {'RSWA_P': {'REM_0': {}, 'REM_1':{}}, 'RSWA_T': {}}

def set_scorer_a_and_h(scorer: autoscorer.Autoscorer, a_and_h_overlap: float = 0, 
                REM_length: int = 60, f_s: int = 100) -> None:
    assert 0 <= a_and_h_overlap <= 1, f"a_and_h_overlap must be between 0 and 1, not {a_and_h_overlap}!"
    scorer.a_and_h_idx = [(0, f_s*REM_length*a_and_h_overlap)]

def make_fake_data(mode = 'same', f_s = 100, baseline_length = 60, 
                   REM_length = 60, baseline_type = 'same', 
                   channels: list = ['Chin', 'L Leg', 'R Leg']) -> dict:
    
    assert mode in ['same', 'normal', 'peaky'], f"Mode must be same, normal, or peaky, not {mode}!"
    assert baseline_type in ['same', 'low', 'high'], f"Baseline mode must same, low, or high, not {baseline_type}!"
    
    data = {'signals': {}}
    
    if mode == 'same':
        for channel in channels:
            data['signals'][channel] = np.ones((REM_length,f_s))
    
    if baseline_type == 'same':
        for channel in channels:
            data['signals'][channel] = np.concatenate((data['signals'][channel][0:baseline_length,:],
                data['signals'][channel]))
    
    return data