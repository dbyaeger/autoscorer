#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 11:56:17 2019

@author: danielyaeger
"""

from autoscorer.autoscorer import Autoscorer
from autoscorer.evaluator import Evaluator
import numpy as np
import pandas as pd

def test_autoscorer(p_mode: str = "quantile", parameter: str = 'p_quantile',
                    start: float = 0.5, stop: float = 0.99, num: int= 5) -> tuple:
    values = np.linspace(start = start, stop = stop, num = num)
    results_dict = {f"{parameter}_value": values, 'balanced_accuracy_event': [],
                    'interlabeler_agreement_epoch': []}
    predict_dict = {}
    baseline_dict = {}
    for i in range(num):
        scorer = Autoscorer(p_mode = p_mode, p_quantile = values[i], return_multilabel_track = False,
                            return_seq = True, return_concat = True, return_tuple = False, verbose = False,
                            t_amplitude_threshold = 10)
        predictions = {scorer.ID: scorer.score_REM()}
        annotations = {scorer.ID: scorer.get_annotations()}
        predict_dict[values[i]] = predictions
#        evaluator = Evaluator(predictions = predictions, annotations = annotations)
#        results_dict['balanced_accuracy_event'].append(evaluator.balanced_accuracy_signals())
#        results_dict['interlabeler_agreement_epoch'].append(evaluator.cohen_kappa_epoch())
        baseline_dict[values[i]] = scorer.baseline_dict
#    print(pd.DataFrame.from_dict(results_dict))
    return (results_dict, predict_dict, baseline_dict)


