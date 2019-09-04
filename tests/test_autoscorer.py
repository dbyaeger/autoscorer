#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for 'autoscorer' package
"""
#import pytest
import numpy as np
from testing_helpers import set_scorer_attributes, make_fake_data
import autoscorer.autoscorer


def test_all_same():
    scorer = autoscorer.Autoscorer(t_amplitude_threshold = 3,
                 t_continuity_threshold = 100, p_mode = 'stdev',
                 p_amplitude_threshold = 5, p_quantile = 0.99,
                 p_continuity_threshold = 10, p_baseline_length = 120,
                 ignore_hypoxics_duration = 15, return_seq = True, 
                 return_tuple = False)
    set_scorer_attributes(scorer)
    scorer.rem_subseq = 0
    data = make_fake_data(mode = 'same',  baseline_type = 'same', REM_length = 60)
    results = scorer.score_Tonics(data)
    assert sum(results) == 0


test_all_same()

    