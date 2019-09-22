#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:34:54 2019

@author: danielyaeger
"""
import re
from autoscorer.autoscorer import Autoscorer
from autoscorer.clinical_scorer import Clinical_Scorer
from pathlib import Path
import sklearn.metrics as sk
import numpy as np
import multiprocessing as mp
import pickle
import collections


class All_Scorer(object):
    """ Wrapper class for rule_based_scorer's score() method that will iterate
    through all files in the data_path and call the score method on each file.
    See rule_based_scorer for information on parameters.
    
    INPUT: data_path, path to directory of sleep study .p files"""
    
    def __init__(self, data_path = '/Users/danielyaeger/Documents/processed_data/processed',
                 f_s = 10, t_amplitude_threshold = 1,
                 t_continuity_threshold = 10, p_mode = 'quantile', 
                 p_amplitude_threshold = 4, p_quantile = 0.67,
                 p_continuity_threshold = 1, p_baseline_length = 120,
                 ignore_hypoxics_duration = 15, return_seq = True, 
                 return_concat = True, return_tuple = False, 
                 phasic_start_time_only = True,
                 return_multilabel_track = True,
                 verbose = True,
                 use_muliprocessors = False,
                 num_processors = 4,
                 use_partition = True,
                 partition_file_name = 'data_partition.p',
                 partition_mode = "train"):
        
        if type(data_path) == str:
            data_path = Path(data_path)
        
        self.data_path = data_path
        self.f_s = f_s
        self.t_amplitude_threshold = t_amplitude_threshold
        self.t_continuity_threshold = t_continuity_threshold    
        self.p_continuity_threshold = p_continuity_threshold
        self.p_amplitude_threshold = p_amplitude_threshold
        self.p_quantile = p_quantile
        self.p_mode = p_mode
        self.p_baseline_length = p_baseline_length
        self.ignore_hypoxics_duration = ignore_hypoxics_duration
        self.return_concat = return_concat
        self.return_seq = return_seq
        self.return_tuple = return_tuple
        self.verbose = verbose
        self.phasic_start_time_only = phasic_start_time_only
        self.return_multilabel_track = return_multilabel_track
        self.use_muliprocessors = use_muliprocessors
        self.num_processors = num_processors
        self.use_partition = True
        self.partition_file_name = partition_file_name
        assert partition_mode in ['train', 'test', 'cv'], "Mode must be either train, test, or cv"
        self.partition_mode = partition_mode
        self.EPOCH_LEN = 30
        self.make_dicts()
        self.get_unique_IDs()
        self.collisions = 0

    def make_dicts(self):
        """Makes results dictionary in the format:
            
        results_dict = {params: {'t_amplitude_threshold': ..., },
                            results: {'XVZ2FFAEC864IPK': {results_dictionary}, 
                            ...} }
        
        The results_dictionary for each sleeper ID is the output of the 
        autoscorer instances's score_REM method.
        
        Also makes annotation dictionary in the format:
        
        annotations_dict = {'XVZ2FFAEC864IPK': annotation_dict, ...}
        
        The annotation_dict is the output of the autoscorer instance's
        get_annotations method.
        """
    
        self.results_dict = {'params': {'t_amplitude_threshold': self.t_amplitude_threshold,
                                   't_continuity_threshold': self.t_continuity_threshold,
                                   'p_mode': self.p_mode, 'p_amplitude_threshold': self.p_amplitude_threshold,
                                   'p_quantile': self.p_quantile, 'p_continuity_threshold': self.p_continuity_threshold,
                                   'p_baseline_length': self.p_baseline_length, 
                                   'ignore_hypoxics_duration': self.ignore_hypoxics_duration}, 
                        'results': {}}
        self.annotations_dict = {}
    
    def get_unique_IDs(self):
        """ Finds all of the unique patient IDs in a directory. A patient ID
        is expected to contain only numbers and uppercase letters"""
        
        if not self.use_partition: 

            ID_set = set()
            
            # Generate list of unique IDs
            for file in list(self.data_path.glob('**/*.p')):
                if len(re.findall('[0-9A-Z]', file.stem)) > 0:
                    ID_set.add(file.stem.split('_')[0])
            self.ID_list = list(ID_set)        
        else:
            
            with self.data_path.joinpath(self.partition_file_name).open('rb') as fh:
                ID_list = list(pickle.load(fh)[self.partition_mode])
                ID_list = [x for x in ID_list if len(re.findall('[0-9A-Z]', x)) > 0]
            ID_list = [s.split('_')[0] for s in ID_list]
            self.ID_list = list(set(ID_list))
            
    def _score(self, ID) -> tuple:
        """ Calls the autoscorer score_REM method with the input ID. Returns
        a tuple containing:
            (results_dict, annotations_dict, collisions)"""
        scorer = Autoscorer(ID = ID, data_path = self.data_path,
                 f_s = self.f_s,
                 t_amplitude_threshold = self.t_amplitude_threshold,
                 t_continuity_threshold = self.t_continuity_threshold, 
                 p_mode = self.p_mode, 
                 p_amplitude_threshold = self.p_amplitude_threshold, 
                 p_quantile = self.p_quantile,
                 p_continuity_threshold = self.p_continuity_threshold, 
                 p_baseline_length = self.p_baseline_length, 
                 ignore_hypoxics_duration = self.ignore_hypoxics_duration,
                 return_seq = self.return_seq, return_concat = self.return_concat, 
                 return_tuple = self.return_tuple, 
                 phasic_start_time_only = self.phasic_start_time_only,
                 return_multilabel_track = self.return_multilabel_track,
                 verbose = self.verbose)
        if self.return_multilabel_track:
            return (scorer.score_REM(), scorer.get_annotations(), scorer.get_collisions())
        else:
            return (scorer.score_REM(), scorer.get_annotations())
    
    def score_all(self) -> dict:
        """Scores all patient studies in a directory. Returns results_dict"""
        if self.use_muliprocessors:
            pool = mp.Pool(processes=self.num_processors)
            results = list(pool.map(self._score, self.ID_list))
        else:
            results = list(map(self._score, self.ID_list))
        for i, result in enumerate(results):
            self.results_dict['results'][self.ID_list[i]] = result[0]
            self.annotations_dict[self.ID_list[i]] = result[1]
            if self.return_multilabel_track:
                self.collisions += result[2]
        return self.get_scores()
    
    def get_scores(self) -> dict:
        """ Returns results_dict"""
        return self.results_dict
    
    def get_annotations(self) -> dict:
        """ Returns annotations dict"""
        return self.annotations_dict
    
    def get_collisions(self) -> int:
        """ Returns number of collisions"""
        assert self.return_multilabel_track, "Collisions only defined when return_multilabel_track set to True!"
        return self.collisions
        

if __name__ == "__main__":
    import pickle
    with open('/Users/danielyaeger/Documents/processed_data/processed/data_partition.p', 'rb') as fh:
        ID_list = list(pickle.load(fh)["train"])
    ID_list = [s.split('_')[0] for s in ID_list]
    ID_list = list(set(ID_list))
    data_path = '/Users/danielyaeger/Documents/processed_data/processed'
    all_scorer = All_Scorer(data_path = data_path, ID_list = ID_list)
    results_dict = all_scorer.score_all()
