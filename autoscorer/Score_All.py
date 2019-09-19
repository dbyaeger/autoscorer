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
        self.combine_IDs()
        self.create_clinical_scorer()
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
    
    def get_human_diagnoses(self) -> dict:
        """Returns human scorer-generated diagnoses based on AASM criteria"""
        return self.clinical_scorer.get_human_diagnoses()
    
    def get_autoscorer_diagnoses(self) -> dict:
        """Returns autoscorer-generated diagnoses based on AASM critera"""
        return self.clinical_scorer.pred_diagnosis
    
    def create_clinical_scorer(self) -> None:
        """Initializes a Clinical_Scorer to provide diagnoses"""
        self.clinical_scorer = Clinical_Scorer(predictions = self.results_dict['results'],
                                               annotations = self.annotations_dict,
                                               offset = 0,
                                               return_multilabel_track = self.return_multilabel_track,
                                               EPOCH_LEN = self.EPOCH_LEN,
                                               f_s = self.f_s,
                                               verbose = self.verbose)
        
    def combine_IDs(self) -> None:
        """ Combines result_dict and annotation_dict across IDs and REM subsequences.
        y_pred and y_true are either 1-D if return_multilabel_track
        is set to True, or 2-D numpy ndarrays if not. The arrays are stored
        as instance attributes"""
        
        if self.return_tuple:
            return
        
        for ID in self.ID_list:
            if self.return_multilabel_track:
                rem_subseqs = self.results_dict['results'][ID].keys()
            else:
                rem_subseqs = self.results_dict['results'][ID]['RSWA_T'].keys()
            for i,subseq in enumerate(rem_subseqs):
                if i == 0:
                    if not self.return_multilabel_track:
                        y_pred = np.vstack((self.results_dict['results'][ID]['RSWA_T'][subseq], 
                                        self.results_dict['results'][ID]['RSWA_P'][subseq]))
                        y_true = np.vstack((self.annotations_dict[ID]['RSWA_T'][subseq], 
                                        self.annotations_dict[ID]['RSWA_P'][subseq]))
                    else:
                        y_pred = self.results_dict['results'][ID][subseq]
                        y_true = self.annotations_dict[ID][subseq]
                else:
                    if not self.return_multilabel_track:
                        y_pred = np.hstack((y_pred,
                                            np.vstack((self.results_dict['results'][ID]['RSWA_T'][subseq], 
                                        self.results_dict['results'][ID]['RSWA_P'][subseq]))))
                        y_true = np.hstack((y_true,
                                            np.vstack((self.annotations_dict[ID]['RSWA_T'][subseq], 
                                        self.annotations_dict[ID]['RSWA_P'][subseq]))))
                    else:
                        y_pred = np.concatenate((y_pred, self.results_dict['results'][ID][subseq]))
                        y_true = np.concatenate((y_true, self.annotations_dict[ID][subseq]))
        
        self.y_pred = y_pred
        self.y_true = y_true
                    
    def label_mode(self, epoch: np.ndarray) -> int:
        """Returns zero if an array contains only zeros. Otherwise returns the
        most common greater than zero element in the array. If there is a tie
        between 1 and 2 (phasic and tonic), returns phasic."""
        if len(epoch[epoch > 0]) == 0:
            return 0
        else:
            counts = collections.Counter(epoch[epoch > 0])
            if counts[2] > counts[1]: 
                return 2
            return 1
            
    def confusion_matrix_signals(self) -> np.ndarray or tuple:
        """Returns signal-level confusion matrix in the form 
        (if return_multilabel_track set to False):
            
            array([[TN, FP],
                   [FN, TP]])
    
        Two confusion matrices will be returned, one for tonic and one for phasic
        in the format:
            
            (tonic confusion matrix, phasic confusion matrix)
            
        if return_multilabel_track set to False
        
        if return_multilabel_track set to True, confusion matrix is in the format:
            
            array([[True None, Predicted Phasic/Actually None, Predicted Tonic/Actually None],
                   [Predicted None/Actually Phasic, True Phasic, Predicted Tonic/Actually Phasic],
                   [Predicted None/Actually Tonic, Predicted Phasic/Actually Tonic, True Tonic]])
        """
        assert not self.return_tuple, "Confusion matrix not yet implemented for return_tuple option"
        
        if not self.return_multilabel_track:
            return (sk.confusion_matrix(y_true = self.y_true[0,:], y_pred = self.y_pred[0,:]),
                    sk.confusion_matrix(y_true = self.y_true[1,:], y_pred = self.y_pred[1,:]))
        else:
            return sk.confusion_matrix(y_true = self.y_true, y_pred = self.y_pred)
    
    def confusion_matrix_diagnoses(self) -> np.ndarray:
        """Returns a confusion matrix for diagnoses of RSWA in the form:
            
             array([[TN, FP],
                   [FN, TP]])
        """
        return self.clinical_scorer.confusion_matrix()
    
    def balanced_accuracy_signals(self) -> float or tuple:
        """Returns the balanced accuracy score, either as a single value if
        return_multilabel_track is set to True, or as separate scores for
        tonic and phasic events in the format:
            
            (tonic balanced accuracy score, phasic balanced accuracy score)
        
        if return_multilabel_track is set to False"""
        
        assert not self.return_tuple, "Balanced accuracy not yet implemented for return_tuple option"
        
        if not self.return_multilabel_track:
            return (sk.balanced_accuracy_score(y_true = self.y_true[0,:], y_pred = self.y_pred[0,:]),
                    sk.balanced_accuracy_score(y_true = self.y_true[1,:], y_pred = self.y_pred[1,:]))
        else:
            return sk.balanced_accuracy_score(y_true = self.y_true, y_pred = self.y_pred)
    
    def cohen_kappa_epoch(self) -> float:
        """Calculates inter-rater agreement on an epoch level between human
        annotations and autoscorer predictions using Cohen's kappa"""
        
        assert not self.return_tuple, "Cohen kappa method not yet implemented for return_tuple option"
        y_pred_label = np.zeros(int(len(self.y_pred)/(self.f_s*self.EPOCH_LEN)))
        y_true_label = np.zeros(int(len(self.y_pred)/(self.f_s*self.EPOCH_LEN)))
        for i in range(0,len(self.y_pred),self.f_s*self.EPOCH_LEN):
            y_pred_label[i//(self.f_s*self.EPOCH_LEN)] = self.label_mode(epoch = self.y_pred[i:i + (self.f_s*self.EPOCH_LEN)])
            y_true_label[i//(self.f_s*self.EPOCH_LEN)] = self.label_mode(epoch = self.y_true[i:i + (self.f_s*self.EPOCH_LEN)])
        return sk.cohen_kappa_score(y_pred_label,y_true_label)
    
    def balanced_accuracy_diagnosis(self) -> float:
        """Returns balanced accuracy for RSWA diagnoses"""
        return self.clinical_scorer.balanced_accuracy()
    
    def cohen_kappa_diagnosis(self) -> float:
        """Returns cohen's kappa for RSWA diagnoses"""
        return self.clinical_scorer.cohen_kappa_diagnosis()
    
    

if __name__ == "__main__":
    import pickle
    with open('/Users/danielyaeger/Documents/processed_data/processed/data_partition.p', 'rb') as fh:
        ID_list = list(pickle.load(fh)["train"])
    ID_list = [s.split('_')[0] for s in ID_list]
    ID_list = list(set(ID_list))
    data_path = '/Users/danielyaeger/Documents/processed_data/processed'
    all_scorer = All_Scorer(data_path = data_path, ID_list = ID_list)
    results_dict = all_scorer.score_all()
    conf_mat = all_scorer.confusion_matrix()
    balanced_accuracy = all_scorer.balanced_accuracy()