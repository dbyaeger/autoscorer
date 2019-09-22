#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 11:12:43 2019

@author: danielyaeger
"""

from autoscorer.clinical_scorer import Clinical_Scorer
import sklearn.metrics as sk
import numpy as np
import collections

class Evaluator(object):
    """Takes as input dictionaries of predicted and human-annotated RSWA events,
    and calculates several error and inter-rater agreement metrics.
    
    INPUT:  predictions: dictionary of predicted P and T events, 
            annotations: human-annotated P and T events
            
        if multilabel_track set to True:
                    
            predictions and annotations should be in the format:
                    
                predictions/annotations = {ID_0: {REM_0: array, REM_1: array, .. REM_n: array},
                ID_1: {REM_0: array, REM_1: array, .. REM_n: array}, ...}}

                where each array is an array of sample-by-sample RSWA signal-level
                event scores (i.e 000022222000011100000...)
                    
        Otherwise,
            
            predictions and annotations should be in the format:
                
                predictions/annotations = {ID_0: {'RSWA_T': 
                {REM_0: array, REM_1: array, .. REM_n: array}, 
                'RSWA_P': {REM_0: array, REM_1: array, .. REM_n: array}},
                ...}

                where each array is an array of sample-by-sample RSWA signal-level
                event scores (i.e 000022222000011100000...)
                
    PARAMETERS:
            
        multilabel_track: if set to True, Evaluator expects
        single scoring track for P and T events. T events are assumed
        to be encoded as 2's and P events as 1's.
        
        EPOCH_LEN: Length of epoch in seconds. Set to 30 by default.
        
        f_s: Sampling rate of EMG signals. Set to 10 by default.
        
        offset: offset in seconds to use when making diagnoses. For example,
            if offset set to 3s, first epoch will be considered as 3 -> 33s
            
    OUTPUTS:
        
        balanced_accuracy_signals method -> balanced accuracy for signal-level 
                                            (Phasic and Tonic) events
        
        cohen_kappa_epoch -> inter-labeler agreement (based on predictions and
                              and annotations) on an epoch level
        
        accuracy_score_diagnosis -> accuracy of diagnoses made by predictions
                                    using clinical rules and treating annotations
                                    as the ground truth.
        
    """
    
    def __init__(self, predictions: dict, annotations: dict, 
                 multilabel_track: bool = True, EPOCH_LEN: int = 30,
                 f_s: int = 10, offset: int = 0):
        
        assert type(predictions) == type(annotations) == dict, "predictions and annotations must be dictionaries!"
        self.predictions = predictions
        self.annotations = annotations
        assert type(multilabel_track) == bool, "multilabel_track must be a boolean!"
        self.multilabel_track = multilabel_track
        self.EPOCH_LEN = EPOCH_LEN
        self.f_s = f_s 
        self.ID_list = set(self.predictions.keys())
        self.offset = offset
        self.clinical_scorer = Clinical_Scorer(predictions = self.predictions,
                                               annotations = self.annotations,
                                               offset = self.offset,
                                               multilabel_track = self.multilabel_track,
                                               EPOCH_LEN = self.EPOCH_LEN,
                                               f_s = self.f_s,
                                               verbose = self.verbose)
        self.combine_IDs()
        
        
    def combine_IDs(self) -> None:
        """ Combines result_dict and annotation_dict across IDs and REM subsequences.
        y_pred and y_true are either 1-D if multilabel_track
        is set to True, or 2-D numpy ndarrays if not. The arrays are stored
        as instance attributes"""
        
        for ID in self.ID_list:
            if self.multilabel_track:
                rem_subseqs = self.predictions[ID].keys()
            else:
                rem_subseqs = self.predictions[ID]['RSWA_T'].keys()
            for i,subseq in enumerate(rem_subseqs):
                if i == 0:
                    if not self.multilabel_track:
                        y_pred = np.vstack((self.predictions[ID]['RSWA_T'][subseq], 
                                        self.predictions[ID]['RSWA_P'][subseq]))
                        y_true = np.vstack((self.annotations[ID]['RSWA_T'][subseq], 
                                        self.annotations[ID]['RSWA_P'][subseq]))
                    else:
                        y_pred = self.predictions[ID][subseq]
                        y_true = self.annotations[ID][subseq]
                else:
                    if not self.multilabel_track:
                        y_pred = np.hstack((y_pred,
                                            np.vstack((self.predictions[ID]['RSWA_T'][subseq], 
                                        self.predictions[ID]['RSWA_P'][subseq]))))
                        y_true = np.hstack((y_true,
                                            np.vstack((self.annotations[ID]['RSWA_T'][subseq], 
                                        self.annotations[ID]['RSWA_P'][subseq]))))
                    else:
                        y_pred = np.concatenate((y_pred, self.predictions[ID][subseq]))
                        y_true = np.concatenate((y_true, self.annotations[ID][subseq]))
        
        self.y_pred = y_pred
        self.y_true = y_true
                               
    def confusion_matrix_signals(self) -> np.ndarray or tuple:
        """Returns signal-level confusion matrix in the form 
        (if multilabel_track set to False):
            
            array([[TN, FP],
                   [FN, TP]])
    
        Two confusion matrices will be returned, one for tonic and one for phasic
        in the format:
            
            (tonic confusion matrix, phasic confusion matrix)
            
        if multilabel_track set to False
        
        if multilabel_track set to True, confusion matrix is in the format:
            
            array([[True None, Predicted Phasic/Actually None, Predicted Tonic/Actually None],
                   [Predicted None/Actually Phasic, True Phasic, Predicted Tonic/Actually Phasic],
                   [Predicted None/Actually Tonic, Predicted Phasic/Actually Tonic, True Tonic]])
        """
        if not self.multilabel_track:
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
        multilabel_track is set to True, or as separate scores for
        tonic and phasic events in the format:
            
            (tonic balanced accuracy score, phasic balanced accuracy score)
        
        if multilabel_track is set to False"""
                
        if not self.multilabel_track:
            return (sk.balanced_accuracy_score(y_true = self.y_true[0,:], y_pred = self.y_pred[0,:]),
                    sk.balanced_accuracy_score(y_true = self.y_true[1,:], y_pred = self.y_pred[1,:]))
        else:
            return sk.balanced_accuracy_score(y_true = self.y_true, y_pred = self.y_pred)
    
    def cohen_kappa_epoch(self) -> float:
        """Calculates inter-rater agreement on an epoch level between human
        annotations and autoscorer predictions using Cohen's kappa"""
        
        y_pred_label = np.zeros(int(len(self.y_pred)/(self.f_s*self.EPOCH_LEN)))
        y_true_label = np.zeros(int(len(self.y_pred)/(self.f_s*self.EPOCH_LEN)))
        for i in range(0,len(self.y_pred),self.f_s*self.EPOCH_LEN):
            y_pred_label[i//(self.f_s*self.EPOCH_LEN)] = self.label_mode(epoch = self.y_pred[i:i + (self.f_s*self.EPOCH_LEN)])
            y_true_label[i//(self.f_s*self.EPOCH_LEN)] = self.label_mode(epoch = self.y_true[i:i + (self.f_s*self.EPOCH_LEN)])
        return sk.cohen_kappa_score(y_pred_label,y_true_label)
    
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
    
    def accuracy_score_diagnosis(self) -> float:
        """Returns balanced accuracy for RSWA diagnoses"""
        return self.clinical_scorer.accuracy_score()
    
    def cohen_kappa_diagnosis(self) -> float:
        """Returns cohen's kappa for RSWA diagnoses"""
        return self.clinical_scorer.cohen_kappa_diagnosis()
        
    def get_human_diagnoses(self) -> dict:
        """Returns human scorer-generated diagnoses based on AASM criteria"""
        return self.clinical_scorer.get_human_diagnoses()
    
    def get_predicted_diagnoses(self) -> dict:
        """Returns predicted diagnoses based on AASM critera"""
        return self.clinical_scorer.get_predicted_diagnoses()