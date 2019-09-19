#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 09:31:17 2019

@author: danielyaeger
"""

import numpy as np
import sklearn.metrics as sk

class Clinical_Scorer(object):
    """Takes dictionary of predicted labels and actual labels as input and produces
    a list of patients predicted (diagnosed) to have REM sleep behavior disorder.
    
    INPUT:  predictions: dictionary of predicted P and T events, 
            annotations: human-annotated P and T events
            
        if return_multilabel_track set to True:
                    
            predictions and annotations should be in the format:
                    
                predictions/annotations = {ID_0: {REM_0: array, REM_1: array, .. REM_n: array},
                ID_1: {REM_0: array, REM_1: array, .. REM_n: array}, ...}}

                where each array is an array of sample-by-sample RSWA signal-level
                event scores
                    
        Otherwise,
            
            predictions and annotations should be in the format:
                
                predictions/annotations = {ID_0: {'RSWA_T': 
                {REM_0: array, REM_1: array, .. REM_n: array}, 
                'RSWA_P': {REM_0: array, REM_1: array, .. REM_n: array}},
                ...}

                where each array is an array of sample-by-sample RSWA signal-level
                event scores
                
    PARAMETERS:
            
            offset: offset in seconds to use when scoring events. For example,
            if offset set to 3s, first epoch will be considered as 3 -> 33s
            
            return_multilabel_track: if set to True, Clinical_Scorer expects
            single scoring track for P and T events. T events are assumed
            to be encoded as 2's and P events as 1's.
    
    Output:
        
        get_human_diagnoses method -> dictionary of human scorer-based diagnoses 
        by patient ID
        
        get_autoscorer_diagnoses -> dictionary of automated scorer-based diagnoses 
        by patient ID
        
        confusion_matrix method -> confusion matrix for diagnoses
        
        balanced_accuracy -> balanced accuracy for diagnoses
        
        cohen_kappa_diagnosis -> cohen's kappa for diagnoses
        
    """
            
    def __init__(self, predictions: dict, annotations: dict = None, offset: int = 0, 
                 return_multilabel_track: bool = True, EPOCH_LEN: int = 30,
                 f_s: int = 10, predict_only: bool = False, verbose: bool = True):
        
        if not predict_only:
            assert type(predictions) == type(annotations) == dict, "predictions and annotations must be dictionaries!"
        else:
            assert type(predictions) == dict and annotations is None, "predictions must be a dictionary and annotations must be None if predict_only options is used!"
        self.predictions = predictions
        self.annotations = annotations
        assert offset >= 0, "Offset must be greater than or equal to zero!"
        assert type(offset) == int, "Offset must be an integer!"
        offset = offset*f_s
        self.offset = offset
        assert type(return_multilabel_track) == bool, "Return_multilabel_track must be an int!"
        self.return_multilabel_track = return_multilabel_track
        self.EPOCH_LEN = EPOCH_LEN
        self.f_s = f_s
        self.verbose = verbose
        self.predict_only = predict_only
        if self.return_multilabel_track:
            if not self.predict_only:
                self.multilabel_clinical_score()
                self.convert_to_vectors()
            else:
                self.predict_only()
        else:
            print("Warning! Methods for case that return_multilabel_track set to False not yet implemented!")
        
    def predict_only(self):
        """Generates diagnoses for each patient from the input dictionary of
        signal-level annotations"""
        
        assert self.return_multilabel_track, "Return_multilabel_track must be true to use this method!"
        
        self.pred_diagnosis = {}
        for ID in self.predictions.keys():
            pred_rswa = False
            for subseq in self.predictions[ID].keys():
                for i in range(self.offset,len(self.predictions[ID][subseq]), self.EPOCH_LEN*self.f_s):
                    if i+self.EPOCH_LEN*self.f_s > len(self.predictions[ID][subseq]):
                        if self.verbose:
                            print(f"With self.offset of {self.offset}, last {(len(self.predictions[ID][subseq]) - i)/self.f_s} seconds unscorable")
                        break
                    pred = self.predictions[ID][subseq][i:i+self.EPOCH_LEN*self.f_s]
                    if self.check_for_tonic(seq = pred, tonic_symbol = 2) or self.check_for_phasic(seq = pred, phasic_symbol = 1):
                        pred_rswa = True
            self.pred_diagnosis[ID] = pred_rswa
            if self.verbose:
                        print(f"For {ID}:\t predicted RSWA: {pred_rswa}")
        
    def multilabel_clinical_score(self):
        """Generates diagnoses for each patient from the predicted and human-generated
        event annotations"""
        
        assert self.return_multilabel_track, "Return_multilabel_track must be true to use this method!"
        
        self.pred_diagnosis, self.human_diagnosis = {}, {}
        for ID in self.annotations.keys():
            pred_rswa = False
            actual_rswa = False
            for subseq in self.annotations[ID].keys():
                for i in range(self.offset,len(self.annotations[ID][subseq]), self.EPOCH_LEN*self.f_s):
                    if i+self.EPOCH_LEN*self.f_s > len(self.annotations[ID][subseq]):
                        if self.verbose:
                            print(f"With self.offset of {self.offset}, last {(len(self.annotations[ID][subseq]) - i)/self.f_s} seconds unscorable")
                        break
                    actual = self.annotations[ID][subseq][i:i+self.EPOCH_LEN*self.f_s]
                    pred = self.predictions[ID][subseq][i:i+self.EPOCH_LEN*self.f_s]
                    if self.check_for_tonic(seq = actual, tonic_symbol = 2) or self.check_for_phasic(seq = actual, phasic_symbol = 1):
                        actual_rswa = True
                    if self.check_for_tonic(seq = pred, tonic_symbol = 2) or self.check_for_phasic(seq = pred, phasic_symbol = 1):
                        pred_rswa = True
            self.human_diagnosis[ID] = actual_rswa
            self.pred_diagnosis[ID] = pred_rswa
            if self.verbose:
                        print(f"For {ID}:\t predicted RSWA: {pred_rswa},\t human-scored RSWA diagnosis: {actual_rswa}")
                
    def check_for_tonic(self, seq: np.ndarray, tonic_symbol: int = 2) -> bool:
        """ Checks if tonic signal-level events meet AASM criteria. Takes sequence
        and tonic_symbol as inputs and returns True if at least half of the elements
        of the array are equal to tonic_symbol. Otherwise returns False.
        Generally, an epoch of signal should be passed to this function.
        """

        if sum(seq == tonic_symbol) >= len(seq)/2: 
            return True
        return False
        
    def check_for_phasic(self, seq: np.ndarray, phasic_symbol: int = 1) -> bool:
        """ Checks if phasic signal-level events meet AASM criteria. Takes sequence
        and phasic_symbol as inputs and returns True if, after dividing the signal
        into 3s mini-epochs, at least 5 of the mini-epochs contain a phasic event.
        Otherwise, returns false. Generally, an epoch of signal should be passed 
        to this function."""
        
        assert len(seq) % 2 == 0, f"Input array should be of even length, but length is {len(seq)}!"
        
        splits = np.split(ary=seq, indices_or_sections = 10)
        
        if sum([(l == phasic_symbol).any() for l in splits]) >= 5:
            return True
        return False
    
    def convert_to_vectors(self):
        """Creates vectors from instance attributes human_diagnosis and pred_diagnosis
        that are used to generate error metrics"""
        
        assert not self.predict_only, "Method cannot be used if predict_only option set to True!"
        
        self.y_pred = np.zeros(len(self.human_diagnosis.keys()))
        self.y_true = np.zeros(len(self.human_diagnosis.keys()))
        for i, ID in enumerate(self.human_diagnosis.keys()):
            self.y_pred[i] = int(self.pred_diagnosis[ID])
            self.y_true[i] = int(self.human_diagnosis[ID])
    
    def get_human_diagnoses(self) -> dict:
        """Returns human scorer-generated diagnoses based on AASM criteria"""
        
        assert not self.predict_only, "Method cannot be used if predict_only option set to True!"
        
        return self.human_diagnosis
    
    def get_predicted_diagnoses(self) -> dict:
        """Returns predicted diagnoses based on AASM critera"""
        return self.pred_diagnosis
    
    def confusion_matrix(self) -> np.ndarray:
        """Returns a confusion matrix for diagnoses of RSWA in the form:
            
             array([[TN, FP],
                   [FN, TP]])
        """
        return sk.confusion_matrix(y_true = self.y_true, y_pred = self.y_pred)
    
    def balanced_accuracy(self) -> float:
        """Returns balanced accuray score for diagnoses"""
        
        assert not self.predict_only, "Method cannot be used if predict_only option set to True!"
        
        return sk.balanced_accuracy_score(y_true = self.y_true, y_pred = self.y_pred)
    
    def cohen_kappa_diagnosis(self) -> float:
        """Calculates inter-rater agreement on diagnosis between human
        annotations and autoscorer predictions using Cohen's kappa"""
        
        assert not self.predict_only, "Method cannot be used if predict_only option set to True!"
        
        return sk.cohen_kappa_score(self.y_true,self.y_pred)
    
        
        