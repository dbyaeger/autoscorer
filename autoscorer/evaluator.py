#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 11:12:43 2019

@author: danielyaeger
"""

from autoscorer.clinical_scorer import Clinical_Scorer
from autoscorer.autoscorer_helpers import (adjust_rswa_event_times, 
                                convert_to_rem_idx,
                                sequence_builder,
                                collapse_p_and_t_events)
import sklearn.metrics as sk
import numpy as np
import collections

class Evaluator(object):
    """Takes as input dictionaries of predicted and human-annotated RSWA events,
    and calculates several error and inter-rater agreement metrics.
    
    INPUT:  predictions: dictionary of predicted P and T events, 
            annotations: human-annotated P and T events
            
        if sequence set to True:
                    
            predictions and annotations should be in the format:
                    
                predictions/annotations = {ID_0: {REM_0: array, REM_1: array, .. REM_n: array},
                ID_1: {REM_0: array, REM_1: array, .. REM_n: array}, ...}}

                where each array is an array of sample-by-sample RSWA signal-level
                event scores (i.e 000022222000011100000...)
                
                If array is a n samples X 3 matrix, (or vice versa), the sequence
                will automatically be converted into a 1-D array with 0's
                representing none_events, 1's representing phasics, and 2's 
                representing tonics. It is assumed that the first position in
                the dimension with length 3 has none_events in first position,
                phasic events in second position, and tonic events in third position.
                    
        if segmentation set to True:
            
            predictions and annotations should be in the format:
                    
                predictions/annotations = {'ID_0': {'REM_0': 'events': [event_list], 
                'rem_start_time': int, 'rem_end_time': int} ... }, 

                where event_list is a segmentation in the format
                [ (event_start, event_end, event_type), 
                (event_start, event_end, event_type), ... ]
            
                
    PARAMETERS:
        
        EPOCH_LEN: Length of epoch in seconds. Set to 30 by default.
        
        f_s: Sampling rate of EMG signals. Set to 10 by default.
        
        offset: offset in seconds to use when making diagnoses. For example,
            if offset set to 3s, first epoch will be considered as 3 -> 33s
        
        single_ID: If set to true, evaluator expects predictions/annotations
            for a single sleeper, i.e. in the format:
                
                {REM_0: array, REM_1: array, .. REM_n: array}
                
                or
                
                {'REM_0': {'events': [event_list], 
                'rem_start_time': int, 'rem_end_time': int},
                'REM_1': {'events': [event_list], 
                'rem_start_time': int, 'rem_end_time': int}, ... }
                
        verbose: If set to True, then diagnoses will be printed out.
            
    OUTPUTS:
        
        balanced_accuracy_signals method -> balanced accuracy for signal-level 
                                            (Phasic and Tonic) events
        
        cohen_kappa_epoch method -> inter-labeler agreement (based on predictions and
                              and annotations) on an epoch level
        
        accuracy_score_diagnosis method -> accuracy of diagnoses made by predictions
                                    using clinical rules and treating annotations
                                    as the ground truth.
        
    """
    
    def __init__(self, predictions: dict, annotations: dict, 
                 sequence: bool = True, segmentation: bool = False,
                 EPOCH_LEN: int = 30, f_s: int = 10, offset: int = 0,
                 single_ID: bool = False, single_subseq: bool = False,
                 verbose: bool = True):
        
        if not single_subseq:
            assert type(predictions) == type(annotations) == dict, "predictions and annotations must be dictionaries!"
        if single_subseq:
            assert type(predictions) == type(annotations) == np.ndarray, "predictions and annotations must be numpy arrays!"
            predictions = {'REM_0': predictions}
            annotations = {'REM_0': annotations}
        assert sequence ^ segmentation, "Only one of segmentation and multilabel track can be set to True"
        if single_subseq:
            assert single_subseq and single_ID, "Both single_subseq and single_ID must be set to true to use the single_subseq option!"
        self.sequence = sequence
        self.EPOCH_LEN = EPOCH_LEN
        self.f_s = f_s
        if single_ID:
            pred_temp, annot_temp = predictions.copy(), annotations.copy()
            del(predictions)
            del(annotations)
            predictions = {}
            annotations = {}
            predictions['ID'] = pred_temp
            annotations['ID'] = annot_temp
        self.ID_list = set(predictions.keys())
        self.offset = offset
        self.verbose = verbose
        if segmentation:
            self.predictions = self.convert_to_sequence(predictions)
            self.annotations = self.convert_to_sequence(annotations)
        else:
            self.predictions = {}
            self.annotations = {}
            for ID in predictions:
                self.predictions[ID] = {}
                self.annotations[ID] = {}
                for subseq in predictions[ID]:
                    self.predictions[ID][subseq] = self.convert_to_1D(predictions[ID][subseq])
                    self.annotations[ID][subseq] = self.convert_to_1D(annotations[ID][subseq])
        self.clinical_scorer = Clinical_Scorer(predictions = self.predictions,
                                               annotations = self.annotations,
                                               offset = self.offset,
                                               EPOCH_LEN = self.EPOCH_LEN,
                                               f_s = self.f_s,
                                               verbose = self.verbose)
        self.combine_IDs()
        
    def combine_IDs(self) -> None:
        """ Combines result_dict and annotation_dict across IDs and REM subsequences."""
        
        for ID in self.ID_list:
            rem_subseqs = self.predictions[ID].keys()
            for i,subseq in enumerate(rem_subseqs):
                if i == 0:
                    y_pred = self.predictions[ID][subseq]
                    y_true = self.annotations[ID][subseq]
                else:
                    y_pred = np.concatenate((y_pred, self.predictions[ID][subseq]))
                    y_true = np.concatenate((y_true, self.annotations[ID][subseq]))
        self.y_pred = y_pred
        self.y_true = y_true
                               
    def confusion_matrix_signals(self) -> np.ndarray or tuple:
        """Returns signal-level confusion matrix in the form 
        (if sequence set to False):
            
            array([[TN, FP],
                   [FN, TP]])
    
        Two confusion matrices will be returned, one for tonic and one for phasic
        in the format:
            
            (tonic confusion matrix, phasic confusion matrix)
            
        if sequence set to False
        
        if sequence set to True, confusion matrix is in the format:
            
            array([[True None, Predicted Phasic/Actually None, Predicted Tonic/Actually None],
                   [Predicted None/Actually Phasic, True Phasic, Predicted Tonic/Actually Phasic],
                   [Predicted None/Actually Tonic, Predicted Phasic/Actually Tonic, True Tonic]])
        """
        return sk.confusion_matrix(y_true = self.y_true, y_pred = self.y_pred)
    
    def confusion_matrix_diagnoses(self) -> np.ndarray:
        """Returns a confusion matrix for diagnoses of RSWA in the form:
            
             array([[TN, FP],
                   [FN, TP]])
        """
        return self.clinical_scorer.confusion_matrix()
    
    def balanced_accuracy_signals(self) -> float or tuple:
        """Returns the balanced accuracy score, either as a single value if
        sequence is set to True, or as separate scores for
        tonic and phasic events in the format:
            
            (tonic balanced accuracy score, phasic balanced accuracy score)
        
        if sequence is set to False"""
                
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
        
    def get_clinically_qualifying_sequences(self) -> dict:
        """Return epoch sequences that meet clinical criteria for RSWA"""
        return self.clinical_scorer.get_clinically_qualifying_sequences()
    
    def convert_to_1D(self, matrix: np.ndarray) -> np.ndarray:
        """Converts a matrix to a 1-D sequence assuming that the first position in
                the dimension with length 3 has none_events in first position,
                phasic events in second position, and tonic events in third position"""
        assert type(matrix) == np.ndarray, f"Matrix should be a numpy array, not a {type(matrix)}!"
        if len(matrix.shape) == 1:
            return matrix
        else:
            assert sum(matrix.shape) >= 3 + self.EPOCH_LEN*self.f_s, f"Matrix has shape {matrix.shape}, but should be at least {self.EPOCH_LEN*self.f_s,3} or {3,self.EPOCH_LEN*self.f_s}!" 
            try:
                event_axis = [i for i,d in enumerate(matrix.shape) if d == 3][0]
                return np.argmax(a=matrix,axis=event_axis)
            except:
                shape_error = Exception(f"Matrix does not have a dimension of size 3. Matrix has dimensions: {matrix.shape}!")
                raise shape_error
    
    def convert_to_sequence(self, event_dict: dict) -> None:
        """Converts prediction and annotation segmentations into sequences. 
        Although perhaps undesirable computationally, many of the metrics 
        require conversion to sequences.
        
        INPUT: event_dict in the format:
            
            {'ID_0': 'REM_0': {'events': [event_list], 
                'rem_start_time': int, 'rem_end_time': int}, 

                where event_list is a segmentation in the format
                [ (event_start, event_end, event_type), 
                (event_start, event_end, event_type), ... ]
        
        OUTPUT: seq_dict in the format:
            
            {'ID_0': 'REM_0': array, 'REM_1': array, ...,} 'ID_2': { ...} )
            
            where each array is an array of sample-by-sample RSWA signal-level
                event scores (i.e 000022222000011100000...)
                """
        seq_dict = {}
        for ID in self.ID_list:
            seq_dict[ID] = {}
            rem_subseqs = event_dict[ID].keys()
            for i,subseq in enumerate(rem_subseqs):
                rem_start_time = event_dict[ID][subseq]['rem_start_time']
                rem_end_time = event_dict[ID][subseq]['rem_end_time']
                event_times = adjust_rswa_event_times(time_list = event_dict[ID][subseq]['events'],
                                                  rem_start_time = rem_start_time,
                                                  rem_end_time = rem_end_time)
                p_idx, t_idx = [], []
                for event in event_times:
                    if event[-1] == 'P':
                        # Convert event times into indexes relative to start of REM
                        p_idx.append(list(np.arange(start = convert_to_rem_idx(time = event[0], 
                                                                               rem_start_time = rem_start_time, 
                                                                               rem_end_time = rem_end_time,
                                                                               f_s = self.f_s),
                                                    stop = convert_to_rem_idx(time = event[1], 
                                                                               rem_start_time = rem_start_time, 
                                                                               rem_end_time = rem_end_time,
                                                                               f_s = self.f_s) + 1)))
                    elif event[-1] == 'T':
                        t_idx.append(list(np.arange(start = convert_to_rem_idx(time = event[0], 
                                                                               rem_start_time = rem_start_time, 
                                                                               rem_end_time = rem_end_time,
                                                                               f_s = self.f_s),
                                                    stop = convert_to_rem_idx(time = event[1], 
                                                                               rem_start_time = rem_start_time, 
                                                                               rem_end_time = rem_end_time,
                                                                               f_s = self.f_s))))
                rem_length_in_idx = int((rem_end_time - rem_start_time) * self.f_s)
                p_seq = sequence_builder(groups = p_idx, length = rem_length_in_idx, phasic_start_time_only = False)
                t_seq = sequence_builder(groups = t_idx, length = rem_length_in_idx, phasic_start_time_only = False)
                combined_array, _ = collapse_p_and_t_events(t_events = t_seq, p_events = p_seq, 
                                                            tuples = False, f_s = self.f_s)
                seq_dict[ID][subseq] = combined_array
        return seq_dict
    
                


            
            
        