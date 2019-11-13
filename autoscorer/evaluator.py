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
        
        stride: The stride used when determining clinical diagnoses and
            inter-labeler agreement. The default stride is 30 s (the default
            epoch length)
                
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
                 stride: int = 30, single_ID: bool = False, 
                 single_subseq: bool = False, verbose: bool = True):
        
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
        self.stride = stride
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
                                               stride = self.stride,
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
        y_pred_label = np.zeros(int(len(self.y_pred)/(self.f_s*self.stride)))
        y_true_label = np.zeros(int(len(self.y_pred)/(self.f_s*self.stride)))
        for i in range(0,len(self.y_pred),self.f_s*self.stride):
            y_pred_label[i//(self.f_s*self.stride)] = self.label_mode(epoch = self.y_pred[i:i + (self.f_s*self.EPOCH_LEN)])
            y_true_label[i//(self.f_s*self.stride)] = self.label_mode(epoch = self.y_true[i:i + (self.f_s*self.EPOCH_LEN)])
        ck = sk.cohen_kappa_score(y_pred_label,y_true_label)
        # sk learn cohen kappa method returns a nan if there are only single labels
        #cohen's kappa can't handle single label cases
        if np.isnan(ck): return 1
        return ck

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
    
    def balanced_accuracy_score_diagnosis(self) -> float:
        """Returns balanced accuracy for RSWA diagnoses"""
        return self.clinical_scorer.balanced_accuracy_score()
    
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


#class Clinical_Scorer(object):
#    """Takes dictionary of predicted labels and actual labels as input and produces
#    a list of patients predicted (diagnosed) to have REM sleep behavior disorder.
#    
#    INPUT:  predictions: dictionary of predicted P and T events, 
#            annotations: human-annotated P and T events
#            
#                    
#        predictions and annotations should be in the format:
#                
#            predictions/annotations = {ID_0: {REM_0: array, REM_1: array, .. REM_n: array},
#            ID_1: {REM_0: array, REM_1: array, .. REM_n: array}, ...}}
#
#            where each array is an array of sample-by-sample RSWA signal-level
#            event scores
#                                 
#    PARAMETERS:
#            
#            offset: offset in seconds to use when scoring events. For example,
#            if offset set to 3s, first epoch will be considered as 3 -> 33s
#                        
#            predict_only: If this option is set to True, clinical_scorer will
#            return predicted diagnoses. When using the class in this mode,
#            only predictions should be provided as an input and the annotations
#            should not be passed to the function. No error metrics can be called
#            when using this option.
#    
#    Output:
#        
#        get_human_diagnoses method -> dictionary of human scorer-based diagnoses 
#        by patient ID
#        
#        get_autoscorer_diagnoses -> dictionary of automated scorer-based diagnoses 
#        by patient ID
#        
#        confusion_matrix method -> confusion matrix for diagnoses
#        
#        balanced_accuracy -> balanced accuracy for diagnoses
#        
#        cohen_kappa_diagnosis -> cohen's kappa for diagnoses
#        
#    """
#            
#    def __init__(self, predictions: dict, annotations: dict = None, offset: int = 0, 
#                 EPOCH_LEN: int = 30, f_s: int = 10, predict_only: bool = False,
#                 stride: int = 30, verbose: bool = True):
#        
#        if not predict_only:
#            assert type(predictions) == type(annotations) == dict, "predictions and annotations must be dictionaries!"
#        else:
#            assert type(predictions) == dict and annotations is None, "predictions must be a dictionary and annotations must be None if predict_only options is used!"
#        self.predictions = predictions
#        self.annotations = annotations
#        assert offset >= 0, "Offset must be greater than or equal to zero!"
#        assert type(offset) == int, "Offset must be an integer!"
#        offset = offset*f_s
#        self.offset = offset
#        self.EPOCH_LEN = EPOCH_LEN
#        self.f_s = f_s
#        self.stride = stride
#        self.verbose = verbose
#        self.predict_only = predict_only
#        self.rswa_epochs = {}
#        if not self.predict_only:
#            self.multilabel_clinical_score()
#            self.convert_to_vectors()
#        else:
#            self.predict_only()
#        
#    def predict_only(self):
#        """Generates diagnoses for each patient from the input dictionary of
#        signal-level annotations"""
#        
#        assert self.multilabel_track, "multilabel_track must be true to use this method!"
#        
#        self.pred_diagnosis = {}
#        for ID in self.predictions.keys():
#            pred_rswa = False
#            for subseq in self.predictions[ID].keys():
#                for i in range(self.offset,len(self.predictions[ID][subseq]), self.stride*self.f_s):
#                    if i+self.EPOCH_LEN*self.f_s > len(self.predictions[ID][subseq]):
#                        if self.verbose:
#                            print(f"With self.offset of {self.offset}, last {(len(self.predictions[ID][subseq]) - i)/self.f_s} seconds unscorable")
#                        break
#                    pred = self.predictions[ID][subseq][i:i+self.EPOCH_LEN*self.f_s]
#                    if self.check_for_tonic(seq = pred, tonic_symbol = 2) or self.check_for_phasic(seq = pred, phasic_symbol = 1):
#                        pred_rswa = True
#            self.pred_diagnosis[ID] = pred_rswa
#            if self.verbose:
#                        print(f"For {ID}:\t predicted RSWA: {pred_rswa}")
#        
#    def multilabel_clinical_score(self):
#        """Generates diagnoses for each patient from the predicted and human-generated
#        event annotations"""
#                
#        self.pred_diagnosis, self.human_diagnosis = {}, {}
#        for ID in self.annotations.keys():
#            pred_rswa = False
#            actual_rswa = False
#            for subseq in self.annotations[ID].keys():
#                for i in range(self.offset,len(self.annotations[ID][subseq]), self.stride*self.f_s):
#                    if i+self.EPOCH_LEN*self.f_s > len(self.annotations[ID][subseq]):
#                        if self.verbose:
#                            print(f"With self.offset of {self.offset}, last {(len(self.annotations[ID][subseq]) - i)/self.f_s} seconds unscorable")
#                        break
#                    actual = self.annotations[ID][subseq][i:i+self.EPOCH_LEN*self.f_s]
#                    pred = self.predictions[ID][subseq][i:i+self.EPOCH_LEN*self.f_s]
#                    if self.check_for_tonic(seq = actual, tonic_symbol = 2) or self.check_for_phasic(seq = actual, phasic_symbol = 1):
#                        actual_rswa = True
#                        if ID not in self.rswa_epochs.keys(): 
#                            self.rswa_epochs[ID] = {'human': [actual], 'predicted': []} 
#                        else: self.rswa_epochs[ID]['human'].append(actual)
#                    if self.check_for_tonic(seq = pred, tonic_symbol = 2) or self.check_for_phasic(seq = pred, phasic_symbol = 1):
#                        pred_rswa = True
#                        if ID not in self.rswa_epochs.keys(): 
#                            self.rswa_epochs[ID] = {'predicted': [pred], 'human': []} 
#                        else: self.rswa_epochs[ID]['predicted'].append(pred)
#            self.human_diagnosis[ID] = actual_rswa
#            self.pred_diagnosis[ID] = pred_rswa
#            if self.verbose:
#                        print(f"For {ID}:\t predicted RSWA: {pred_rswa},\t human-scored RSWA diagnosis: {actual_rswa}")
#                
#    def check_for_tonic(self, seq: np.ndarray, tonic_symbol: int = 2) -> bool:
#        """ Checks if tonic signal-level events meet AASM criteria. Takes sequence
#        and tonic_symbol as inputs and returns True if at least half of the elements
#        of the array are equal to tonic_symbol. Otherwise returns False.
#        Generally, an epoch of signal should be passed to this function.
#        """
#
#        if sum(seq == tonic_symbol) >= len(seq)/2: 
#            return True
#        return False
#        
#    def check_for_phasic(self, seq: np.ndarray, phasic_symbol: int = 1) -> bool:
#        """ Checks if phasic signal-level events meet AASM criteria. Takes sequence
#        and phasic_symbol as inputs and returns True if, after dividing the signal
#        into 3s mini-epochs, at least 5 of the mini-epochs contain a phasic event.
#        Otherwise, returns false. Generally, an epoch of signal should be passed 
#        to this function."""
#        
#        assert len(seq) % 2 == 0, f"Input array should be of even length, but length is {len(seq)}!"
#        
#        splits = np.split(ary=seq, indices_or_sections = 10)
#        
#        if sum([(l == phasic_symbol).any() for l in splits]) >= 5:
#            return True
#        return False
#    
#    def get_clinically_qualifying_sequences(self):
#        """Returns dictionary of epoch sequences meeting clinical RSWA diagnoses
#        in the format:
#            
#            {ID: {'human': [seq1, seq2, ...], 'predicted': [seq1, seq2, ...]}, }
#        """
#        
#        return self.rswa_epochs
#    
#    def convert_to_vectors(self):
#        """Creates vectors from instance attributes human_diagnosis and pred_diagnosis
#        that are used to generate error metrics"""
#        
#        assert not self.predict_only, "Method cannot be used if predict_only option set to True!"
#        
#        self.y_pred = np.zeros(len(self.human_diagnosis.keys()))
#        self.y_true = np.zeros(len(self.human_diagnosis.keys()))
#        for i, ID in enumerate(self.human_diagnosis.keys()):
#            self.y_pred[i] = int(self.pred_diagnosis[ID])
#            self.y_true[i] = int(self.human_diagnosis[ID])
#    
#    def get_human_diagnoses(self) -> dict:
#        """Returns human scorer-generated diagnoses based on AASM criteria"""
#        
#        assert not self.predict_only, "Method cannot be used if predict_only option set to True!"
#        
#        return self.human_diagnosis
#    
#    def get_predicted_diagnoses(self) -> dict:
#        """Returns predicted diagnoses based on AASM critera"""
#        return self.pred_diagnosis
#    
#    def balanced_accuracy_score_diagnosis(self) -> float:
#        """Returns balanced accuracy for RSWA diagnoses"""
#        return self.clinical_scorer.balanced_accuracy_score()
#    
#    def confusion_matrix(self) -> np.ndarray:
#        """Returns a confusion matrix for diagnoses of RSWA in the form:
#            
#             array([[TN, FP],
#                   [FN, TP]])
#        """
#        return sk.confusion_matrix(y_true = self.y_true, y_pred = self.y_pred)
#    
#    def accuracy_score(self) -> float:
#        """Returns balanced accuray score for diagnoses"""
#        
#        assert not self.predict_only, "Method cannot be used if predict_only option set to True!"
#        
#        return sk.accuracy_score(y_true = self.y_true, y_pred = self.y_pred)
#    
#    def cohen_kappa_diagnosis(self) -> float:
#        """Calculates inter-rater agreement on diagnosis between human
#        annotations and autoscorer predictions using Cohen's kappa"""
#        
#        assert not self.predict_only, "Method cannot be used if predict_only option set to True!"
#        
#        return sk.cohen_kappa_score(self.y_true,self.y_pred)
#
#def convert_to_rem_idx(time: float, rem_start_time: int, rem_end_time: int,
#                       f_s: int) -> int:
#    """Converts a time during REM to an index. Returns the index relative
#    to the start of the signal only containing REM.
#    """
#
#    assert rem_start_time <= time <= rem_end_time, f"Time {time} is not between REM start at {rem_start_time} and REM end at {rem_end_time}!"
#
#    time -= rem_start_time
#
#    seconds_idx = (time // 1)*f_s
#
#    frac_idx = np.floor((time % 1)*f_s)
#
#    return min(int(seconds_idx +  frac_idx), (rem_end_time - rem_start_time)*f_s - 1)
#    
#        
#        
#    
#                


            
            
        
