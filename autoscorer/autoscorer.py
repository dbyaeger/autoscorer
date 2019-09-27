#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 09:38:26 2019

@author: danielyaeger
"""

import numpy as np
import more_itertools
from pathlib import Path
from autoscorer.autoscorer_helpers import (get_data,
                                      make_event_idx,
                                      convert_to_rem_idx,
                                      tuple_builder,
                                      sequence_builder,
                                      round_time,
                                      adjust_rswa_event_times,
                                      collapse_p_and_t_events,
                                      make_matrix_event_track)

class Autoscorer(object):
    """ Returns the times of T events and P events scored according to
    *interpretations* of the AASM guidelines for a given sleep study. Also
    returns the human-annotated phasic and tonic event times.

    INPUT: Sleeper ID (ID) and path to data files (data_path). Assumes
    data files are pickled dictionaries of the format:

        {
            "ID":ID (str),
            "study_start_time": study start time in datatime format
            "staging": [(start_time_sleep_stage_i, end_time_sleep_stage_i, sleep_stage_type_i), ...] (float, float, str)
            "apnia_hypopnia_events":[(start_time_event_i, end_time_event_i, event_type_i), ...] (float, float, str)
            "rswa_events":[(start_time_event_i, end_time_event_i, event_type_i), ...] (float, float, str)
            "signals":{"channel_i":array(n_seconds, sampling_rate)}
        }

    OUTPUT: Dictionary of scored REM subsequences in the format:
        {'RSWA_P': {'REM_0': {scores}, .., 'REM_n': {scores}},
        'RSWA_T': {'REM_0': {scores}, .., 'REM_n': {scores}}}

    PARAMETERS:
        f_s: Sampling rate in Hz. The signal data is assumed to have the dimensions
            seconds by sampling rate. An assertion error will be raised if f_s
            is set incorrectly.

        t_amplitude_threshold: the multiple of the non_REM minimum that the
            elements in data must exceed in order to be given a value of one.
            This value should be greater than zero.

        t_continuity threshold: the number of consecutive samples above the
            t_amplitude_threshold that must be observed in order for a sequence of
            elements in data to be considered as a qualifying tonic signal-level
            event.

        p_mode: Can be set to 'mean', 'quantile', or 'stdev'.

            'mean': phasic amplitudes must exceed the product of the
            mean of the REM baseline and p_amplitude_threshold in order to be
            considered a phasic signal-level event.

            'quantile': phasic amplitudes must exceed the value given
            by a quantile of the REM baseline (given by p_quantile) in order to
            be considered a phasic signal-level event.

            'stdev': phasic amplitudes must be p_amplitude_threshold
            standard deviations above the mean of the REM baseline in order to
            be considered a phasic signal-level event.

        p_amplitude_threshold: the mulitple by which phasic ampltudes must
            exceed the mean, p_quantile, (or the number of standard deviations 
            by which the phasic amplitudes must exceed the mean, depending on 
            p_mode) in order to be considered a phasic signal-level event.

        p_quantile: the quantile of the REM baseline signal that must be
            exceeded (by p_amplitude_threshold times) in order for a phasic 
            signal to be considered a phasic event when p_mode is set to 'quantile.'

        p_continuity_threshold: the minimum number of consecutive samples
            exceeding the phasic amplitude threshold in order for a signal to
            be considered a phasic singal-level event.

        p_baseline_length: The number of seconds that should be considered
            as the baseline when scoring phasic events. Must be a number greater
            than zero. When a sample is being considered that is more than
            p_baseline_length seconds from the start of the REM subsequence,
            the previous p_baseline_length seconds are considered from the current
            sample. When a sample is less than p_baseline_length seconds from
            the start of the REM subsequence, the first p_baseline_length seconds
            are used as the baseline.

        ignore_hypoxics_duration: The number of seconds before and after hypopnea
            or apnea events that are ignored. For instance, if set to 0, then
            only phasic and tonic events that occur during the specified
            interval are ignored. Typical values are 0 - 30.

        return_seq: If set to True, a sequence of 1's and 0's is returned in
            which a 1 indicates a qualifying Tonic or Phasic event and a 0
            indicates a non-event. There is one value per sample in the
            input signal data. The tonic sequence is 1-D because tonic events
            are scored using only Chin EMG channels. The phasic sequence is
            either a 1-D sequence in which the sequences from the L Leg, R Leg,
            and Chin channels are collapsed into a single sequence (if
            return_concat is True) or is a length of signal X 3 array, in which
            the first row corresponds to the 'Chin' channel, the second to the
            'L Leg' and the third to the 'R Leg' (if return_concat is False).
            One, and only one, of return_seq and return_tuple must be set to True.

        return_concat: If set to False, a length of signal X 3 array is returned,
            in which each row corresponds to the signal-level phasic events
            in each channel ('Chin', 'L Leg', and 'R Leg', respectively). If false,
            the signal length X 3 array is collapsed into a 1-D array by giving
            each row in the 1-D array a value of 1 if the 3-D array contains
            a 1 anywhere in the corresponding column and a value of 0 if the column
            only contains zeros. An assertion error will be raised if return_concat
            is set to True and return_tuple is set to True.

        return_tuple: If set to true, signal-level events are returned as a list
            of tuples. Each tuple follows the format:

                (event_start_time, event_end_time, event_type)

            Event_start_time and Event_end_times are in units of study time.
            One, and only one, of return_seq and return_tuple must be set to True.
        
        phasic_start_time_only: If set to true, only the start time of a phasic
            event will be returned. For instance, if return_tuples is set to true,
            events will be returned in the format:
                
                (event_start_time, event_start_time, event_type)
        
        return_multilabel_track: If set to true, tonic and phasic events will
            be collapsed into a single track for both the annotations and the
            scored results. Tonic events are given precedence over phasic events.
            
        return_matrix_event_track: If set to true, events will be represented by
            a matrix with dimension number of samples X 3, in which the first
            column represents no events, the second column is P events, and the
            third column is T events.
        
        verbose: If set to True, ID and REM subsequence number will be printed
            out during scoring.

    """

    def __init__(self, ID= 'XVZ2FFAEC864IPK', data_path = '/Users/danielyaeger/Documents/processed_data/processed',
                 f_s = 10, t_amplitude_threshold = 10,
                 t_continuity_threshold = 10, p_mode = 'quantile',
                 p_amplitude_threshold = 1, p_quantile = 0.5,
                 p_continuity_threshold = 1, p_baseline_length = 120,
                 ignore_hypoxics_duration = 15, return_seq = False,
                 return_concat = False, return_tuple = True,
                 phasic_start_time_only = False, return_multilabel_track = True,
                 return_matrix_event_track = True, verbose = True):
        self.ID = ID
        if type(data_path) == str:
            data_path = Path(data_path)
        self.data_path = data_path
        self.f_s = f_s

        assert t_amplitude_threshold >= 0, "T Ampltiude threshold should be greater than or equal to zero!"
        self.t_amplitude_threshold = t_amplitude_threshold

        assert t_continuity_threshold >= 1, "T Continuity threshold should be greater than or equal to one!"
        self.t_continuity_threshold = t_continuity_threshold

        assert p_continuity_threshold >= 1, "P Continuity threshold should be greater than or equal to one!"
        self.p_continuity_threshold = p_continuity_threshold

        assert p_amplitude_threshold >= 0, "P Amplitude threshold should be greater than or equal to zero!"
        self.p_amplitude_threshold = p_amplitude_threshold

        assert 0 <= p_quantile <= 1, "P quantile threshold must be between 0 and 1!"
        self.p_quantile = p_quantile

        assert p_mode in ['quantile', 'mean', 'stdev']
        self.p_mode = p_mode

        # Baseline length is in epochs
        assert p_baseline_length > 0, "Baseline length must be greater than 0!"
        self.p_baseline_length = p_baseline_length

        assert ignore_hypoxics_duration >= 0, "ignore_hypoxics_duration must be greater than or equal to 0!"
        self.ignore_hypoxics_duration = ignore_hypoxics_duration

        assert return_seq ^ return_tuple, "Only one return_seq or return_tuple option can be chosen!"

        if return_tuple:
            assert return_concat == False, "Return_concat must be False when return_tuple option set to true!"
        
        if return_seq and return_multilabel_track:
            assert return_concat, "return_concat must be set to true when return_seq and return_multilabel_track set to True"
        
        if return_matrix_event_track:
            assert return_seq and return_multilabel_track, "return_seq and return_multilabel_track must be set to true when return_matrix_event_track set to True!"

        self.return_concat = return_concat
        self.return_seq = return_seq
        self.return_tuple = return_tuple
        self.verbose = verbose
        self.phasic_start_time_only = phasic_start_time_only
        self.return_multilabel_track = return_multilabel_track
        self.return_matrix_event_track = return_matrix_event_track
        self.fields = ['ID', 'study_start_time', 'staging', 'apnia_hypopnia_events', 'rswa_events', 'signals']
        self.channels = ['Chin', 'L Leg', 'R Leg']
        self.make_dicts()

    def score_REM(self):
        """ For each REM subsequence in directory, scores tonic and phasic events.
        Stores scoring in results_dict. Also creates a dictionary of human-annotated
        P and T events. Returns dictionary of results.
        """
        self.rem_subseq = 0
        while self.data_path.joinpath(f"{self.ID}_{self.rem_subseq}.p").exists():
            if self.verbose:
                print(f"Processing ID: {self.ID} REM subsequence: {self.rem_subseq}")
            data = get_data(filename = f"{self.ID}_{self.rem_subseq}.p",
                                    data_path = self.data_path, channels = self.channels,
                                    f_s = self.f_s, fields = self.fields)
            self.set_times(data)
            self.p_event_idx = make_event_idx(channels = self.channels)
            # add a and h indices to p_event_idx
            if len(self.a_and_h_idx) > 0: self.add_a_and_h_to_p_event_idx()
            self.add_human_annotations(data)
            self.results_dict['RSWA_P'][f'REM_{self.rem_subseq}'] = self.findP_over_threshold(data)
            if self.verbose:
                print("\tFinished analyzing phasic events...")
            self.results_dict['RSWA_T'][f'REM_{self.rem_subseq}'] = self.score_Tonics(data)
            if self.verbose:
                print("\tFinished analyzing tonic events...")
            self.rem_subseq += 1
        if self.return_multilabel_track: self.collapse_rswa_events()
        return self.results_dict
    
    def add_a_and_h_to_p_event_idx(self):
        """Adds events in the a_and_h_idx to the p_event_idx so that the events
        will be excluded from baseline for P events"""
        for event in self.a_and_h_idx:
            for channel in self.channels:
                self.p_event_idx[channel].extend(np.arange(event[0],event[1]+1))
    
    def collapse_rswa_events(self) -> None:
        """Changes results_dict and annotations_dict so that T and P events are
        collapsed into a single track. If return_seq set to True (i.e. results
        and annotations are returned as sequences, tonic events are coded by 2,
        phasic events are coded by a 1, and none-events are coded by a 0)
        """
        self.collisions = 0
        multilabel_results, multilabel_annotaions = {}, {}
        subseq_idxs = sorted(list(self.results_dict['RSWA_T'].keys()))
        for subseq_idx in subseq_idxs:
            multilabel_results[subseq_idx], collisions = collapse_p_and_t_events(t_events = self.results_dict['RSWA_T'][subseq_idx],
                                                                                  p_events = self.results_dict['RSWA_P'][subseq_idx],
                                                                                  tuples = self.return_tuple,
                                                                                  f_s = self.f_s)
            self.collisions += collisions
            multilabel_annotaions[subseq_idx], _ = collapse_p_and_t_events(t_events = self.annotation_dict['RSWA_T'][subseq_idx],
                                                                                  p_events = self.annotation_dict['RSWA_P'][subseq_idx],
                                                                                  tuples = self.return_tuple,
                                                                                  f_s = self.f_s)
        if not self.return_matrix_event_track:
            self.results_dict, self.annotation_dict = multilabel_results, multilabel_annotaions
        else:
            self.results_dict, self.annotation_dict = make_matrix_event_track(multilabel_results), make_matrix_event_track(multilabel_annotaions)
    
    def get_annotations(self) -> dict:
        """ Returns dictionary of human annotations of tonic and phasic signal-
        level events"""
        return self.annotation_dict
    
    def get_collisions(self) -> int:
        """ Returns number of collisions of phasic and tonic events. The number
        of collisions is the number of overlapping phasic and tonic events if
        scoring results are returned as tuples (return_tuple set to True). If
        scoring results are returned as sequences (return_seq set to True), then
        the number of collisions is the number of samples which were predicted
        as both phasic and tonic"""
        assert self.return_multilabel_track, "Collisions are not calculated if return_multilabel_track option set to False!"
        return self.collisions
        

    def set_times(self, data: dict) -> None:
        """Sets the rem_start_time, rem_end_time, nrem_start_time, and
        nrem_end_time instance attributes.

        NOTE: There may be a gap between nrem_end_time and rem_start_time. This
        may occur if the patient woke up for 1 or more epochs. For the purposes
        of indexing into the signal, nrem_end_time - nrem_start_time should be
        used as the beginning of the REM period, whereas rem_start_time should
        be used to convert the index of identified phasic and tonic events
        into units of study time.

        Returns None"""

        assert 0 < len(data['staging']) < 3, f"Staging array length must be between 1 and 2, not {len(data['staging'])}!"

        for tup in data['staging']:
            assert tup[0] < tup[1], f"Start time in tuple {tup} not less than end time!"

        rem_time = [key for key in data['staging'] if key[2] == 'R'][0]
        self.rem_start_time, self.rem_end_time = rem_time[0], rem_time[1]

        try:
            nrem_time = [key for key in data['staging'] if key[2] == 'N'][0]
            self.nrem_start_time, self.nrem_end_time = nrem_time[0], nrem_time[1]
        except:
            self.nrem_start_time = self.nrem_end_time = None

        # Make list of apnea and hypoapnea indices
        self.make_a_and_h_list(data)


    def make_a_and_h_list(self, data: dict) -> None:
        """Creates a list of indices of tuples as an instance variable called
        a_and_h_idx, which is a list of apnea and hypoapneas occuring during
        REM sleep in the format (start index, end index). Returns None."""
        # Create list of tuples of apnea and hypoapnea start and end times
        self.a_and_h_idx = []
        if len(data['apnia_hypopnia_events']) >= 0:
            #print(f"apnia_hypopnia_events: {data['apnia_hypopnia_events']}")
            for event in data['apnia_hypopnia_events']:
                if event[1] + self.ignore_hypoxics_duration <= self.rem_start_time:
                    continue
                else:
                    if (event[0] - self.ignore_hypoxics_duration) < self.rem_start_time:
                        start = self.rem_start_time
                    else:
                        start = event[0] - self.ignore_hypoxics_duration
                    if (event[1] + self.ignore_hypoxics_duration) > self.rem_end_time:
                        end = self.rem_end_time
                    else:
                        end = event[1] + self.ignore_hypoxics_duration
                    self.a_and_h_idx.append((convert_to_rem_idx(time = start,
                                                                rem_start_time = self.rem_start_time,
                                                                rem_end_time = self.rem_end_time,
                                                                f_s = self.f_s),
                                             convert_to_rem_idx(time = end,
                                                                rem_start_time = self.rem_start_time,
                                                                rem_end_time = self.rem_end_time,
                                                                f_s = self.f_s)))


    def make_dicts(self):
        """ Scans through the data_path directory and finds all REM subsequence
        files associated with the given ID. Builds the results dictionary using
        the format:
            {'RSWA_P': {'REM_0': {} .. 'REM_1': {}},
            'RSWA_T': {'REM_0': {} .. 'REM_1': {}}}

        Also builds a baseline dictionary at the same time with the format:
            {'RSWA_P': {'REM_0': {'Chin': {}, 'L Leg': {}, 'R Leg': {}}},
            .. 'REM_1': {'Chin': {}, 'L Leg': {}, 'R Leg': {}}, ... ,
            'RSWA_T': {'REM_0': {} .. 'REM_1': {}}}
        
        Also builds a dictionary of the human-annotated RSWA events with the
        format:
            {'RSWA_P': {'REM_0': {} .. 'REM_1': {}},
            'RSWA_T': {'REM_0': {} .. 'REM_1': {}}}
            
        """
        self.results_dict = {'RSWA_P': {}, 'RSWA_T': {}}
        self.baseline_dict = {'RSWA_P': {}, 'RSWA_T': {}}
        self.annotation_dict = {'RSWA_P': {}, 'RSWA_T': {}}
        for file in list(self.data_path.glob('**/*.p')):
            if self.ID in file.stem:
                num = file.stem.split('_')[1]
                self.results_dict['RSWA_P'][f'REM_{num}'] = []
                self.results_dict['RSWA_T'][f'REM_{num}'] = []
                self.baseline_dict['RSWA_P'][f'REM_{num}'] = {}
                for channel in self.channels:
                    self.baseline_dict['RSWA_P'][f'REM_{num}'][channel] = []
                self.baseline_dict['RSWA_T'][f'REM_{num}'] = []
                self.annotation_dict['RSWA_P'][f'REM_{num}'] = []
                self.annotation_dict['RSWA_T'][f'REM_{num}'] = []
                
    def add_human_annotations(self, data: dict):
        """ Adds human annotations to the instanc attribute annotation_dict. Both
        P and T event annotations are added for each REM subsequence. If the
        return_tuples option is chosen, then tuples will be returned. If return_seq
        option is chosen, then sequences will be returned.
        """
        # Sometimes annotated event end time after REM end time and need to correct
        event_times = adjust_rswa_event_times(time_list = data['rswa_events'],
                                              rem_start_time = self.rem_start_time,
                                              rem_end_time = self.rem_end_time)
        if self.return_tuple:
            for event in event_times:
                if event[-1] == 'P': 
                    self.annotation_dict['RSWA_P'][f"REM_{self.rem_subseq}"].append(round_time(times = event,
                                                                                    f_s = self.f_s,
                                                                                    phasic_start_time_only = self.phasic_start_time_only))
                elif event[-1] == 'T':
                    self.annotation_dict['RSWA_T'][f"REM_{self.rem_subseq}"].append(round_time(times = event,
                                                                                    f_s = self.f_s,
                                                                                    phasic_start_time_only = self.phasic_start_time_only))
        
        elif self.return_seq:
            p_idx, t_idx = [], []
            for event in event_times:
                if event[-1] == 'P':
                    # Convert event times into indexes relative to start of REM
                    p_idx.append(list(np.arange(start = convert_to_rem_idx(time = event[0], 
                                                                           rem_start_time = self.rem_start_time, 
                                                                           rem_end_time = self.rem_end_time,
                                                                           f_s = self.f_s),
                                                stop = convert_to_rem_idx(time = event[1], 
                                                                           rem_start_time = self.rem_start_time, 
                                                                           rem_end_time = self.rem_end_time,
                                                                           f_s = self.f_s) + 1)))
                elif event[-1] == 'T':
                    t_idx.append(list(np.arange(start = convert_to_rem_idx(time = event[0], 
                                                                           rem_start_time = self.rem_start_time, 
                                                                           rem_end_time = self.rem_end_time,
                                                                           f_s = self.f_s),
                                                stop = convert_to_rem_idx(time = event[1], 
                                                                           rem_start_time = self.rem_start_time, 
                                                                           rem_end_time = self.rem_end_time,
                                                                           f_s = self.f_s))))
            rem_length_in_idx = int((self.rem_end_time - self.rem_start_time) * self.f_s)
            self.annotation_dict['RSWA_P'][f"REM_{self.rem_subseq}"] = sequence_builder(groups = p_idx, 
                                                                        length = rem_length_in_idx,
                                                                        phasic_start_time_only = self.phasic_start_time_only)
            self.annotation_dict['RSWA_T'][f"REM_{self.rem_subseq}"] = sequence_builder(groups = t_idx, 
                                                                        length = rem_length_in_idx,
                                                                        phasic_start_time_only = False)

    def set_rem_baseline(self, data: np.ndarray, channel: str,
                         index: int) -> None:
        """ Calculates the baseline REM sleep period given the array of EMG values
        and the index.

        Case 1: The first REM subsequence is being analyzed, and the index
        corresponds to a time of less than the value of the p_baseline_length
        parameter. The first p_baseline_length seconds of the array are
        used as baseline.
        
            Note: Case 1 can also occur in a later REM subsequence if there is
            no baseline available from earlier REM subsequences, which occurs
            when there are apnea and hypoapnea events that occlude the baseline
            of the earlier REM subsequences.

        Case 2: A REM subsequence other than the first is being analyzed, and the
        index to a time of less than the value of the p_baseline_length
        parameter. The baseline consists of the last f_s * p_baseline_length -
        index samples from the previous REM subsequence baseline plus the first
        index samples from the current REM subsequence.

        Case 3: The index corresponds to a time of less than the value of the
        p_baseline_length parameter. The baseline consists of the last p_baseline_length
        worth of samples (i.e. p_baseline_length seconds, or f_s * p_baseline_length
        samples).

        Events are filtered out of the baseline by the function delete_events_from_baseline,
        so the length of the baseline will only correspond to p_baseline_length
        seconds when no phasic events have been detected in the last p_baseline_length
        seconds.

        Returns None
        """

        assert index < len(data), "Index must be less than the length of the data array!"

        if (index - (self.f_s * self.p_baseline_length)) <= 0:
            limit_lo = 0
            if self.rem_subseq == 0:
                baseline = self.delete_events_from_baseline(data=data[0:self.f_s * self.p_baseline_length],
                        channel = channel, limit_lo = limit_lo)
            else:
                baseline_from_current = self.delete_events_from_baseline(data = data[0:index],
                        channel = channel, limit_lo = limit_lo)
                i = self.rem_subseq-1
                baseline_from_last = self.baseline_dict['RSWA_P'][f'REM_{i}'][channel][-(self.f_s * self.p_baseline_length - index):]
                # Make sure that baseline is non-zero
                while len(baseline_from_last) == 0:
                    i -= 1
                    if i < 0:
                        # Baseline of 0th REM subsequence is missing so use next two minutes of current subsequence
                        baseline_from_current = self.delete_events_from_baseline(data=data[0:self.f_s * self.p_baseline_length],
                        channel = channel, limit_lo = limit_lo)
                        break
                    else: 
                        baseline_from_last = self.baseline_dict['RSWA_P'][f'REM_{i}'][channel][-(self.f_s * self.p_baseline_length - index):]
                baseline = np.concatenate((baseline_from_last,baseline_from_current))
        else:
            baseline = self.delete_events_from_baseline(data=data[index-self.p_baseline_length*self.f_s:index],
                        channel = channel, limit_lo = index-self.p_baseline_length*self.f_s)

        self.baseline_dict['RSWA_P'][f'REM_{self.rem_subseq}'][channel] = baseline

    def delete_events_from_baseline(self, data: np.ndarray, channel: str,
                                    limit_lo: int) -> np.ndarray:
        """Deletes indices from a data array. The indices are specified by
        the p_event_idx dictionary, which is an instance attribute of
        rule_based_scorer. The limit_lo parameter is the start index of the
        data array with reference to the EMG signal array and the channel
        parameter is the channel that is being analyzed."""
        
        index = len(data) - 1

        if len(self.p_event_idx[channel]) > 0:
            # Get previous events and convert to an array
            events = list(set(self.p_event_idx[channel][:]))

            # Only consider events occuring in baseline
            events = np.array(list(filter(lambda x: limit_lo <= x <= index, events)))
            events -= limit_lo

            # delete events from baseline
            np.delete(data, events)

        return data


    def findP_over_threshold(self, data: dict):
        """Transforms EMG signal for each channel into sequences of ones and
        zeros, or tuples of (start_time, end_time, 'RSWA_P') depending on
        whether EMG signal meets criteria specified by the input arguments
        p_continuity_threshold and p_amplitude_threshold/p_quantile.

        If p_mode is quantile, the signal must exceed the specified p_quantile
        of the baseline REM to exceed the ampltiude threshold. If p_mode is
        amplitude, the signal must exceed the product of the baseline mean
        and p_amplitude_threshold. If p_mode is stdev, the signal must exceed
        the baseline plus the specified number of standard deviations.

        INPUT: data, a dictionary of signal values.

        OUTPUT: if return_seq, the output is a sequence of 1's and 0's, one for
        each sample point in the input data array (for each channel).
        If return_tuple the output
        is a list of tuples of the form (Start_sample, end_sample, event_type)
        """
        assert type(data) == dict, f"Input data type must be a dictionary, not a {type(data)}!"

        if self.return_seq:
            if self.nrem_start_time is None:
                out = np.zeros((len(data['signals']['Chin'].ravel()),len(self.channels)))
            else:
                out = np.zeros((len(data['signals']['Chin'][self.nrem_end_time - self.nrem_start_time:].ravel()),len(self.channels)))
        else:
            out = []

        # Check if signal in each channel exceeds amplitude criterion
        for i,channel in enumerate(self.channels):
            if self.nrem_start_time is None:
                signal = data['signals'][channel].ravel()
            else:
                signal = data['signals'][channel][self.nrem_end_time - self.nrem_start_time:].ravel()
            # Mask events during hypoapnea and apnea
            if i == 0:
                rem_indices = np.arange(0,len(signal)-1)
                if len(self.a_and_h_idx) > 0:
                    for event in self.a_and_h_idx:
                        rem_indices = np.array(list(filter(lambda x: x < event[0] or x > event[1], rem_indices)))
            chan_out = np.zeros(len(signal))
            for j in range(len(rem_indices)):
                x = rem_indices[j]
                chan_out[x] = self.p_amplitude_checker(index = x, data = signal, channel = channel)
                # check if an event occured in the past and add to p_event_idx
                if x >= (self.p_continuity_threshold - 1):
                    if sum(chan_out[x-self.p_continuity_threshold+1:x+1]) >= self.p_continuity_threshold:
                        event_idx = list(np.arange(x-self.p_continuity_threshold+1,x+1))
                        self.p_event_idx[channel].extend(event_idx)
            # Remove events under continuity criterion
            idx = np.nonzero(chan_out > 0)
            groups = self.continuity_thresholder(index = idx[0], event_type ='RSWA_P')
            if self.return_tuple:
                out.extend(tuple_builder(groups = groups,
                                         event_type ='RSWA_P',
                                         f_s = self.f_s,
                                         rem_start_time = self.rem_start_time,
                                         rem_end_time = self.rem_end_time,
                                         phasic_start_time_only = self.phasic_start_time_only))
            elif self.return_seq:
                out[:,i] = sequence_builder(groups = groups,
                                           length = len(chan_out),
                                           phasic_start_time_only = self.phasic_start_time_only)
        if not self.return_concat:
            return out
        else:
            return np.amax(a=out,axis=1)

    def p_amplitude_checker(self, index: int, data: np.ndarray, channel: str) -> int:
        """ Takes the index of the data array, the data array, and the channel
        as inputs. Checks if the value of data at the index exceeds a metric
        computed based on the baseline. The metric is calculated according to the
        p_mode argument (quantile, mean, or stdev). Returns a one if data at the
        index value exceeds the metric and a zero otherwise.
        """
        # update baseline
        self.set_rem_baseline(data = data, channel = channel, index = index)
        baseline = self.baseline_dict['RSWA_P'][f'REM_{self.rem_subseq}'][channel]
        if self.p_mode == 'quantile':
            if data[index] >= self.p_amplitude_threshold*np.quantile(baseline, self.p_quantile): 
                return 1
        if self.p_mode == 'mean':
            if data[index] >= np.mean(baseline)*self.p_amplitude_threshold:
                return 1
        if self.p_mode == 'stdev':
            if data[index] >= (np.std(baseline)*self.p_amplitude_threshold + np.mean(baseline)):
                return 1
        return 0

    def get_nrem_baseline(self, signal) -> None:
        """ Calculates the 5th quantile in the baseline Non-REM sleep Chin EMG.
        Assumes baseline is composed of only non-REM sleep. Updates the
        instance attribute baseline_dict with the minimum of Chin EMG channel
        during baseline period. If insufficient baseline length is encountered
        for the current REM subsequence, the baseline from the last REM
        subsequence will also be used.

        INPUT:
            baseline: None or tuple in format (start, end, N)
            signal: None or numpy array

        OUTPUT:
            None.
        """
        if signal is not None:
            assert type(signal) == np.ndarray, f"Signal must a numpy array if baseline is a tuple, not a {type(signal)}!"
            self.baseline_dict['RSWA_T'][f'REM_{self.rem_subseq}']= np.quantile(a = signal.ravel(), q= 0.10)

        if signal is None:
            if self.rem_subseq > 0:
                self.baseline_dict['RSWA_T'][f'REM_{self.rem_subseq}'] = self.baseline_dict['RSWA_T'][f'REM_{self.rem_subseq-1}']
            else:
                nobaseline = Exception("No baseline available for first REM subsequence!")
                raise nobaseline

    def findT_over_threshold(self, data: dict):

        """ Transforms EMG signal into sequence of ones and zeros or tuples
        of (start_time, end_time, 'RSWA_T') depending on whether EMG signal
        meets criteria specified by the input arguments t_continuity_threshold
        and t_amplitude_threshold.

        INPUT: data, a 1-D array, and nrem_min, the minimum signal observed
        in the non-REM baseline period.

        OUTPUT: if return_seq, the output is a sequence of 1's and 0's, one for
        each sample point in the input data array. If return_tuple the output
        is a list of tuples of the form (Start_sample, end_sample, event_type)
        """

        assert len(data) > 0, "Data is empty!"

        assert len(data.shape) == 1, "Data must be a flattend array!"

        nrem_min = self.baseline_dict['RSWA_T'][f'REM_{self.rem_subseq}']

        assert type(nrem_min) == np.float64, f"Baseline data unavailable! Baseline: {nrem_min}"
        
        # Make copy of signal to avoid altering it
        signal = data.copy()

        # Mask indices during apneas and hypoapneas
        for event in self.a_and_h_idx:
            signal[event[0]:event[1]+1] = float("-Inf")

        # Calculate indices for values over threshold
        idx = np.nonzero(signal > self.t_amplitude_threshold*nrem_min)[0]

        groups = self.continuity_thresholder(index = idx, event_type ='RSWA_T')

        if self.return_tuple: return tuple_builder(groups = groups,
                                        event_type ='RSWA_T',
                                        f_s = self.f_s,
                                        rem_start_time = self.rem_start_time,
                                        rem_end_time = self.rem_end_time,
                                        phasic_start_time_only = False)
        
        elif self.return_seq: return sequence_builder(groups = groups,
                                        length = len(data),
                                        phasic_start_time_only = False)

    def continuity_thresholder(self,index: int, event_type: str) -> list:
        """Takes an array of indices as input. Returns an array in which all
        indices occur in runs of consecutive indices that exceed the continuty
        threshold for a given event type (RSWA_P or RSWA_T)

        INPUT: Array of indices, e.g array([1, 2, 3, 89, 90, 100])

        OUTPUT: List of arrays of indices filterd based on continuity threshold. For
        example, if continuity threshold were 2, the input array would be
        transformed to [array([1, 2, 3]), array([89,90])]"""

        assert event_type in ['RSWA_T', 'RSWA_P'], f"Event type must be RSWA_T or RSWA_P, not {event_type}!"

        assert type(index) == np.ndarray, f"Index must be a numpy ndarray, not a {type(index)}!"

        if event_type == 'RSWA_T': continuity_threshold = self.t_continuity_threshold

        if event_type == 'RSWA_P': continuity_threshold = self.p_continuity_threshold

        # get list of consecutive samples above threshold
        groups = [list(group) for group in more_itertools.consecutive_groups(index)]

        # apply continuity threshold to groups of samples
        groups = [group for group in groups if len(group) >= continuity_threshold]

        return groups

    def score_Tonics(self, data: dict) -> dict:
        """Takes in data and returns either a list of tuples of the start
        and end times of tonic events or a binary sequence in which 1 indicates
        a tonic event and 0 indicates a non-event. Returns None if baseline
        is not available for the first REM subsequence."""
        if self.nrem_start_time is not None:
            self.get_nrem_baseline(data['signals']['Chin'][0:self.nrem_end_time - self.nrem_start_time])
            out = self.findT_over_threshold(data = data['signals']['Chin'][self.nrem_end_time - self.nrem_start_time:].ravel())
        else:
            self.get_nrem_baseline(None)
            out = self.findT_over_threshold(data = data['signals']['Chin'].ravel())
        return out

if __name__ == "__main__":
    scorer = Autoscorer()
    results = scorer.score_REM()

