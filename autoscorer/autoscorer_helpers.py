#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:12:42 2019

@author: danielyaeger
"""
import numpy as np
from pathlib import Path
import pickle

def get_data(filename: str, data_path: Path, channels: list, f_s: int,
             fields: list) -> dict:
    """Opens a .p file with the given name in the given directory and returns
    the dictionary that is pickled within the file. Also checks that dictionary
    contains the correct keys, that the input sampling rate is correct.
    """

    assert data_path.joinpath(filename).exists(), f"The file {filename} was not found in {data_path}"

    with data_path.joinpath(filename).open('rb') as f_in:
        data = pickle.load(f_in)

    check_fields(data = data, filename = filename, fields = fields)
    check_signal(data = data, channels = channels, f_s = f_s)
    check_times(data = data)

    return data

def check_signal(data: dict, channels: list, f_s: int) -> None:
    """ Checks if the value of the sampling rate argument provided when
    instantiating the rule_based_scorer is correct."""

    for channel in channels:
        assert data['signals'][channel].shape[1] == f_s, f"Sampling rate argument set to {f_s}, but actual sampling rate is {data['signals'][channel].shape[1]}"

def check_times(data: dict) -> None:
    """ Checks if the length of the data in signals corresponds to the
    length of data in staging"""

    total_time_staging = sum([t[1] - t[0] for t in data['staging']])
    signal_time = data['signals']['Chin'].shape[0]
    assert total_time_staging == signal_time, f"Staging length {total_time_staging} not equal to length of signal time {signal_time}.\n Staging = {data['staging']}!"


def check_fields(data: dict, filename: str, fields: list) -> None:
    """ Checks if all of the keys on a list are in a dictionary. Takes
    the dictionary as input and raises an assertion error if a key in
    the dictionary is not found"""
    for key in fields:
        assert key in data.keys(), f"{key} not found in {filename}!"

def make_event_idx(data: dict, channels: list) -> dict:
    """Makes the p_event_idx, which stores phasic events that exceed the
    continuity threshold as these events occur during processing."""
    p_event_idx = {}
    for channel in channels:
        p_event_idx[channel] = []
    return p_event_idx

def convert_to_rem_idx(time: float, rem_start_time: int, rem_end_time: int,
                       f_s: int) -> int:
    """Converts a time during REM to an index. Returns the index relative
    to the start of the signal only containing REM.
    """

    assert rem_start_time <= time <= rem_end_time, f"Time {time} is not between REM start at {rem_start_time} and REM end at {rem_end_time}!"

    time -= rem_start_time

    seconds_idx = (time // 1)*f_s

    frac_idx = np.floor((time % 1)*f_s)

    return min(int(seconds_idx +  frac_idx), (rem_end_time - rem_start_time)*f_s - 1)

def tuple_builder(groups: list, event_type: str, f_s: int, rem_start_time: int, rem_end_time: int,
                  phasic_start_time_only: bool = False) -> list:
    """Builds tuples of (event_start, event_end, event_type) given a list
    of arrays of indices.  Returns a list of tuples.

    INPUT: groups, a list of arrays of indices in consecutive order, and
    the type of event ('RSWA_T' or 'RSWA_P'), and phasic_start_time_only,
    a boolean which, if True, causes RSWA_P events to have an end time
    equal to their start time
    
    OUTPUT: list of arrays of event_start, event_end, event_type)
    """
    assert event_type in ['RSWA_T', 'RSWA_P'], f"Event type must be RSWA_T or RSWA_P, not {event_type}!"

    long_tuples = [(g[0],g[-1],event_type) for g in groups if len(g) > 1]
    short_tuples = [(g[0],g[-1],event_type) for g in groups if len(g) == 1]
    out = convert_to_study_time(tuples = long_tuples + short_tuples,
                                 f_s = f_s,
                                 rem_start_time = rem_start_time,
                                 rem_end_time = rem_end_time)
    
    if event_type == 'RSWA_P' and phasic_start_time_only:
        out = [(t[0], t[0], t[-1]) for t in out]
        
    return out

def convert_to_study_time(tuples: list, f_s: int, rem_start_time: int, rem_end_time: int) -> list:
    """ Takes in the start time of the REM subsequence and tuples in the
    format (start_time, stop_time, event_type), where start_time and
    stop_time are in units of samples since the beginning of the study
    and converts the start_time and stop_time in each tuple to units of
    study time"""

    times = [(round(t[0]/f_s + rem_start_time,2),
             round(t[1]/f_s + rem_start_time,2),
              t[2]) for t in tuples]

    # Make sure times do not exceed rem_end_time
    for time in times:
        assert time[1] <= rem_end_time, "End time of event greater than REM end time!"

    return times

def sequence_builder(groups: list, length: int, 
                     phasic_start_time_only: bool = False) -> np.ndarray:
    """Builds sequences of 0 and 1's given a list of indices in consecutive
    order. Returns a numpy array.

    INPUT: groups, a list of arrays of indices in consecutive order,
    the length of the array to be returned, and phasic_start_time_only, a boolean,
    which, if set to True, causes only the first index in a group of consecutive
    indices to be set to one. This parameter should only be used with phasic
    events.

    OUTPUT: 1-D numpy array where index entries are ones and all other values
    are zero.
    """
    if len(groups) > 0:
        assert (length - 1) >= np.max([np.max(l) for l in groups]), f"Length must be at least as big as maximum index! Length: {length} and max idx: {np.max([np.max(l) for l in groups])}"
    
    out = np.zeros(length)
    if len(groups) > 0:
        if not phasic_start_time_only:
            idx = np.concatenate(groups)
        elif phasic_start_time_only:
            idx = [l[0] for l in groups]
        out[idx] = 1
    return out

def round_time(times: tuple, f_s: int, phasic_start_time_only: bool = True) -> tuple:
    """ Rounds times in a tuple of the format
    
    (start time, end time, type)
    
    to nearest time increment based on f_s (sampling rate in Hz). For example,
    if start time is 3.34 and f_s is 10, then 3.3 will be returned.
    
    INPUT: tuple with first two values as floats, and f_s, sampling rate as an 
    integer
    
    PARAMETER: phasic_start_time_only: if set to True, then the end time in
    the tuple will be set equal to the start time.
    
    OUTPUT: tuple with times rounded down to nearest increment of time according
    to f_s"""
    times = list(times)
    for i in range(2):
        times[i] = np.round(times[i], int(np.log10(f_s)))
    if phasic_start_time_only: times[1] = times[0]
    return tuple(times)

def adjust_rswa_event_times(time_list: list, rem_start_time: int,
                            rem_end_time: int) -> list:
    """ Adjusts the start time of RSWA events to be no greater than the
    REM start time and the end time of RSWA events to be no greater than the
    REM end time. This is required because some annotated events end after
    the REM end time.
        
        new start time = max(event start time, REM start time)
        new end time = min(event end time, REM end time)
    
    INPUT: list of tuples in the format 
        (event start time, event end time, event type)
    
    OUTPUT: list of tuples
    """
    return [(max(t[0], rem_start_time), min(t[1], rem_end_time), t[2]) for t in time_list]

def collapse_p_and_t_events(t_events: list or np.ndarray, p_events: list or np.ndarray,
                            tuples: bool = False, f_s: int = 10) -> list or np.ndarray:
    """ Collapses phasic and tonic events into a single track. When events overlap,
    tonic events are given precedence over phasic events.
    
    INPUT:  t_events: list of tuples or array of tonic events, 
            p_events: list of tuples or array of tonic events,
            tuples, a boolean that indicates whether events are in tuple or
            sequence form
    
    OUTPUT: list of numpy array with tonic and phasic events collapsed into a
            single trac. Tonic events are given priority over phasic events.
            Also returns the number of events where there was overlap of tonic
            and phasic events.
    
    NOTE: Overlaps are counted differently depending on whether tuples option is
    set to True (and input data type are lists of tuples). In this case, overlaps
    are counted whenever a T event overlaps with a P event, on an event-by-event
    basis. If tuples option is set to False (and input data type are numpy arrays),
    overlaps are counted on a sample-by-sample basis.
    """
    if tuples and (len(t_events) > 0) and (len(p_events) > 0):
        assert type(t_events[0]) == type(p_events[0]) == tuple, f"Tuples option set to true but data in t_events is of type {type(t_events[0])} and data in p_events is of type {type(p_events[0])}!"
    
    
    if not tuples:
        assert type(t_events) == type(p_events) == np.ndarray, f"If tuples options set to False, both t_events and p_events must be numpy arrays, not {type(t_events)} and {type(p_events)}!"
    
    if tuples:
        overlap_counter = 0
        single_track = t_events[:]
        for phasic_event in p_events:
            add_event = True
            start, end = phasic_event[0], phasic_event[1]
            for tonic_event in t_events:
                if tonic_event[0] <= start <= tonic_event[1]:
                    overlap_counter += 1
                    if tonic_event[0] <= end <= tonic_event[1]:
                        add_event = False
                    elif end > tonic_event[1]:
                        start = tonic_event[1] + 1/f_s
                elif start < tonic_event[0]:
                    if tonic_event[0] <= end <= tonic_event[1]:
                        overlap_counter += 1
                        end = tonic_event[0] - 1/f_s
            if add_event:
                single_track.append((start, end, phasic_event[2]))
    else:
        t_events[np.nonzero(t_events)] = 2
        single_track = np.maximum(t_events, p_events)
        overlap_counter = sum(p_events == 1) - sum(single_track == 1)
    return single_track, overlap_counter
        
        
        
        
    
    
