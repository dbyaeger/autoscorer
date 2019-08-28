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
    for channel in self.channels:
        p_event_idx[channel] = []
    return p_event_idx

def convert_to_rem_idx(time: float, rem_start_time: int, rem_end_time: int,
                       f_s: int) -> int:
    """Converts a time during REM to an index. Returns the index relative
    to the start of the signal only containing REM.
    """

    assert rem_start_time <= time <= rem_end_time, f"Time {time} is not between REM start and end times!"

    time -= rem_start_time

    seconds_idx = (time // 1)*f_s

    frac_idx = np.floor((time % 1)*f_s)

    return int(seconds_idx +  frac_idx)

def tuple_builder(groups: list, event_type: str, f_s: int, rem_start_time: int, rem_end_time: int) -> list:
    """Builds tuples of (event_start, event_end, event_type) given a list
    of arrays of indices.  Returns a list of tuples.

    INPUT: groups, a list of arrays of indices in consecutive order, and
    the type of event ('RSWA_T' or 'RSWA_P')

    OUTPUT: list of arrays of event_start, event_end, event_type)
    """
    assert event_type in ['RSWA_T', 'RSWA_P'], f"Event type must be RSWA_T or RSWA_P, not {event_type}!"

    long_tuples = [(g[0],g[-1],event_type) for g in groups if len(g) > 1]
    short_tuples = [(g[0],g[-1],event_type) for g in groups if len(g) == 1]
    return convert_to_study_time(tuples = long_tuples + short_tuples,
                                 f_s = f_s,
                                 rem_start_time = rem_start_time,
                                 rem_end_time = rem_end_time)

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

def sequence_builder(groups: list, length: int) -> np.ndarray:
    """Builds sequences of 0 and 1's given a list of indices in consecutive
    order. Returns a numpy array.

    INPUT: groups, a list of arrays of indices in consecutive order and
    the length of the array to be returned.

    OUTPUT: 1-D numpy array where index entries are ones and all other values
    are zero.
    """
    out = np.zeros(length)
    if len(groups) > 0:
        idx = np.concatenate(groups)
        out[idx] = 1
    return out
