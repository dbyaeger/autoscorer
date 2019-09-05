# autoscorer
autoscorer is a module for automated scoring of EMG from sleep studies according to AASM guidelines.

## Getting Started

Open up a terminal window and type:

`git clone https://github.com/dbyaeger/autoscorer.git`

## Usage

Change directories into autoscorer with

`cd autoscorer`

Then start a python session with

`python`

To analyze a single patient ID using the default parameters:

```python
>>> from autoscorer.autoscorer import Autoscorer
>>> scorer = Autoscorer(ID = 'XVZ2FFAEC864IPK', data_path = './processed_data/processed')
>>> results = scorer.score_REM()
Processing ID: XVZ2FFAEC864IPK REM subsequence: 0
        Finished analyzing tonic events...
        Finished analyzing phasic events...
Processing ID: XVZ2FFAEC864IPK REM subsequence: 1
        Finished analyzing tonic events...
        Finished analyzing phasic events...
Processing ID: XVZ2FFAEC864IPK REM subsequence: 2
        Finished analyzing tonic events...
        Finished analyzing phasic events...
Processing ID: XVZ2FFAEC864IPK REM subsequence: 3
        Finished analyzing tonic events...
        Finished analyzing phasic events...
>>> human_annotations = scorer.get_annotations()
```

To analyze all of the patient files in a directory:

```python
>>> from autoscorer.Score_All
>>> scoring_results, human_annotations = scoreAll(data_path = './processed_data/processed')

Processing ID: XVZ2FYAQH80F05L REM subsequence: 0
        Finished analyzing tonic events...
        Finished analyzing phasic events...
Processing ID: XVZ2FYAQH80F05L REM subsequence: 1
        Finished analyzing tonic events...
        Finished analyzing phasic events...
Processing ID: XVZ2FYAQH80F05L REM subsequence: 2
        Finished analyzing tonic events...
        Finished analyzing phasic events...
Processing ID: XVZ2FYAQH80F05L REM subsequence: 3
        Finished analyzing tonic events...
        Finished analyzing phasic events...
...
```

### Input

Input data files are expected to named in the format

`patientID_i.p`

where each i corresponds to the index of a REM subsequence, defined as 1 or more consecutive epochs of REM sleep

The input data files are pickled Python dictionaries with the following key-value pairs:

```python
"ID":ID (str),
"study_start_time": study start time in datatime format
"staging": [(start_time_event_i, end_time_event_i, sevent_type_i), ...] (float, float, str)
"apnia_hypopnia_events":[(start_time_event_i, end_time_event_i, event_type_i), ] (float, float, str)
"rswa_events":[(start_time_event_i, end_time_event_i, event_type_i), ...] (float, float, str)
"signals":{"channel_i":array(n_seconds, sampling_rate)}
```

* `staging` contains the end and start times of sleep stages, which are defined as `R` or REM and 
`N` for non-REM. 
* `apnia_hypopnia_events` contains apnea events, denoted by `A` and hypoapnea events, denoted by `H`
* `rswa_events` contains phasic RSWA events, denoted by `P`, and tonic RSWA events, indicated by `T`
* `signals` can contain any collection of signals but must include `Chin`, `L Leg`, and `R Leg` channels

### Output

Dictionary of scored REM subsequences in the format:
```python
        {'RSWA_P': {'REM_0': {scores}, .., 'REM_n': {scores}},
        'RSWA_T': {'REM_0': {scores}, .., 'REM_n': {scores}}}
```

Scores can either be outputted as tuples in the format:
```python
(event_start, event_end, event_type)
```
* `event_start` and `event_end` are in units of seconds relative to the start of the sleep study.  
* `event_type` is `RSWA_T` for tonic events and `RSWA_P` for phasic events.

Or as sequences:

`000000000111111100000000000011100000000000...`

* Each `0` or `1` corresponds to one sample of the signal. A `1` indicates that an RSWA event was detected.

### Parameters

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
            exceed the mean (or the number of standard deviations by which
            the phasic amplitudes must exceed the mean, depending on p_mode) in
            order to be considered a phasic signal-level event.

        p_quantile: the quantile of the REM baseline signal that must be
            exceeded in order for a phasic signal to be considered a phasic
            event when p_mode is set to 'quantile.'

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

        verbose: If set to True, ID and REM subsequence number will be printed
            out during scoring.
