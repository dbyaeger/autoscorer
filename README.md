# autoscorer
autoscorer is a module for automated scoring of EMG from sleep studies according to AASM guidelines.
___________________________________________________________________________________________________

## Getting Started

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
* event_start` and `event_end` are in units of seconds relative to the start of the sleep study.  
* `event_type` is `RSWA_T` for tonic events and `RSWA_P` for phasic events.
