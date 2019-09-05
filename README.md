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
>>> scorer = Autoscorer(ID = 'XVZ2FFAEC864IPK', data_path = '/Users/danielyaeger/Documents/processed_data/processed')
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
>>> scoring_results, human_annotations = scoreAll(data_path = '/Users/danielyaeger/Documents/processed_data/processed')

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



