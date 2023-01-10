# piranha_3model_classification

### install pytorch from their website

`pip install -r requirements.txt`

`python train.py`

## steps

- copy your training data to data/query_file.jsonl. Should be in the format produced by prodigy tool used for annotation
- `cp combined_all_annotation_jan9th2023.jsonl query_file.jsonl`
- `python train.py`

 
best way to view the progress of training is to use the wandb graph link pasted at the beginnning of log file

