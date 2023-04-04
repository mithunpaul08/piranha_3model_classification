# piranha_3model_classification

### install pytorch from their website

`pip install -r requirements.txt`

`python train.py`

## steps

- download annotated dataset from osf [repo]( https://osf.io/3rcu4/?view_only=2800ce01e12645a48bbeae53cc2fb201) 
- unzip piranha.zip
- move the type of data you want to train on to the folder `data/`
  - e.g., data/signature.tsv
- `cp combined_all_annotation_jan9th2023.jsonl query_file.jsonl`
- `python train.py`

 
best way to view the progress of training is in the corresponding the wandb graph [project](https://wandb.ai/nazgul588/training_3model_piranha?workspace=user-nazgul588)

