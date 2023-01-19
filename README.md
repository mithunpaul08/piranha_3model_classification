# piranha_3model_classification

### install pytorch from their website

`pip install -r requirements.txt`

`python train.py`

## steps

- copy your training data to data/query_file.jsonl. Should be in the format produced by prodigy tool used for annotation
- `cp combined_all_annotation_jan9th2023.jsonl query_file.jsonl`
- `python train.py`

 
best way to view the progress of training is

to use the wandb graph link pasted at the beginnning of log file

#steps for internal isi run server


- connect to isi vpn
```
- ssh mithun@piranha-sub-01
- cd tuning_3model_classification/
- git pull
- cd
- cd all_mithun_slurm_scripts/ 
- sbatch tuning_3model_transformers_training_slurm_script_gpu.sh
- squeue | grep mithun
- ls -alrt
- tail -f classification_3model_message_level-73787.out 
```

