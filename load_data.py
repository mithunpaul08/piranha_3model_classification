import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


# import csv
# with open('/Users/mitch/research/piranha/piranha_3model_classification/data/train.csv') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',')
#     for row in spamreader:
#         print(', '.join(row))

df = pd.read_csv("/Users/mitch/research/piranha/piranha_3model_classification/data/train.csv",sep="," ,on_bad_lines='skip')
df['list'] = df[df.columns[2:]].values.tolist()
new_df = df[['text', 'list']].copy()
print(new_df.head())