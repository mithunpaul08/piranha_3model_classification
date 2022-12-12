#to train a model to predict message, sentence and token level labels
#Note: run convert_data_piranha_to_kaggle_format.py before this train.py

import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
import convert_data_piranha_to_kaggle_format

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
NO_OF_CLASSES=len(convert_data_piranha_to_kaggle_format.labels_in_this_training)
MAX_LEN = 500
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
TESTING_BATCH_SIZE=1
EPOCHS = 50
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
SAVED_MODEL_PATH="./output/best_model.pt"
TESTING_FILE_PATH="./data/testing_data.csv"

#is it training or testing. testing means will load a saved model and test =["train","test"]all_data.csv
TYPE_OF_RUN="test"


def train(epoch):
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _ % 5000 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, NO_OF_CLASSES)

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1['pooler_output'])
        output = self.l3(output_2)
        return output
model = BERTClass()
model.to(device)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)



def validation(epoch):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(validation_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


def testing():
    model.eval()
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs

def get_label_string_given_index(labels_boolvalue):
    all_labels_string_value=[]
    for each_truple in labels_boolvalue:
        string_truple_labels = []
        for index, bool_value in enumerate(each_truple):
            if bool_value==1:
                string_truple_labels.append(convert_data_piranha_to_kaggle_format.dict_all_index_labels[index])
            else:
                string_truple_labels.append(0)
        all_labels_string_value.append(string_truple_labels)
    return all_labels_string_value


if TYPE_OF_RUN=="train":
    convert_data_piranha_to_kaggle_format.create_training_data()
    df = pd.read_csv(convert_data_piranha_to_kaggle_format.OUTPUT_FILE_NAME, sep=",", on_bad_lines='skip')
    df['list'] = df[df.columns[2:]].values.tolist()
    new_df = df[['text', 'list']].copy()
    train_size = 0.8
    train_dataset = new_df.sample(frac=train_size, random_state=200)
    validation_dataset = new_df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    # print("FULL Dataset: {}".format(new_df.shape))
    # print("TRAIN Dataset: {}".format(train_dataset.shape))
    # print("VALIDATION Dataset: {}".format(validation_dataset.shape))

    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
    validation_set = CustomDataset(validation_dataset, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    validation_params = {'batch_size': VALID_BATCH_SIZE,
                         'shuffle': True,
                         'num_workers': 0
                         }

    training_loader = DataLoader(training_set, **train_params)
    validation_loader = DataLoader(validation_set, **validation_params)

    print(f"************found that the device is {device}\n")
    for epoch in range(EPOCHS):
        train(epoch)
        outputs, targets = validation(epoch)
        outputs = np.array(outputs) >= 0.5
        outputs_float = outputs.astype(float)
        torch.save(model.state_dict(), SAVED_MODEL_PATH)
        print(f"precision={metrics.precision_score(targets,outputs_float,average='micro')}")
        print(f"recall={metrics.recall_score(targets, outputs_float,average='micro')}")
        print(f"Gold labels:{get_label_string_given_index(targets)}")
        print(f"predicted:{get_label_string_given_index(outputs_float)}")
        accuracy = metrics.accuracy_score(targets, outputs_float)
        f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
        f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
        print(f"Validation at epoch : {epoch}")
        print(f"F1 Score (Micro) = {f1_score_micro}")
        print(f"F1 Score (Macro) = {f1_score_macro}")
        print(f"end of epoch {epoch}")
        print(f"---------------------------")
else:
    if TYPE_OF_RUN=="test":
        df = pd.read_csv(TESTING_FILE_PATH, sep=",", on_bad_lines='skip')
        df['list'] = df[df.columns[2:]].values.tolist()
        new_df = df[['text', 'list']].copy()
        train_size = 1
        testing_dataset = new_df.sample(frac=train_size,)

        # print("FULL Dataset: {}".format(new_df.shape))
        # print("TRAIN Dataset: {}".format(train_dataset.shape))
        # print("VALIDATION Dataset: {}".format(validation_dataset.shape))

        testing_set = CustomDataset(testing_dataset, tokenizer, MAX_LEN)


        test_params = {'batch_size': TESTING_BATCH_SIZE,
                        'shuffle': False,
                        'num_workers': 0
                        }


        testing_loader = DataLoader(testing_set, **test_params)

        model.load_state_dict(torch.load(SAVED_MODEL_PATH))
        model.eval()
        predictions=testing()
        print(predictions)

