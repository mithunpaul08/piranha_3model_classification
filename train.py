#to train a model to predict message, sentence and token level labels
#Note: run convert_data_piranha_to_kaggle_format.py before this train.py

import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

import convert_data_piranha_to_kaggle_format
import os
from torch import cuda
from configs import *


f1_score_global=0
precision_global=0
device = 'cuda' if cuda.is_available() else 'cpu'
NO_OF_CLASSES=len(convert_data_piranha_to_kaggle_format.labels_in_this_training)

print(f"found that the type of run is: {TYPE_OF_RUN}")
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


def testing(loader):
    model.eval()
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
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
        all_labels_string_value.extend(string_truple_labels)
    return all_labels_string_value


def print_per_label_metrics(gold_labels_boolean_tuples, pred_labels_boolean_tuples):


    # to calculate per label accuracy- increase counter for each true positive
    assert len(gold_labels_boolean_tuples) == len(pred_labels_boolean_tuples)
    label_counter_accuracy = {}
    label_counter_overall = {}

    # have a dictionary inside a dictionary to keep track of TP,FN etc for each label
    # e.g.,{"words_location_TP:24}
    true_positive_true_negative_etc_per_label = {}

    #initializing the dictionaries with zeores
    for x in range(len(gold_labels_boolean_tuples[0])):
        label_string = convert_data_piranha_to_kaggle_format.dict_all_index_labels[x]
        label_counter_accuracy[label_string] = 0

        label_tp = label_string + "_TP"
        true_positive_true_negative_etc_per_label[label_tp] = 0
        label_tn = label_string + "_TN"
        true_positive_true_negative_etc_per_label[label_tn] = 0
        label_fp = label_string + "_FP"
        true_positive_true_negative_etc_per_label[label_fp] = 0
        label_fn = label_string + "_FN"
        true_positive_true_negative_etc_per_label[label_fn] = 0




    all_labels_string_value = []
    for gold_truple, pred_truple in zip(gold_labels_boolean_tuples, pred_labels_boolean_tuples):

        assert len(gold_truple)==len(pred_truple)



        for index,value in enumerate(gold_truple):

            #to calculate overall count of labels... should be same as len(gold)
            label_string=convert_data_piranha_to_kaggle_format.dict_all_index_labels[index]


            if label_string in label_counter_overall:
                current_count = label_counter_overall[label_string]
                label_counter_overall[label_string] = current_count + 1
            else:
                label_counter_overall[label_string] = 1




            if gold_truple[index] == pred_truple[index]:

                # calculate accuracy as long as both gold and pred match- irrespective of TP, FP etc
                if label_string in label_counter_accuracy:
                    current_count = label_counter_accuracy[label_string]
                    label_counter_accuracy[label_string] = current_count + 1
                else:
                    label_counter_accuracy[label_string] = 1

                #finding true positive
                if int(gold_truple[index]) ==1:
                    current_label_tp=label_string+"_TP"
                    if current_label_tp in true_positive_true_negative_etc_per_label:
                        old_value=true_positive_true_negative_etc_per_label[current_label_tp]
                        true_positive_true_negative_etc_per_label[current_label_tp]=old_value+1
                    else:
                        true_positive_true_negative_etc_per_label[current_label_tp]=1

                #true negative
                if int(gold_truple[index]) ==0:
                    current_label_tn = label_string + "_TN"
                    if current_label_tn in true_positive_true_negative_etc_per_label:
                        old_value=true_positive_true_negative_etc_per_label[current_label_tn]
                        true_positive_true_negative_etc_per_label[current_label_tn]=old_value+1
                    else:
                        true_positive_true_negative_etc_per_label[current_label_tn]=1

            #false negative
            else:
                if int(gold_truple[index]) == 1 and int(pred_truple[index])==0:
                    current_label_fn = label_string + "_FN"
                    if current_label_fn in true_positive_true_negative_etc_per_label:
                        old_value=true_positive_true_negative_etc_per_label[current_label_fn]
                        true_positive_true_negative_etc_per_label[current_label_fn]=old_value+1
                    else:
                        true_positive_true_negative_etc_per_label[current_label_fn]=1
                else:
                    if int(gold_truple[index]) == 0 and int(pred_truple[index]) == 1:
                        current_label_fp = label_string + "_FP"
                        if current_label_fp in true_positive_true_negative_etc_per_label:
                            old_value = true_positive_true_negative_etc_per_label[current_label_fp]
                            true_positive_true_negative_etc_per_label[current_label_fp] = old_value + 1
                        else:
                            true_positive_true_negative_etc_per_label[label_fp] = 1








    for label, v in label_counter_accuracy.items():
        total = label_counter_overall[label]

        print(f"------\nFor the  label {label}:")
        print(f"accuracy {label}={v / total}")
        tp=true_positive_true_negative_etc_per_label[label+"_TP"]
        tn = true_positive_true_negative_etc_per_label[label + "_TN"]
        fp = true_positive_true_negative_etc_per_label[label + "_FP"]
        fn=true_positive_true_negative_etc_per_label[label+"_FN"]

        print(f"true positive:{tp}")
        print(f"true negative:{tn}")
        print(f"false positive:{fp}")
        print(f"false negative :{fn}")

        if (tp+fp)==0:
            precision=0
        else:
            precision =tp / (tp + fp)

        if (tp + fn) == 0:
            recall=0
        else:
            recall =tp / (tp + fn)
        if (precision+recall)==0:
            F1=0
        else:
            F1 = 2 * (precision * recall) / (precision + recall)

        print(f"precision={precision}")
        print(f"recall={recall}")
        print(f"F1={F1}")



def given_dataframe_return_loader(df):
    new_df = df[['text', 'list']].copy()
    testing_dataset = new_df.sample()
    testing_set = CustomDataset(testing_dataset, tokenizer, MAX_LEN)
    return DataLoader(testing_set, **test_params)

if TYPE_OF_RUN=="train":
    convert_data_piranha_to_kaggle_format.create_training_data()
    df = pd.read_csv(convert_data_piranha_to_kaggle_format.OUTPUT_FILE_NAME, sep=",", on_bad_lines='skip')
    df['list'] = df[df.columns[2:]].values.tolist()
    new_df = df[['text', 'list']].copy()
    train_size = 0.8
    train_dataset = new_df.sample(frac=train_size, random_state=200)
    validation_dataset = new_df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

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

        precision=metrics.precision_score(targets,outputs_float,average='micro')

        #rewrite the best model every time the f1 score improves
        if precision> precision_global:
            precision_global=precision
            torch.save(model.state_dict(), SAVED_MODEL_PATH)



        # print(f"precision_micro={metrics.precision_score(targets,outputs_float,average='micro')}")
        # print(f"recall={metrics.recall_score(targets, outputs_float,average='micro')}")

        print_per_label_metrics(targets, outputs_float)
        gold=get_label_string_given_index(targets)
        predicted=get_label_string_given_index(outputs_float)




        print(f"Gold labels:{get_label_string_given_index(targets)}\n\n")
        print(f"predicted:{get_label_string_given_index(outputs_float)}")
        accuracy = metrics.accuracy_score(targets, outputs_float)
        f1_score_micro = metrics.f1_score(targets, outputs, average='micro')


        if f1_score_micro>f1_score_global:
            f1_score_global=f1_score_micro



        f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
        # print(f"Validation at epoch : {epoch}")
        # print(f"F1 Score (Micro) = {f1_score_micro}")

        print(f"end of epoch {epoch}")
        print(f"---------------------------")
else:
    if TYPE_OF_RUN=="test":
        df = pd.read_csv(TESTING_FILE_PATH, sep=",", on_bad_lines='skip')
        df['list'] = df[df.columns[2:]].values.tolist()

        #SPLIT the incoming email into 4 categories. the full email will directly go as is to the BEST_MODEL_MESSAGE_LEVEL

        testing_loader=given_dataframe_return_loader(df)


        best_model_path=os.path.join(OUTPUT_DIRECTORY,BEST_MODEL_MESSAGE_LEVEL)
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        predictions=testing(testing_loader)
        outputs = np.array(predictions) >= 0.5
        outputs_float = outputs.astype(float)
        print(f"predicted:{get_label_string_given_index(outputs_float)}")


        #split the incoming text based into sentences
        df[['text']] = "samnple"

