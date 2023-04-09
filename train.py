#to train a model to predict message, sentence and token level labels
#Note: run convert_data_piranha_to_kaggle_format.py before this train.py
#dedication/reference:https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb

import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

import convert_data_piranha_to_kaggle_format
import os
from torch import cuda
from configs import *
import utils
from utils import *
import spacy

global_f1_validation=0
global_validation_loss=999999
precision_global=0
device = 'cuda' if cuda.is_available() else 'cpu'

if (DISABLE_WANDB):
    os.environ['WANDB_DISABLED'] = DISABLE_WANDB
else:
    wandb.init(project="training_3model_piranha")
print(f"found that the type of run is: {TYPE_OF_RUN}")
def train(epoch,NO_OF_CLASSES,model):
    model.train()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)
        outputs = model(ids, mask, token_type_ids)
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss




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


class ModelWithNN(torch.nn.Module):
    def __init__(self,NO_OF_CLASSES,base_model):
        super(ModelWithNN, self).__init__()
        self.l1 = base_model
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(LAST_LAYER_INPUT_SIZE,NO_OF_CLASSES)

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1['pooler_output'])
        output = self.l3(output_2)
        return output


class RobertaWithFFNN(torch.nn.Module):
    def __init__(self,NO_OF_CLASSES,base_model):
        super(RobertaWithFFNN, self).__init__()
        self.l1 = base_model
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(LAST_LAYER_INPUT_SIZE,NO_OF_CLASSES)

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1['pooler_output'])
        output = self.l3(output_2)
        return output



def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)




def validation(epoch,model):
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
            validation_loss = loss_fn(outputs, targets)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets,validation_loss


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
                string_truple_labels.append(dict_all_index_labels[index])
            else:
                string_truple_labels.append(0)
        all_labels_string_value.extend(string_truple_labels)
    return all_labels_string_value


def print_return_per_label_metrics(gold_labels_boolean_tuples, pred_labels_boolean_tuples):

    # to calculate per label accuracy- increase counter for each true positive
    assert len(gold_labels_boolean_tuples) == len(pred_labels_boolean_tuples)
    label_counter_accuracy = {}
    label_counter_overall = {}

    avg_f1=0
    avg_precision=0
    avg_recall=0
    sum_f1=0
    sum_accuracy=0
    sum_precision=0
    sum_recall=0
    # have a dictionary inside a dictionary to keep track of TP,FN etc for each label
    # e.g.,{"words_location_TP:24}
    true_positive_true_negative_etc_per_label = {}

    #initializing the dictionaries with zeores
    for x in range(len(gold_labels_boolean_tuples[0])):
        label_string = dict_all_index_labels[x]
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
            label_string=dict_all_index_labels[index]
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
        accuracy=v / total

        tp=true_positive_true_negative_etc_per_label[label+"_TP"]
        tn = true_positive_true_negative_etc_per_label[label + "_TN"]
        fp = true_positive_true_negative_etc_per_label[label + "_FP"]
        fn=true_positive_true_negative_etc_per_label[label+"_FN"]

        print(f"accuracy ={accuracy}")
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

        precision_label_name="precision"+"_"+label
        recall_label_name = "recall" + "_" + label
        f1_label_name = "f1" + "_" + label
        accuracy_label_name = "accuracy" + "_" + label
        wandb.log({precision_label_name: precision,'epoch': epoch})
        wandb.log({recall_label_name: recall, 'epoch': epoch})
        wandb.log({f1_label_name: F1, 'epoch': epoch})
        wandb.log({accuracy_label_name: accuracy, 'epoch': epoch})

        sum_accuracy = sum_accuracy+accuracy
        sum_f1=sum_f1+F1
        sum_precision = sum_precision + precision
        sum_recall = sum_recall + recall

    avg_f1=sum_f1/len(label_counter_accuracy.items())
    avg_accuracy=sum_accuracy/len(label_counter_accuracy.items())
    wandb.log({'average_precision': sum_precision/len(label_counter_accuracy.items()), 'epoch': epoch})
    wandb.log({'average_recall': sum_recall/len(label_counter_accuracy.items()), 'epoch': epoch})
    wandb.log({'average_accuracy': avg_accuracy, 'epoch': epoch})

    return avg_f1



def given_dataframe_return_loader(df):
    new_df = df[['text', 'list']].copy()
    testing_dataset = new_df.sample()
    testing_set = CustomDataset(testing_dataset, tokenizer, MAX_LEN)
    return DataLoader(testing_set, **test_params)

def get_per_label_positive_negative_examples(df, no_of_classes):
    per_label_positive_examples={}
    per_label_negative_examples = {}
    labels=df.columns[2:2+no_of_classes].tolist()
    for label in labels:
        for datapoint in df[label]:
            if datapoint==1:
                increase_counter(label,per_label_positive_examples)
            else:
                increase_counter(label, per_label_negative_examples)
    print(f"per_label_positive_examples={per_label_positive_examples}")
    print(f"per_label_negative_examples={per_label_negative_examples}")
    return per_label_positive_examples, per_label_negative_examples


if TYPE_OF_RUN=="train":
    no_of_classes,dict_all_labels_index, dict_all_index_labels,labels_in_this_training=convert_data_piranha_to_kaggle_format.create_training_data()
    df = pd.read_csv(convert_data_piranha_to_kaggle_format.OUTPUT_FILE_NAME, sep=",", on_bad_lines='skip')
    df['list'] = df[df.columns[2:]].values.tolist()
    columns_combined= ['text', 'list']
    for m in labels_in_this_training.keys():
        columns_combined.append(m)
    new_df= df[columns_combined].copy()

    train_size = 0.8
    dev_size = 0.5
    train_dataset = new_df.sample(frac=train_size, random_state=200)
    print("------during removal of threshold")
    print("for train")
    per_label_positive_examples, per_label_negative_examples = get_per_label_positive_negative_examples(train_dataset, no_of_classes)
    validation_dev_dataset = new_df.drop(train_dataset.index).reset_index(drop=True)
    validation_dataset = validation_dev_dataset.sample(frac=dev_size, random_state=200).reset_index(drop=True)
    test_dataset = validation_dev_dataset.drop(validation_dataset.index).reset_index(drop=True)
    print(f"total number of train datapoints={len(train_dataset)}")
    print("for validation")
    per_label_positive_examples, per_label_negative_examples = get_per_label_positive_negative_examples(validation_dataset, no_of_classes)
    print(f"total number of validation_dataset datapoints={len(validation_dataset)}")
    print("for test")
    per_label_positive_examples, per_label_negative_examples = get_per_label_positive_negative_examples(
        test_dataset, no_of_classes)
    print(f"total number of test_dataset datapoints={len(test_dataset)}")

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
    patience_counter=0
    overall_accuracy=0
    accuracy_validation=0
    for epoch in range(EPOCHS):
        wandb.log({'patience_counter': patience_counter, 'epoch': epoch})
        if(patience_counter>PATIENCE):
            print(f"found that validation loss is not improving after hitting patience of {PATIENCE}. Quitting")
            sys.exit()


        model = ModelWithNN(no_of_classes,MODEL)
        model.to(device)
        train_loss=train(epoch, no_of_classes, model)
        predictions_validation, gold_validation ,validation_loss = validation(epoch,model)


        if validation_loss<global_validation_loss:
            global_validation_loss=validation_loss
        else:
            patience_counter+=1



        wandb.log({'train_loss': train_loss,'epoch': epoch})
        wandb.log({'validation_loss': validation_loss,'epoch': epoch})
        predictions_validation = np.array(predictions_validation) >= 0.5
        accuracy_validation_scikit_version = metrics.accuracy_score(gold_validation, predictions_validation)
        overall_accuracy=overall_accuracy+accuracy_validation
        avg_accuracy_scikit_version=overall_accuracy/(epoch+1)
        outputs_float = predictions_validation.astype(float)


        avg_f1_validation_this_epoch = print_return_per_label_metrics(gold_validation, outputs_float)
        #rewrite the best model every time the f1 score improves
        if avg_f1_validation_this_epoch > global_f1_validation:
            global_f1_validation = avg_f1_validation_this_epoch
            torch.save(model.state_dict(), SAVED_MODEL_PATH)

        gold=get_label_string_given_index(gold_validation)
        predicted=get_label_string_given_index(outputs_float)

        print(f"avg F1:{avg_f1_validation_this_epoch}\n")
        wandb.log({'average_f1': avg_f1_validation_this_epoch})
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
        predictions_validation = np.array(predictions) >= 0.5
        outputs_float = predictions_validation.astype(float)
        print(f"predicted:{get_label_string_given_index(outputs_float)}")


        #split the incoming text based into sentences
        df[['text']] = "samnple"

