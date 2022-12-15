# high level: read the piranha annotated data and convert it into the kaggle toxic comment format as shown below. This will be
#format:# used for running the training:
#"id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"
#"00190820581d90ce"," YOUR FILTHY MOTHER IN THE ASS, DRY!",1,0,1,0,1,0
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
#this will be used for multilabel classification in train.py

#detailed level:
# go through each of the annotated data point, extract text and its label into a dictionary dict_spantext_to_labels
# then create a one hot vector based on how many labels a given text corresponds to/was marked with
#e.g. an entire email could be marked with multiple labels. remember it is a multilabel classifier
import json
import csv
import hashlib

from configs import *
dict_spantext_to_labels={}
dict_all_labels_index = {}
dict_all_index_labels = {}
labels_in_this_training=[]


#creating  different input data for each of messsage level, sentence level, signature, word


print(f"found that the type of label in this training run is: {TYPE_OF_LABEL}")
if TYPE_OF_LABEL=="all":
    for label in labels_all:
        labels_in_this_training.append(label)

else:
    for label in labels_all:
        if TYPE_OF_LABEL in label:
            labels_in_this_training.append(label)

assert len(labels_in_this_training)>1



for label in labels_in_this_training:
    header.append(label)


def create_label_index_mapping_both_directions():
    for index,label in enumerate(labels_in_this_training):
        dict_all_labels_index[label]=index
        dict_all_index_labels[index]=label



#given a label, and annotation span, retrieve a gold text (sentence or email ) which has annotations for that label
def given_label_retrieve_gold_text(in_file,label_to_check):
    Lines = in_file.readlines()
    for index, line in enumerate(Lines):
        annotations = json.loads(line)
        if "spans" in annotations:
            for entry in annotations["spans"]:
                label = entry["label"]

                #if the label in the span is the same as the one we are looking for- get the text corresponding to the start and end character indices
                if label == label_to_check:
                    full_text = get_spans_text_given_start_end_tokens(entry['start'], entry['end'], annotations)
                    return full_text

def get_negative_examples(dict_spantext_to_labels,plain_text_whole_email,empty_labels):
    if TYPE_OF_LABEL=="message":
        dict_spantext_to_labels[plain_text_whole_email]=empty_labels
    else:
        if TYPE_OF_LABEL=="sentence":
            seg = pysbd.Segmenter(language="en", clean=True)
            email_split_sentences = seg.segment(each_retrieved_email)
            for each_sent in email_split_sentences:
                dict_spantext_to_labels[plain_text_whole_email] = empty_labels
    return


#go through each of the spans, find each of the labels in the spans, and check if that label is one of the labels we are
#searching for. if yes, add it to a dictionary which maps text->label
def get_text_for_label_from_all_spans(Lines):
    for index, line in enumerate(Lines):
        annotations = json.loads(line)
        # for adding negative examples as text, 0,0,0
        empty_labels = [""] * len(labels_in_this_training)
        plain_text_whole_email=annotations['text'].replace("\n","")
        get_negative_examples(((dict_spantext_to_labels)),plain_text_whole_email,empty_labels)

        #existence of label spans means there was atleast one label in this email that was annotated
        if "spans" in annotations:
            for entry in annotations["spans"]:
                label = entry["label"]
                if label in labels_in_this_training:
                    if "message" in label:
                        #explicitly picking same text of the email because we want the empty entry in dict_spantext_to_labels to be replaced by
                        text=plain_text_whole_email
                        if text is not None :
                            if text in dict_spantext_to_labels:
                                old_value = dict_spantext_to_labels[text]
                                idx=dict_all_labels_index[label]
                                old_value[idx]=label
                                dict_spantext_to_labels[text] = old_value
                    else:
                        text = get_spans_text_given_start_end_tokens(entry['token_start'], entry['token_end'],
                                                                     annotations)
                        if text is not None:
                            text=text.replace("\n","")
                            if text in dict_spantext_to_labels:
                                old_value = dict_spantext_to_labels[text]
                                if label not in old_value:
                                    old_value.append(label)
                                    dict_spantext_to_labels[text] = old_value
                            else:
                                dict_spantext_to_labels[text] = [label]


#given the start and end of a span return the collection of the tokens corresponding to this in string format
def get_spans_text_given_start_end_tokens(token_start_of_span, token_end_of_span, annotations):
    starts_ends_tokens = []

    for index, token in enumerate(annotations['tokens']):
        if index>=token_start_of_span and index<=token_end_of_span:
            starts_ends_tokens.append(token['text'])


        # if (token['start']>=token_start_of_span and token['end']<=token_end_of_span):
        #     starts_ends_tokens.append(token['text'])
        # if (token['start'] >= token_start_of_span and token['end'] > token_end_of_span):
    assert len(starts_ends_tokens) >0
    return " ".join(starts_ends_tokens)

def create_training_data():
    with open(OUTPUT_FILE_NAME, 'w') as out:
        out.write(",".join(header))
        out.write("\n")


    with open("data/query_file.jsonl", 'r') as in_file:
        Lines = in_file.readlines()
        create_label_index_mapping_both_directions()
        # go through each of the annotated data point, extract text and its label into a dictionary dict_spantext_to_labels
        get_text_for_label_from_all_spans(Lines)
        # once the dict_spantext_to_labels is filled with a mapping from spantext to corresponding labels, write it out in a one hot vector
        with open(OUTPUT_FILE_NAME, 'a') as out:
            counter=0
            line_counter=0
            positive_examples_counter=0
            negative_examples_counter = 0
            for datapoint, labels in dict_spantext_to_labels.items():
                line_counter+=1
                #one hot vector to finally write the datapoint vs labels as to disk e.g., text,[1,0,1]
                # i.e.,maximum one hot vector will be all 1s
                labels_onehot = [0]*len(labels_in_this_training)
                write_flag=False
                if datapoint!=None:
                    #if there is more than one label for the given span update the one hot vector to 1
                    if len(labels) > 1:
                        for lblindx,label in enumerate(labels):
                            if label in dict_all_labels_index:

                                label_index=dict_all_labels_index[label]
                                labels_onehot[label_index]=1

                        #if sum(labels_onehot)>0: #if atleast one label was found for this datapoint
                            write_flag=True
                    else:
                        #if that span has only one label it will be in labels[0]
                        if labels[0] in dict_all_labels_index:
                            label_index = dict_all_labels_index[labels[0]]
                            labels_onehot[label_index] = 1
                            write_flag = True

                #maximum one hot vector must be all 1s
                assert sum(labels_onehot)<=len(labels_in_this_training)
                if sum(labels_onehot) == 0:
                    negative_examples_counter += 1
                else:
                    positive_examples_counter += 1

                # writing to the disk
                # Note: this is an IO bottleneck. Should store everything in memory and write once ideally.
                if(write_flag==True):
                    oneHotString=",".join([str(x) for x in labels_onehot])
                    out.write(f"{counter},\"{datapoint}\",{oneHotString}\n")
                    counter = counter + 1
            print(f"total data points for label of type {TYPE_OF_LABEL} is {len(dict_spantext_to_labels)} of which "
                  f"there are {positive_examples_counter} positive examples and {negative_examples_counter} negative examples")


create_training_data()

