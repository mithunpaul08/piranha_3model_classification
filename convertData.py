# high level: read the piranha annotated data and convert it into the kaggle toxic comment format. :
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
OUTPUT_FILE_NAME= "data/all_data.csv"
header=["id","text","message_contact_person_asking","message_contact_person_org","message_org"]
labels_all=["message_contact_person_asking","message_contact_person_org","message_org","sentence_intent_attachment","sentence_intent_click","sentence_intent_intro","sentence_intent_money","sentence_intent_phonecall","sentence_intent_products","sentence_intent_recruiting","sentence_intent_scheduling","sentence_intent_service","sentence_intent_unsubscribe","sentence_org_used_by_employer","sentence_passwd","sentence_tone_polite","sentence_tone_urgent","sentence_url_no_name","sentence_url_third_party","signature","signature_email","signature_fullname","signature_jobtitle","signature_org","signature_phone","signature_signoff","signature_url","signaure_address","signaure_handle","words_reciever_organization","words_sender_location","words_sender_organization"]
dict_spantext_to_labels={}




def create_label_index_mapping_both_directions():
    dict_all_labels_index = {}
    dict_all_index_labels={}
    for index,label in enumerate(labels_all):
        dict_all_labels_index[label]=index
        dict_all_index_labels[index]=label
    return (dict_all_labels_index,dict_all_index_labels)


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


#go through each of the spans, find each of the labels in the spans, and check if that label is one of the labels we are
#searching for. if yes, add it to a dictionary which maps text->label
def get_text_for_label_from_all_spans(Lines):
    for index, line in enumerate(Lines):
        annotations = json.loads(line)
        if "spans" in annotations:
            for entry in annotations["spans"]:
                label = entry["label"]
                if label in labels_all:
                    if "message" in label:
                        # if its a message level annotation
                        # get the entire text of the email.
                        # note, this is being done only for message level labels.
                        # DO NOT USE THIS FOR SENTENCE LEVEL OR LESS- USE SPANS
                        text = annotations['text']
                        if text is not None:
                            text=text.replace("\n","")
                            if text in dict_spantext_to_labels:
                                old_value = dict_spantext_to_labels[text]
                                if label not in old_value:
                                    old_value.append(label)
                                    dict_spantext_to_labels[text] = old_value
                            else:
                                dict_spantext_to_labels[text] = [label]
                    else:
                        #if it is sentence or word level or signature level label, get the actual span text
                        text=get_spans_text_given_start_end_tokens(entry['token_start'],entry['token_end'],annotations )
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

with open(OUTPUT_FILE_NAME, 'w') as out:
    out.write(",".join(header))
    out.write("\n")


with open("data/query_file.jsonl", 'r') as in_file:
    Lines = in_file.readlines()
    # go through each of the annotated data point, extract text and its label into a dictionary dict_spantext_to_labels
    get_text_for_label_from_all_spans(Lines)
    # once the dict_spantext_to_labels is filled with a mapping from spantext to corresponding labels, write it out in a one hot vector
    with open(OUTPUT_FILE_NAME, 'a') as out:
        counter=0
        line_counter=0
        dict_all_labels_index,dict_all_index_labels=create_label_index_mapping_both_directions()
        for sentence, labels in dict_spantext_to_labels.items():
            line_counter+=1
            #to check gold sentences for this label has been retreieved or not
            # maximum one hot vector must be all 1s
            labels_onehot = [0]*len(labels_all)
            write_flag=False
            if sentence!=None:
                #if there is more than one label for the given span
                if len(labels) > 1:
                    for label in labels:
                        if label in dict_all_labels_index:
                            label_index=dict_all_labels_index[label]
                            labels_onehot[label_index]=1
                    if sum(labels_onehot)>0:
                        write_flag=True
                else:
                    #if that span has only one label it will be in labels[0]
                    if labels[0] in dict_all_labels_index:
                        label_index = dict_all_labels_index[labels[0]]
                        labels_onehot[label_index] = 1
                        write_flag = True

            #maximum one hot vector must be all 1s
            #writing to the disk
            #Note: this is an IO bottleneck. Should store everything in memory and write once ideally.
            assert sum(labels_onehot)<=len(labels_all)
            if(write_flag==True):
                oneHotString=",".join([str(x) for x in labels_onehot])
                out.write(f"{counter},\"{sentence}\",{oneHotString}\n")
                counter = counter + 1




