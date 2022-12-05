# read the piranha annotated data and convert it into the kaggle toxic comment format:
#"id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"
#"00190820581d90ce"," YOUR FILTHY MOTHER IN THE ASS, DRY!",1,0,1,0,1,0
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
import json
import csv
import hashlib
OUTPUT_FILE_NAME= "data/all_data.csv"
header=["id","text","message_contact_person_asking","message_contact_person_org","message_org"]
labels_all=["message_contact_person_asking","message_contact_person_org","message_org","sentence_intent_attachment","sentence_intent_click","sentence_intent_intro","sentence_intent_money","sentence_intent_phonecall","sentence_intent_products","sentence_intent_recruiting","sentence_intent_scheduling","sentence_intent_service","sentence_intent_unsubscribe","sentence_org_used_by_employer","sentence_passwd","sentence_tone_polite","sentence_tone_urgent","sentence_url_no_name","sentence_url_third_party","signature","signature_email","signature_fullname","signature_jobtitle","signature_org","signature_phone","signature_signoff","signature_url","signaure_address","signaure_handle","words_reciever_organization","words_sender_location","words_sender_organization"]
message_labels={}
message_labels={}

message_level_labels_index={
"message_contact_person_asking":0,
"message_contact_person_org":1,
"message_org":2,
}

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
                    full_text = get_spans(entry['start'], entry['end'], annotations)
                    return full_text



def get_message_level_text_labels(Lines):
    for index, line in enumerate(Lines):
        annotations = json.loads(line)
        if "spans" in annotations:
            for entry in annotations["spans"]:
                label = entry["label"]
                text=annotations['text']
                if "message" in label:
                    assert label in labels_all
                    # get the entire text of the email. note, this is being done only for message level labels.DO NOT USE THIS FOR SENTENCE LEVEL OR LESS< USE SPANS
                    if text is not None:
                        text=text.replace("\n","")
                        if text in message_labels:
                            old_value = message_labels[text]
                            if label not in old_value:
                                old_value.append(label)
                                message_labels[text] = old_value
                        else:
                            message_labels[text] = [label]


#given the start and end of a span return the collection of the tokens corresponding to this in string format
def get_spans(token_start_of_span, token_end_of_span, annotations):
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

with open("./data/enron_head_10.jsonl", 'r') as in_file:
    Lines = in_file.readlines()
    get_message_level_text_labels(Lines)

    with open(OUTPUT_FILE_NAME, 'a') as out:
        counter=0
        line_counter=0
        for sentence, labels in message_labels.items():
            line_counter+=1
            labels_onehot = [0, 0, 0]
            write_flag=False
            if sentence!=None:
                #get the index of message level label
                if len(labels) > 1:
                    for label in labels:
                        if "message" in label:
                            label_index=message_level_labels_index[label]
                            labels_onehot[label_index]=1
                    if sum(labels_onehot)>0:
                        write_flag=True
                else:
                    if "message" in labels[0]:
                        label_index = message_level_labels_index[label]
                        labels_onehot[label_index] = 1
                        write_flag = True
            assert sum(labels_onehot)<3
            if(write_flag==True):

                oneHotString=",".join([str(x) for x in labels_onehot])
                out.write(f"{counter},\"{sentence}\",{oneHotString}\n")
                counter = counter + 1




