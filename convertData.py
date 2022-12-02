# read the piranha annotated data and convert it into the kaggle toxic comment format:
#"id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"
#"00190820581d90ce","FUCK YOUR FILTHY MOTHER IN THE ASS, DRY!",1,0,1,0,1,0
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
import json
import csv
import hashlib
OUTPUT_FILE_NAME= "data/train.csv"

labels_all=["message_contact_person_asking","message_contact_person_org","message_org","sentence_intent_attachment","sentence_intent_click","sentence_intent_intro","sentence_intent_money","sentence_intent_phonecall","sentence_intent_products","sentence_intent_recruiting","sentence_intent_scheduling","sentence_intent_service","sentence_intent_unsubscribe","sentence_org_used_by_employer","sentence_passwd","sentence_tone_polite","sentence_tone_urgent","sentence_url_no_name","sentence_url_third_party","signature","signature_email","signature_fullname","signature_jobtitle","signature_org","signature_phone","signature_signoff","signature_url","signaure_address","signaure_handle","words_reciever_organization","words_sender_location","words_sender_organization"]
sentence_labels={}


message_level_labels_index={
"message_contact_person_asking":0,
"message_contact_person_org":1,
"message_org":2,
}

#given the start and end of a span return the collection of the tokens corresponding to this in string format
def get_text(span_start, span_end, annotations):
    starts_ends_tokens = []
    for token in annotations['tokens']:
        if (token['start']>=span_start and token['end']<=span_end):
            starts_ends_tokens.append(token['text'])
        if (token['start'] >= span_start and token['end'] > span_end):
            return " ".join(starts_ends_tokens)


with open(OUTPUT_FILE_NAME, 'w') as out:
    out.write("")

with open("/Users/mitch/research/piranha/prodigy-tools/datasets/ta3_complete_extraction_nov30th2022_onlyuma.jsonl", 'r') as in_file:
    Lines = in_file.readlines()
    for index,line in enumerate(Lines):
        annotations = json.loads(line)
        if "spans" in annotations:
            for entry in annotations["spans"]:
                label=entry["label"]
                assert label in labels_all
                full_text = get_text(entry['start'], entry['end'], annotations)
                if full_text is not None:
                    if full_text in sentence_labels:
                            old_value= sentence_labels[full_text]
                            if label not in old_value:
                                old_value.append(label)
                                sentence_labels[full_text]=old_value
                    else:
                            sentence_labels[full_text] = [label]




    with open(OUTPUT_FILE_NAME, 'a') as out:
        counter=0
        line_counter=0
        for sentence, labels in sentence_labels.items():
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
                out.write(f"{counter},\"{sentence}\",{labels_onehot}\n")
                counter = counter + 1




