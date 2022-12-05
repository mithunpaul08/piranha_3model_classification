#if there are labels that doesnt have high representation in piranha annotated emails, retrieve emails from the unannotated
#dataset which are similar to the very few labels we have so far.

import json
import hashlib
import random
import datetime
import traceback
from tqdm import tqdm
from pysbd.utils import PySBDFactory
import pysbd
from sentence_transformers import SentenceTransformer, util
import convertData
import sys
import json
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'
print(f"***********found that the device available is a {device}\n")
#how many emails do you want ot retireve for each label. if you hit this number break the loop and move onto the next label
NO_OF_EMAILS_TO_RETRIEVE_PER_LABEL=10

COSINE_SIM_THRESHOLD=0.75
#how many emails in the unannotated dataset should we search through. i.e we cant search through all of 600k emails in enron
#so even after searching NO_OF_MAX_EMAILS_TO_SEARCH_THROUGH emails, we can't find 50 emails of the given label, we quit and move onto next label.
NO_OF_MAX_EMAILS_TO_SEARCH_THROUGH=10000

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model.to(device)
PATH_RETRIEVED_EMAILS_FILE="output/retrieved_emails"+str(datetime.datetime.now())+".jsonl"
PATH_PER_SIGNATURE_RETREIVED_EMAILS="output/per_signature_retrieved_emails"+str(datetime.datetime.now())+".jsonl"


#list of labels for which the emails have to be retrievedl
#all of them LABELS_TO_RETRIEVE=["signature_fullname", "sentence_tone_urgent", "sentence_url_no_name", "sentence_intent_products", "signature_signoff", "words_sender_location", "signature_phone", "sentence_url_third_party", "sentence_intent_unsubscribe", "sentence_intent_attachment", "signature_org", "sentence_org_used_by_employer", "signature_jobtitle", "sentence_passwd", "signature_email", "sentence_intent_recruiting", "signature_address", "signature_url", "words_receiver_organization", "sentence_intent_intro", "words_sender_organization"]

#the top 20- or weakest 20- will retrieve 10 emails per
LABELS_TO_RETRIEVE=[ "sentence_org_used_by_employer", "signature_jobtitle", "sentence_passwd", "signature_email", "sentence_intent_recruiting", "signature_address", "signature_url", "words_receiver_organization", "sentence_intent_intro", "words_sender_organization"]

#stronger20- will retrieve only 5 emails per
#LABELS_TO_RETRIEVE=["signature_fullname", "sentence_tone_urgent", "sentence_url_no_name", "sentence_intent_products", "signature_signoff", "words_sender_location", "signature_phone", "sentence_url_third_party", "sentence_intent_unsubscribe", "sentence_intent_attachment", "signature_org",]

#a serial number assigning dict - to use in bit vector
label_index={}
for index, label in enumerate(LABELS_TO_RETRIEVE):
    label_index[label]=index

##a bit vector to check which all labels have already been retrieved
bit_vector_retrieved_labels=[0]*len(LABELS_TO_RETRIEVE)




#the ones which will be used as gold emails to retrieve similar ones
path_annotated_emails= "data/query_file.jsonl"

#the ones from which data will be retreived
path_non_annotated_emails= "data/file_to_do_information_retrieval.jsonl"
#forserver
#path_non_annotated_emails="./datasets/enron_head_5k.jsonl"

#a dictionary to store each label and a gold text with sentences containing that label
label_text_gold={}

with open(path_annotated_emails, 'r') as annotated_file:
    # for label_to_check in LABELS_TO_RETRIEVE:
    #     full_text=convertData.given_label_retrieve_gold_text(annotated_file, label_to_check)
    #     label_text_gold[label]=full_text

    Lines = annotated_file.readlines()
    for index, line in enumerate(Lines):
        if sum(bit_vector_retrieved_labels) <= len(LABELS_TO_RETRIEVE):
            annotations = json.loads(line)
            if "spans" in annotations:
                for entry in annotations["spans"]:
                    label = entry["label"]
                    if label in label_index:
                        lbl_index=label_index[label]
                        if bit_vector_retrieved_labels[lbl_index]==0:
                            full_text = convertData.get_spans(entry['token_start'], entry['token_end'], annotations)
                            if full_text is not None:
                                full_text=full_text.replace("\n","")
                                bit_vector_retrieved_labels[lbl_index] = 1
                                label_text_gold[label]=full_text
                            else:
                                print(f"Error:Found no gold text for this label {label}. Going to exit ")
                                sys.exit()


    #annotated_emails = annotated_file.readlines()

with open('output/labels_gold_texts.jsonl', mode="w") as writer:
    for k,v in label_text_gold.items():
        writer.write(f"{k}:{v}\n")


##########now that we have gold text/emails for the low frequency labels, lets go retrieve more similar emails form the unannotated text
non_annotated_emails_text=[]
retrieved_emails={}
#to ensure duplicatees are not added, checking each emails sha in a dictionary
retrieved_emails_sha={}



#store all non_annotated_emails in memory
with open(path_non_annotated_emails, 'r') as non_annotated_file:
    non_annotated_emails = non_annotated_file.readlines()
    for non_annotated_email in non_annotated_emails:
        annotations = json.loads(non_annotated_email)
        non_annotated_emails_text.append((annotations['text']).strip().replace("\n", " "))

def split_reply_part_email(na_email):
    # if the email has "---" it is probably the reply to part. remove it.
    common_reply_splitters = ["From:", "________________________________", "'‐‐‐‐‐‐‐ Original Message",
                              "Original Message","-----Original Message-----","To:","Subject:","cc:"]
    for splitter in common_reply_splitters:
        if splitter in na_email:
            this_splitter = ""
            for splitter in common_reply_splitters:
                if splitter in na_email:
                    this_splitter = splitter
                    break
            assert this_splitter != ""
            na_email_split = na_email.split(this_splitter)
            if len(na_email_split)>0:
                return na_email_split[0]
    return na_email


with open(PATH_RETRIEVED_EMAILS_FILE, mode="w") as writer:
    writer.write("")

# a dictionary which maps each label to the n emails retrieved per that label- and the sentence in that email which was retrieved
label_retrieved_emails={}

#if this email has already been retrieved for some reason, it is useless spending cpu cycles retrieving it again
check_if_unique_email={}
# a list of dictionaries which contains each of the retrieved emails
overall_retrieved_emails=[]
label_counter=0
try:
    for label,query_text in tqdm(label_text_gold.items(),desc="labels",total=len(label_text_gold.items())):
        random.shuffle(non_annotated_emails_text)
        retrieved_emails_per_label = []
        for overall_unannotated_emails_parsed_counter,each_retrieved_email in enumerate(tqdm(non_annotated_emails_text,desc="retrieving_emails",total=len(non_annotated_emails_text))):
            retrieved_texts_json_format={}
            if overall_unannotated_emails_parsed_counter<NO_OF_MAX_EMAILS_TO_SEARCH_THROUGH or len(retrieved_emails_per_label)<NO_OF_EMAILS_TO_RETRIEVE_PER_LABEL:
                if "message" not in label:
                    each_retrieved_email=split_reply_part_email(each_retrieved_email)
                    seg = pysbd.Segmenter(language="en", clean=True)
                    email_split_sentences = seg.segment(each_retrieved_email)
                    for result_text in email_split_sentences:
                        embedding_1 = model.encode(result_text, convert_to_tensor=False)
                        embedding_2 = model.encode(query_text, convert_to_tensor=False)
                        cosine_sim = util.pytorch_cos_sim(embedding_1, embedding_2)
                        if cosine_sim.item()>COSINE_SIM_THRESHOLD:
                            retrieved_texts_json_format["text"]=each_retrieved_email
                            if each_retrieved_email not in check_if_unique_email:
                                check_if_unique_email[each_retrieved_email]=0
                                if label in label_retrieved_emails:
                                    current_emails=label_retrieved_emails[label]
                                    current_emails.append((each_retrieved_email,result_text))
                                    label_retrieved_emails[label] = current_emails
                                else:
                                    label_retrieved_emails[label]=[(each_retrieved_email,result_text)]
                                retrieved_emails_per_label.append(retrieved_texts_json_format)
            else:
                break
except:
    traceback.print_exc()


    if len(retrieved_emails_per_label)>0:
        overall_retrieved_emails.extend(retrieved_emails_per_label)

        with open(PATH_RETRIEVED_EMAILS_FILE, mode="a") as writer:
            for each_email in retrieved_emails_per_label:
                json.dump(each_email,writer)
                writer.write("\n")

with open(PATH_PER_SIGNATURE_RETREIVED_EMAILS, mode="w") as writer:

    for label,list_emails in label_retrieved_emails. items():
        writer.write("-------------\n")
        writer.write(f"label:{label}\n")
        for index,each_email in enumerate(list_emails):
            writer.write(f"{index}:\n")
            writer.write(f"retrieved_sentence:{each_email[1]}\n retrieved_email:{each_email[0]}\n")






