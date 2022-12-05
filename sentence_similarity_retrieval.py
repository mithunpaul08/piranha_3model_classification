#if there are labels that doesnt have high representation in piranha annotated emails, retrieve emails from the unannotated
#dataset which are similar to the very few labels we have so far.

import json
import hashlib
from tqdm import tqdm
from pysbd.utils import PySBDFactory
import pysbd
from sentence_transformers import SentenceTransformer, util
import convertData
import sys

#how many emails do you want ot retireve for each label. if you hit this number break the loop and move onto the next label
NO_OF_EMAILS_TO_RETRIEVE_PER_LABEL=50

COSINE_SIM_THRESHOLD=0.5
#how many emails in the unannotated dataset should we search through. i.e we cant search through all of 600k emails in enron
#so even after searching NO_OF_MAX_EMAILS_TO_SEARCH_THROUGH emails, we can't find 50 emails of the given label, we quit and move onto next label.
NO_OF_MAX_EMAILS_TO_SEARCH_THROUGH=1000

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
PATH_RETRIEVED_EMAILS_FILE="output/retrieved_emails.jsonl"


#list of labels for which the emails have to be retrievedl
LABELS_TO_RETRIEVE=["signature_fullname", "sentence_tone_urgent", "sentence_url_no_name", "sentence_intent_products", "signature_signoff", "words_sender_location", "signature_phone", "sentence_url_third_party", "sentence_intent_unsubscribe", "sentence_intent_attachment", "signature_org", "sentence_org_used_by_employer", "signature_jobtitle", "sentence_passwd", "signature_email", "sentence_intent_recruiting", "signature_address", "signature_url", "words_receiver_organization", "sentence_intent_intro", "words_sender_organization"]

#a serial number assigning dict - to use in bit vector
label_index={}
for index, label in enumerate(LABELS_TO_RETRIEVE):
    label_index[label]=index

##a bit vector to check which all labels have already been retrieved
bit_vector_retrieved_labels=[0]*len(LABELS_TO_RETRIEVE)




#the ones which will be used as gold emails to retrieve similar ones
path_annotated_emails="./data/enron_combined_all_uma_annotations_so_far_extraction_nov30th2022.jsonl"

#the ones from which data will be retreived
path_non_annotated_emails="./data/enron_head_10.jsonl"
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

#for each given input email retrieve 10 similar emails
# annotations can be marked as spans (sentence, messages) or tokens (one word).
# Given a type of annotation marker (e.g., span) this code will return a sentence/word of that type
def get_similar_emails(annotation_type,label):
    top_retrieved=[]
    if (annotation_type in annotations):
        spans = annotations[annotation_type]
        for each_annotation in spans:
            try:
                if "label" in each_annotation:
                    if (each_annotation['label'] == label):
                        annotated_text = ((annotations['text']).strip().replace("\n", " "))

                        #find the most similar sentence in each email. not just the whole email as an embedding
                        seg = pysbd.Segmenter(language="en", clean=False)

                        #store the scores of each email in a list. one on one mapping to non_annotated_emails_text:
                        email_scores=[]
                        for index,na_email in enumerate(non_annotated_emails_text):

                            # if the email has "---" it is probably the reply to part. remove it.
                            common_reply_splitters=["From:","________________________________","'‐‐‐‐‐‐‐ Original Message","Original Message"]
                            if any(splitter in na_email for splitter in common_reply_splitters):
                                this_splitter=""
                                for splitter in common_reply_splitters:
                                    if splitter in na_email:
                                        this_splitter=splitter
                                        break
                                assert this_splitter!=""
                                na_email_split=na_email.split(this_splitter)
                                na_email=na_email_split[0]
                            email_split_sentences=seg.segment(na_email)

                            if len(email_split_sentences)>NO_OF_EMAILS_TO_RETRIEVE_PER_LABEL:
                                #add code to break out of this loop and go to next label
                                pass

                            embedding_1 = model.encode(annotated_text, convert_to_tensor=True)
                            embedding_2 = model.encode(email_split_sentences, convert_to_tensor=True)

                            email_sent_scores = util.pytorch_cos_sim(embedding_1, embedding_2)
                            #for each email sum up the scores for individual sentences so that we have a single score per email.
                            score_email=sum(email_sent_scores)
                            email_scores.append(score_email)

                        assert len(email_scores)==len(non_annotated_emails_text)
                        #now from all the emails and their scores, pick top 10
                        if len(email_scores)>0:
                            data_sorted= sorted(email_scores,reverse=True)
                            top10=data_sorted[0:9]

                            for score in top10:
                                #for each of the 10 top emails, find the corresponding index
                                highest_score_index = email_scores.index(score)
                                #get the email text from that index
                                retrieved_email=non_annotated_emails_text[highest_score_index]
                                sha_retrieved_email=hashlib.sha256(retrieved_email.encode('utf-8')).hexdigest()
                                if sha_retrieved_email not in retrieved_emails_sha:
                                    retrieved_emails_sha[sha_retrieved_email]=1
                                    top_retrieved.append(retrieved_email)
            except Exception as e:
                print(e)

    return top_retrieved


with open(PATH_RETRIEVED_EMAILS_FILE, mode="w") as writer:
    writer.write("")


    overall_retrieved_emails=[]
    for label,query_text in label_text_gold.items():
        retrieved_emails_per_label = []
        for overall_unannotated_emails_parsed_counter,each_email in enumerate(non_annotated_emails_text):
            if overall_unannotated_emails_parsed_counter<NO_OF_MAX_EMAILS_TO_SEARCH_THROUGH or len(retrieved_emails_per_label)<NO_OF_EMAILS_TO_RETRIEVE_PER_LABEL:
                if "message" not in label:
                    seg = pysbd.Segmenter(language="en", clean=True)
                    email_split_sentences = seg.segment(each_email)
                    for result_text in email_split_sentences:
                        embedding_1 = model.encode(result_text, convert_to_tensor=False)
                        embedding_2 = model.encode(query_text, convert_to_tensor=False)
                        cosine_sim = util.pytorch_cos_sim(embedding_1, embedding_2)
                        if cosine_sim.item()>COSINE_SIM_THRESHOLD:
                            retrieved_emails_per_label.append(each_email)
        if len(retrieved_emails_per_label)>0:
            overall_retrieved_emails.extend(retrieved_emails_per_label)

            with open(PATH_RETRIEVED_EMAILS_FILE, mode="a") as writer:
                for each_retrieved_email in retrieved_emails_per_label:
                    all_emails_text_this_label = "\n".join(each_retrieved_email)
                    writer.write(each_retrieved_email)





