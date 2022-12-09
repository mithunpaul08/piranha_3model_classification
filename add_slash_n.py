#after retrieving data using sentence_similarity_retrieval it was found that there was no new line after each sentence.
#this code with use pybsd to segment sentence, add \n and write it back to disk
import json
import pysbd

INPUT_FILE="./output/combined_retrieved_emails_dec5th_6th.jsonl"
OUTPUT_FILE="./output/combined_retrieved_emails_dec5th_6th_with_slash_n.jsonl"

all_emails=[]
with open(INPUT_FILE) as input_file:
    dict_emails_with_slashn = {}
    lines=input_file.readlines()
    for each_email in lines:
        seg = pysbd.Segmenter(language="en", clean=True)
        email_split_sentences = seg.segment(each_email)
        with_slash_n="\n".join(email_split_sentences)
        dict_emails_with_slashn["text"]=with_slash_n
        all_emails.append(dict_emails_with_slashn)

    input_file.close()

with open(OUTPUT_FILE) as output_file:
    for each_slashn_email in all_emails:
        json.dump(each_slashn_email,output_file)
        output_file.write("\n")



