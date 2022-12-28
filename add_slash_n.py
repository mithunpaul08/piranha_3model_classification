#after retrieving data using sentence_similarity_retrieval it was found that there was no new line after each sentence.
#this code with use pybsd to segment sentence, add \n and write it back to disk
import json
import pysbd

INPUT_FILE="./data/fraud_db_focused_retrieval_dec27th2022.jsonl"
OUTPUT_FILE="./output/with_slashn_fraud_db_focused_retrieval_dec27th2022.jsonl"

all_emails=[]
with open(INPUT_FILE) as input_file:
        lines=input_file.readlines()
        for each_email in lines:
            dict_input_email=json.loads(each_email)
            dict_emails_with_slashn = {}
            seg = pysbd.Segmenter(language="en", clean=True)
            email_split_sentences = seg.segment(dict_input_email['text'])
            with_slash_n="\n".join(email_split_sentences)
            dict_emails_with_slashn["text"]=with_slash_n
            all_emails.append(dict_emails_with_slashn)

        input_file.close()
with open(OUTPUT_FILE,"w") as output_file:
    output_file.write("")

with open(OUTPUT_FILE,"a") as output_file:
        for each_slashn_email in all_emails:
            json.dump(each_slashn_email,output_file)
            output_file.write("\n")



