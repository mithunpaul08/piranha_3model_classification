#will take the fraudulent email foundin the following dataset and convert it to piranha format of {"text":"email body"}
#format which can then be directly loaded to prodigy for annotation. WIll add \n also after each line
#https://www.kaggle.com/datasets/llabhishekll/fraud-email-dataset
import json
import pysbd
import csv
from tqdm import tqdm

INPUT_FILE="./data/fraud_email_.csv"
OUTPUT_FILE="./output/fraud_email_piranha_format.jsonl"

all_emails=[]
with open(INPUT_FILE) as input_file:
    lines=csv.reader(input_file,delimiter=",")
    for index,each_email in enumerate(tqdm(lines,total=len(lines))):
        if index>0:
            dict_emails_with_slashn = {}
            seg = pysbd.Segmenter(language="en", clean=True)
            #in the fraudulent email format the first entry in a list is the email body and the second entry is a label that we dont care for
            email_split_sentences = seg.segment(each_email[0])
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



