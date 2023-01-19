import json
import hashlib


def check_add_dict(input_dict,text):
    if text not in input_dict.keys():
        input_dict[text]=1
    return input_dict

unique_emails={}
unique_spans={}
path_annotated_emails= "data/all_ta3_emails.jsonl"
annotator1= "uma"
annotator2= "mithun"
labels_count={}
total_emails=0
with open(path_annotated_emails, 'r') as annotated_file:
    Lines = annotated_file.readlines()
    for index, line in enumerate(Lines):

            annotations = json.loads(line)
            if "spans" in annotations:
                for entry in annotations["spans"]:
                    label = entry["label"]
                    str_entry="".join(entry)
                    if "_annotator_id" in annotations:
                            if (annotator1 in annotations['_annotator_id'])  or(annotator2 in annotations['_annotator_id']):
                                unique_emails = check_add_dict(unique_emails, line)
                                unique_spans = check_add_dict(unique_spans, str_entry)
                                total_emails+=1
                                if label in labels_count:
                                    old_count=labels_count[label]
                                    labels_count[label]=old_count+1
                                else:
                                    labels_count[label] = 1

print(labels_count)
print(len(unique_emails))
print(len(unique_spans))



