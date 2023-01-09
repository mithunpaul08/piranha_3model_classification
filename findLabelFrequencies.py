import json

path_annotated_emails= "data/combined_all_annotation_jan9th2023.jsonl"
annotator1_name="uma"
annotator2_name="mithun"
labels_count={}
with open(path_annotated_emails, 'r') as annotated_file:
    Lines = annotated_file.readlines()
    for index, line in enumerate(Lines):
            annotations = json.loads(line)
            if "spans" in annotations:
                for entry in annotations["spans"]:
                    label = entry["label"]
                    if "_annotator_id" in annotations:
                            if annotator1_name in annotations['_annotator_id']:
                                if label in labels_count:
                                    old_count=labels_count[label]
                                    labels_count[label]=old_count+1
                                else:
                                    labels_count[label] = 1

print(labels_count)
print(len(labels_count))

