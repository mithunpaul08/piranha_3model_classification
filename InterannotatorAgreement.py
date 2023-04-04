# Neel Gupta
# 09/01/2022
# Check how many of the annotations agree between annotators

import json
import sys

numAnnotators = 2
emailAnnotatorsCount = {}
emails = {}
annotators_annotations = {}
annotator1Idx = 0
annotator2Idx = 0
annotator1_name="mithun"
annotator2_name="uma"
#what kind of label would you like to know more inter annotator details about [message,sentence, token]
label_stub="sentence"

flag_print_hashes_text_emails=False
flag_show_kappa_cohen=True
flag_show_per_person_labels_missed=True

#final output print should look like- of all emails
# no of emails both annotated:
count_emails_both_annotated=0
# no of labels annotator1 annotated:
count_labels_annotated_by_annotator1=0
# no of emails annotator2 annotated:
count_labels_annotated_by_annotator2=0
# no of labels both got right
count_labels_intersection=0
labels_per_email=0
cumulative_of_per_email_agreement=0
annotator_id_vs_name={1:annotator1_name, 2:annotator2_name}
hash_vs_text={}



print(f" Analysis of  {label_stub} level labels annotations between {annotator2_name} vs {annotator1_name}")


#separate out emails annotated by annotator1_name and annotator2_name
#laptop


with open("data/combined_all_annotation_jan9th2023.jsonl", 'r') as f:

#server
#with open("/Users/mitch/research/piranha/annotated_datasets/ta3_reloading_oct18th_message_level_annotated_3annotators_oct26th_extraction.jsonl", 'r') as f:

    Lines = f.readlines()
    for line in Lines:
        entry = json.loads(line)
        email_hash = entry['_input_hash']
        if email_hash == int("1767252249"):
            pass
        if email_hash in emailAnnotatorsCount:
            emailAnnotatorsCount[email_hash] += 1
        else:
            emailAnnotatorsCount[email_hash] = 1
        emails[email_hash] = entry['text']

        if 'spans' in entry:
            if "_annotator_id" in entry:

                if annotator1_name in entry['_annotator_id'] :
                    annotators_annotations[annotator1_name+"_"+str(email_hash)] = entry['spans']
                    annotator1Idx += 1
                elif  annotator2_name in entry['_annotator_id'] :
                    annotators_annotations[annotator2_name+"_"+str(email_hash)] = entry['spans']
                    annotator2Idx += 1
if (flag_print_hashes_text_emails==True):
    for hash,text in (emails.items()):
            print(f"hash:{hash}\ntext:{text}\n****************************")
            print("\n")



emails_annotators_list = {}
#by now annotators_annotations will contain only those of 2 required annotators
#then take each email hash and find the number of people who annotated this and store it in emails_annotators_list.
# Rather, if its just one of the 2 annotators or both. This is useful in finding intersecting annotaions
for key in annotators_annotations:
    notes = annotators_annotations[key]
    keyarr = key.split('_')
    name = keyarr[0]
    email_hash = keyarr[1]
    if email_hash == "1767252249":
        pass
    if email_hash in emails_annotators_list:
        emails_annotators_list[email_hash].append(name)
    else:
        emails_annotators_list[email_hash] = [name]



intersecting_annotations = {}
#now separate out each email hash with labels annotated by _1 and annotated _2. this will be stored in intersecting_annotations
for email_hash in emails_annotators_list:
    if len(emails_annotators_list[email_hash]) == 2: #numAnnotators:
        startIdx = 1
        for name in emails_annotators_list[email_hash]:
            notes = annotators_annotations[name+'_'+email_hash]
            intersecting_annotations[email_hash+'_'+str(startIdx)] = notes
            startIdx += 1

track_labels_per_email={}

#e.g., {19239202; dict(ann1;set(labels1), ann2:set(labels2))}
email_hash_vs_bothannotatorlabelset={}

for k,v in intersecting_annotations.items():
    splits = k.split("_")
    hash = splits[0]
    anotattor = splits[1]
    labels_this_annotator=[]
    for entry in v:
        if "label" in entry:
            labels_this_annotator.append(entry["label"])
    set_labels_this_annotator=set(labels_this_annotator)
    this_annotator_vs_labelset_dict_charlie={}
    this_annotator_vs_labelset_dict_charlie[anotattor]=set_labels_this_annotator

    if hash in email_hash_vs_bothannotatorlabelset:
        dict_charlie=email_hash_vs_bothannotatorlabelset[hash]
        if anotattor in dict_charlie:
            print("error. this annotator shouldnt be here")
        else:
            dict_charlie[anotattor]=set_labels_this_annotator
            email_hash_vs_bothannotatorlabelset[hash]=dict_charlie
    else:
        email_hash_vs_bothannotatorlabelset[hash]=this_annotator_vs_labelset_dict_charlie


annotator_message_label_vs_count={}
dict_hash_messageLevelLabel_vs_annotatorId={}
list_of_shortlisted_emails_with_this_label_stub=[]
#go through each of the email vs annotator set labels and see if there is a label_stub*
for hash,dict_email_hash_label_set in email_hash_vs_bothannotatorlabelset.items():
    for annotator_id, setlabels in dict_email_hash_label_set.items():
        for label in setlabels:
            if label_stub  in label:
                list_of_shortlisted_emails_with_this_label_stub.append(hash)
                key_hash_label = hash + "@" + label
                if label in dict_hash_messageLevelLabel_vs_annotatorId:

                    current=dict_hash_messageLevelLabel_vs_annotatorId[label]
                    current.append(annotator_id)
                    dict_hash_messageLevelLabel_vs_annotatorId[key_hash_label]=current
                else:
                    dict_hash_messageLevelLabel_vs_annotatorId[key_hash_label] = [annotator_id]


                name_annotator=annotator_id_vs_name[int(annotator_id)]
                key_annotator_label= name_annotator +"_" + label
                if key_annotator_label in annotator_message_label_vs_count:
                    current_value=annotator_message_label_vs_count[key_annotator_label]
                    annotator_message_label_vs_count[key_annotator_label]  =current_value+1
                else:
                    annotator_message_label_vs_count[key_annotator_label]=1
# print(f"{label_stub} level label count overall (doesnt reflect per email matched or not)")
# for k,v in annotator_message_label_vs_count.items():
#     print(f" {k}:{v}")

if len(dict_hash_messageLevelLabel_vs_annotatorId.items())==0:
    print(f"No messages for the given label stub {label_stub}")
    sys.exit(1)
print(f"-------------------------\n{label_stub} label per email")

if(flag_show_per_person_labels_missed==True):
    for k,v in dict_hash_messageLevelLabel_vs_annotatorId.items():
        if len(v)<2:
            split_key=k.split("@")
            email_hash=split_key[0]
            label=split_key[1]
            annotator_id=v[0]
            annotator_name=annotator_id_vs_name[int(annotator_id)]
            # print(f" For email with hash {email_hash} only {annotator_name} annotated the label {label}")



def cohen_kappa(ann1, ann2):
    """Computes Cohen kappa for pair-wise annotators.
    :param ann1: annotations provided by first annotator
    :type ann1: list
    :param ann2: annotations provided by second annotator
    :type ann2: list
    :rtype: float
    :return: Cohen kappa statistic
    """
    count = 0
    for an1, an2 in zip(ann1, ann2):
        if an1 == an2:
            count += 1
    A = count / len(ann1)  # observed agreement A (Po)

    uniq = set(ann1 + ann2)
    E = 0  # expected agreement E (Pe)
    for item in uniq:
        cnt1 = ann1.count(item)
        cnt2 = ann2.count(item)
        count = ((cnt1 / len(ann1)) * (cnt2 / len(ann2)))
        E += count

    return round((A - E) / (1 - E), 4)


labels_per_email=0
count_emails_both=0
email_hash_labels_annotator={}
cohen_kappa_score_overall=0

#given a list and max number fill it with empty nodes to max lenght
def fill_list_empty_nodes(list_name, max):
    for x in range(max-len(list_name)):
        #adding a label zzz because earlier we were adding blank and after sorting, blank was coming first
        list_name.append("zzzzzz")
    return list_name

def get_label_stub_labels_only(label_stub,labels_this_annotator):
    list_labels_annotator2_shortlist_with_stub_label=[]
    for labels in labels_this_annotator:
        if label_stub in labels:
            list_labels_annotator2_shortlist_with_stub_label.append(labels)
    return list_labels_annotator2_shortlist_with_stub_label


dict_per_tag_agreement_count={}

#given a dict and a key, get its value, increase by one put it back. else add as one
def increase_counter(d,k,value_to_increase_by):
    if k in d:
        v=d[k]
        v=v+value_to_increase_by
        d[k]=v
    else:
        d[k] = 0
    return d




#main workhorse: for annotations by each annotator, check if his label was annotated by the second guy also
for index, hash_annotator_id in enumerate(intersecting_annotations.keys()):
    ehasharr = hash_annotator_id.split('_')
    email_hash = ehasharr[0]
    annotatorNum = ehasharr[1]

    #check if that email_hash exists in the shortlist with label_stub
    if email_hash in list_of_shortlisted_emails_with_this_label_stub:
    #if the email hash already existsin the dictionary- we know we are at the second annotator (we call him annotator1- that needs to change). if yes
    # retrive the labels anotated bhy first guy which would hav ebeen stored in it by the else below.
        if email_hash in email_hash_labels_annotator:
            #get the labels by first guy- which was stored in the dictionary email_hash_labels_annotator by the else statement below
            set_labels_annotator1 = email_hash_labels_annotator[email_hash]
            #then go get the labels by second guy
            labels_annotator1 = []
            # find the list of labels annotated by this guy
            for each_annotation in intersecting_annotations[hash_annotator_id]:
                labels_annotator1.append(each_annotation["label"])
            list_labels_annotator1_shortlist_with_stub_label=get_label_stub_labels_only(label_stub,labels_annotator1)

            count_labels_annotated_by_annotator1 += len(list_labels_annotator1_shortlist_with_stub_label)
            #labels_annotator2_set = set(labels_annotator1)

            #find intersection-print
            intersecting_labels=(set(list_labels_annotator1_shortlist_with_stub_label).intersection(
                set(list_labels_annotator2_shortlist_with_stub_label)))
            count_labels_intersection+= len(intersecting_labels)

            denominator=max(len(set(list_labels_annotator1_shortlist_with_stub_label)),len(set(list_labels_annotator2_shortlist_with_stub_label)))
            #to calculate the per LABEL agreement
            #OUT OF 4 labels in this email, did 1 overlap. so 4 is the
            for each_overlapping_label in intersecting_labels:
                dict_per_tag_agreement_count=increase_counter(dict_per_tag_agreement_count,each_overlapping_label,1/denominator)

            labels_per_email+= max(len(set(list_labels_annotator1_shortlist_with_stub_label)),len(set(list_labels_annotator2_shortlist_with_stub_label)))
            cumulative_of_per_email_agreement += (count_labels_intersection/labels_per_email)
            count_emails_both += 1
            if(flag_show_kappa_cohen==True):

                #kappa cohen fails for len=1
                if(len(list_labels_annotator1_shortlist_with_stub_label) == len(list_labels_annotator2_shortlist_with_stub_label)):
                    if ((list_labels_annotator1_shortlist_with_stub_label) == (
                            list_labels_annotator2_shortlist_with_stub_label) ):
                        cohen_kappa_score_overall += 1
                else:
                    # kappa cohen doesnt like different lenghts
                    # pass lists
                    # sort them
                    # if the lenght of lists are differnt, fill the smaller one with junk values

                    if(len(list_labels_annotator1_shortlist_with_stub_label) != len(list_labels_annotator2_shortlist_with_stub_label)):
                        if len(list_labels_annotator1_shortlist_with_stub_label) > len(list_labels_annotator2_shortlist_with_stub_label):
                            fill_list_empty_nodes(list_labels_annotator2_shortlist_with_stub_label,len(list_labels_annotator1_shortlist_with_stub_label))
                        else:
                            fill_list_empty_nodes(list_labels_annotator1_shortlist_with_stub_label, len(list_labels_annotator2_shortlist_with_stub_label))
                        #this is a hack, if there is only one overlapping label in atleast one of the sets, just assign them both as same
                        #labels_annotator2_set==set_labels_annotator1
                        #cohen_kappa_score_overall += cohen_kappa(sorted(labels_annotator2_set), sorted(set_labels_annotator1))
                        # cohen_kappa_score_overall +=1
                    cohen_kappa_score_overall+=cohen_kappa(sorted(list_labels_annotator1_shortlist_with_stub_label),sorted(list_labels_annotator2_shortlist_with_stub_label))

        else:
            #so the email hash doesnt exist. i.e this is the first time we are seeing this email. add the annotations of second annotator to it
            labels_annotator2=[]
            #find the list of labels annotated by this guy- which has label_stub
            #todo_ dont go back and retrieve it from intersecting_annotations
            #note: here we just call the first hash_annotator_id as annotator2, while the if will have annotator_1 
            for each_annotation in intersecting_annotations[hash_annotator_id]:
                labels_annotator2.append(each_annotation["label"])
            count_labels_annotated_by_annotator2 += len(labels_annotator2)
            list_labels_annotator2_shortlist_with_stub_label=get_label_stub_labels_only(label_stub, labels_annotator2)
            # no more set businesslabels_annotator2_set=set(labels_annotator2)
            #email_hash_labels_annotator[email_hash]=labels_annotator2_set
            email_hash_labels_annotator[email_hash] =list_labels_annotator2_shortlist_with_stub_label


if len(intersecting_annotations.keys())==0:
    print("no common emails annotated between the given annotators")
else:
    percentage_v1=count_labels_intersection/labels_per_email
    percentage_v2=cumulative_of_per_email_agreement/count_emails_both
    count_emails_both_annotated=len(email_hash_labels_annotator)
    print(dict_per_tag_agreement_count)

    #
    # print(f"total labels_per_email:{labels_per_email}")
    # print(f"count_labels_intersection:{count_labels_intersection}")
    # print(f"count_emails_both:{count_emails_both}")
    # print(f"cumulative_of_per_email_agreement:{cumulative_of_per_email_agreement}")
    #
    print(f"all intersection/alllabels ratio={percentage_v1},")


    #print(f"Inter annotator agreement-cumulative_average_of_per_email_ratio={percentage_v2},")
    if(flag_show_kappa_cohen):
        print(f"average kappa cohen score={cohen_kappa_score_overall/count_emails_both},")



        