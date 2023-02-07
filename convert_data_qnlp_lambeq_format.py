# high level: read the piranha annotated data and convert it into the kaggle toxic comment format as shown below. This will be
#format:# used for running the training:
#"id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"
#"00190820581d90ce"," YOUR FILTHY MOTHER IN THE ASS, DRY!",1,0,1,0,1,0
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
#this will be used for multilabel classification in train.py

#detailed level:
# go through each of the annotated data point, extract text and its label into a dictionary dict_spantext_to_labels
# then create a one hot vector based on how many labels a given text corresponds to/was marked with
#e.g. an entire email could be marked with multiple labels. remember it is a multilabel classifier
import json
import csv
import hashlib

from configs import *
dict_spantext_to_labels={}
dict_all_labels_index = {}
dict_all_index_labels = {}
labels_in_this_training=[]
seg = pysbd.Segmenter(language="en", clean=True)

#creating  different input data for each of messsage level, sentence level, signature, word


print(f"found that the type of label in this training run is: {TYPE_OF_LABEL}")
if TYPE_OF_LABEL=="all":
    for label in labels_all:
        labels_in_this_training.append(label)

else:
    for label in labels_all:
        if TYPE_OF_LABEL in label:
            labels_in_this_training.append(label)

assert len(labels_in_this_training)>1



for label in labels_in_this_training:
    header.append(label)


def create_label_index_mapping_both_directions():
    for index,label in enumerate(labels_in_this_training):
        dict_all_labels_index[label]=index
        dict_all_index_labels[index]=label



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
                    full_text = get_spans_text_given_start_end_tokens(entry['start'], entry['end'], annotations)
                    return full_text



# we are adding negative examples also. i.e sentences/messages which had zero labels. we will still add it as 0,0,0
#e.g.,,"Your encrypted password was protected so your actual passwordwas not visible.",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

def get_negative_examples(dict_spantext_to_labels,plain_text_whole_email,empty_labels):
    if TYPE_OF_LABEL=="message":
        dict_spantext_to_labels[plain_text_whole_email]=empty_labels
    else:
        if TYPE_OF_LABEL=="sentence":
            email_split_sentences = seg.segment(plain_text_whole_email)
            for each_sent in email_split_sentences:
                dict_spantext_to_labels[each_sent] = empty_labels
        else:
            #to find negative examples for ner_span level
            # For GPEs and LOCs and ORGs, classify +/- 5 tokens
            #Run a name finder over the whole document to collect candidate spans (i.e. +/- N words of “things of a type we care about”)
            if TYPE_OF_LABEL == "words":
                email_split_sentences = seg.segment(plain_text_whole_email)
                for each_sent in email_split_sentences:
                    text1 = NER(each_sent)
                    words_this_sentence=[]
                    #collect all the words in that sentence first
                    # for ner_span in enumerate(text1):
                    #     words_this_sentence.append(str(ner_span[1]))
                    for ner_span in text1.ents:
                        #collect +/- N words of “things of a type we care about”)
                        if ner_span.label_ in ["GPE","LOC","ORG"]:
                            #there are some weird edge cases where the tokenizer's token doesnt match with that of NER> fucking maa ka lavda
                            #if ner_span.text in text1.doc:
                            #split the foundNER span, get its first token, and
                            #split_ner_text=NER(ner_span.text )
                            #if ner_span.text in words_this_sentence:
                                #this_word_index_in_sent = words_this_sentence.index(ner_span.text)
                            start_index=ner_span.start-SPAN_LENGTH_NEGATIVE_EXAMPLE_SPAN_WORDS
                            end_index=ner_span.end+SPAN_LENGTH_NEGATIVE_EXAMPLE_SPAN_WORDS
                            if (start_index < 0):
                                start_index = 0
                            if (end_index > len(text1)):
                                end_index = len(text1)
                            span_tokens=str(text1[start_index:end_index])
                            #this_word_index_in_sent-SPAN_LENGTH_NEGATIVE_EXAMPLE_SPAN_WORDS

                            # end_index = this_word_index_in_sent + SPAN_LENGTH_NEGATIVE_EXAMPLE_SPAN_WORDS
                            #
                            # #get those n tokens before and after this token
                            # span_tokens=words_this_sentence[start_index:end_index]
                            #add that as a negative example and move on with life
                            dict_spantext_to_labels[span_tokens] = empty_labels
                            # else:
                            #     print("test")






    return


#go through each of the spans, find each of the labels in the spans, and check if that label is one of the labels we are
#searching for. if yes, add it to a dictionary which maps text->label
def get_text_for_label_from_all_spans(Lines):
    for index, line in enumerate(Lines):

        annotations = json.loads(line)
        # for adding negative examples as text, 0,0,0
        empty_labels = [""] * len(labels_in_this_training)
        plain_text_whole_email=annotations['text'].replace("\n","")
        get_negative_examples(((dict_spantext_to_labels)),plain_text_whole_email,empty_labels)

        #existence of label spans means there was atleast one label in this email that was annotated
        if "spans" in annotations:
            for entry in annotations["spans"]:
                label = entry["label"]
                if label in labels_in_this_training:
                    if "message" in label and TYPE_OF_LABEL=="message":
                        #explicitly picking same text of the email because we want the empty entry in dict_spantext_to_labels to be replaced by
                        text=plain_text_whole_email
                        if text is not None :
                            if text in dict_spantext_to_labels:
                                old_value = dict_spantext_to_labels[text]
                                idx=dict_all_labels_index[label]
                                old_value[idx]=label
                                dict_spantext_to_labels[text] = old_value
                    else:
                        text = get_spans_text_given_start_end_tokens(entry['token_start'], entry['token_end'],
                                                                     annotations)
                        if text is not None:
                            text=text.replace("\n","")
                            if text in dict_spantext_to_labels:
                                old_value = dict_spantext_to_labels[text]
                                if label not in old_value:
                                    old_value.append(label)
                                    dict_spantext_to_labels[text] = old_value
                            else:
                                dict_spantext_to_labels[text] = [label]


#given the start and end of a span return the collection of the tokens corresponding to this in string format
def get_spans_text_given_start_end_tokens(token_start_of_span, token_end_of_span, annotations):
    starts_ends_tokens = []

    for index, token in enumerate(annotations['tokens']):
        if index>=token_start_of_span and index<=token_end_of_span:
            starts_ends_tokens.append(token['text'])


        # if (token['start']>=token_start_of_span and token['end']<=token_end_of_span):
        #     starts_ends_tokens.append(token['text'])
        # if (token['start'] >= token_start_of_span and token['end'] > token_end_of_span):
    assert len(starts_ends_tokens) >0
    return " ".join(starts_ends_tokens)


#creating data for this tutorial: https://cqcl.github.io/lambeq/tutorials/trainer_classical.html
def create_training_data_lambeq():
    with open(OUTPUT_FILE_NAME, 'w') as out:
        out.write("\n")

    with open("data/query_file.jsonl", 'r') as in_file:
        Lines = in_file.readlines()
        create_label_index_mapping_both_directions()
        # go through each of the annotated data point, extract text and its label into a dictionary dict_spantext_to_labels
        get_text_for_label_from_all_spans(Lines)
        # once the dict_spantext_to_labels is filled with a mapping from spantext to corresponding labels, write it out in a one hot vector
        with open(OUTPUT_FILE_NAME, 'a') as out:
            counter=0
            for datapoint, labels in dict_spantext_to_labels.items():
                outline=""
                if "sentence_tone_urgent" in labels:
                    outline="1\t"+datapoint
                # else:
                #     outline="0\t"+datapoint
                out.write(f"{outline}\n")
                counter+=1
                if counter % 10 == 0:
                    out.flush()





create_training_data_lambeq()
import sys
sys.exit()

