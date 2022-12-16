from transformers import BertTokenizer, BertModel, BertConfig
import os
import pysbd

TYPE_OF_LABEL="sentence" #["message","words","signature","sentence","all"]
#is it training or testing. testing means will load a saved model and test
TYPE_OF_RUN="train" # ["train","test"]

OUTPUT_FILE_NAME= "data/training_data.csv"
header=["id","text"]
labels_all=["message_contact_person_asking", "message_contact_person_org", "message_org", "sentence_intent_attachment", "sentence_intent_click", "sentence_intent_intro", "sentence_intent_money", "sentence_intent_phonecall", "sentence_intent_products", "sentence_intent_recruiting", "sentence_intent_scheduling", "sentence_intent_service", "sentence_intent_unsubscribe", "sentence_org_used_by_employer", "sentence_passwd", "sentence_tone_polite", "sentence_tone_urgent", "sentence_url_no_name", "sentence_url_third_party", "signature", "signature_email", "signature_fullname", "signature_jobtitle", "signature_org", "signature_phone", "signature_signoff", "signature_url", "signaure_address", "signaure_handle", "words_reciever_organization", "words_sender_location", "words_sender_organization"]

OUTPUT_DIRECTORY="./output/"
MAX_LEN = 500
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
TESTING_BATCH_SIZE=1
EPOCHS = 100
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

TESTING_FILE_PATH="./data/testing_data.csv"

SAVED_MODEL_NAME="best_model_for_"+TYPE_OF_LABEL+"_"+"level_trained"+".pth"
SAVED_MODEL_PATH=os.path.join("./output/",SAVED_MODEL_NAME)

BEST_MODEL_MESSAGE_LEVEL="best_model_for_message_level_trained.pth"

test_params = {'batch_size': TESTING_BATCH_SIZE,
               'shuffle': False,
               'num_workers': 0
               }