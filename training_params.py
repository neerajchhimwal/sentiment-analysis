import torch
from transformers import BertTokenizer
import json

TRAIN_DATA = '/home/neeraj/Downloads/sentiment_data/df_train_with_preds.csv'
VALID_DATA = '/home/neeraj/Downloads/sentiment_data/df_valid_with_preds.csv'
LABEL_DICT_PATH = '/home/neeraj/Downloads/sentiment_data/label_map.json'
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'runs'
LOAD_CHECKPOINT = True
CHECKPOINT_PATH = 'checkpoints/checkpoint_last.pt'

WANDB_PROJECT_NAME = 'sentiment-analysis'

f = open(LABEL_DICT_PATH)
LABEL_DICT = json.load(f)

EPOCHS=10
MAX_LEN = 64
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
MAX_GRAD_NORM = 1.0

FULL_FINETUNING = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
TOKENIZER = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)



