import os
from datasets.dataset import IMDB, YELP_13,aapd,reuters
from models.AGmeory import AGMencoder

def ensureDirs(*dir_paths):
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

PRE_TRAINED_VECTOR_PATH = 'pretrained_Vectors'
PRE_TRAINED_BERT_PATH = 'pretrained_model'
SAVED_MODEL_PATH = 'saved_models'
DATASET_PATH = './corpus'

ensureDirs(PRE_TRAINED_BERT_PATH, PRE_TRAINED_VECTOR_PATH, SAVED_MODEL_PATH)

DATASET_PATH_MAP = {
    "yelp_13": os.path.join(DATASET_PATH, 'yelp_13'),
    'IMDB':os.path.join(DATASET_PATH, 'IMDB1'),
    'aapd':os.path.join(DATASET_PATH, 'aapd'),
    'reuters': os.path.join(DATASET_PATH, 'reuters')
}

LABLES = {
    "yelp_13": 5,
    'IMDB':10,
    'aapd':54,
'reuters': 90,
}

DATASET_MAP = {
    'IMDB':IMDB,
    "yelp_13": YELP_13,
    'aapd': aapd,
    'reuters': reuters

}

DECODER = {
    'e1': AGMencoder,
}
