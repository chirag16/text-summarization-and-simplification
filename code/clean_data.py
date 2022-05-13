
import argparse
import time
import matplotlib.pyplot as plt
import torch as T
import torch.nn.functional as F
import torch.nn as nn
from numpy import random
from rouge import Rouge
# from torch.distributions import Categorical
from pp import Vocab
from model import Encoder_Decoder_Model
from data_util import config
from train_util import *
import pandas as pd
import pp
import os
from GPUtil import showUtilization as gpu_usage
from beam_search import beam_search

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # setting trace back for more clear error
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set cuda device

random.seed(123)
T.manual_seed(123)
if T.cuda.is_available():
    T.cuda.manual_seed_all(123)

#############################
# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequences
############################

def clean_data():
    data_set = pd.read_csv(config.train_data_path, encoding="ISO-8859-1")  # [:400]
    # model = Model(opt)
    
    # plot_loss = []
    # start_time =  time.time()
    
    prev = 0
    batch_count = 0
    mle_total = 0
    simplification_total =0
    count=0

    all_articles = []
    all_summaries = []

    for next in range(config.batch_size, len(data_set), config.batch_size):
        print('batch: ', batch_count)
        batch_count+=1
        articles, summaries  = pp.get_next_batch(data_set, prev, next)
        all_articles += articles
        all_summaries += summaries
        
        prev = next
    
    new_dataset = {'article': all_articles, 'summary': all_summaries}
    new_dataset_df = pd.DataFrame.from_dict(new_dataset).dropna()

    print(new_dataset_df.head())
    print(new_dataset_df.describe())

    FILE_PATH = 'ssd-chunked'
    new_dataset_df.to_csv(FILE_PATH, sep='^', index=False)

    new_dataset_df = pd.read_csv(FILE_PATH, sep='^')
    print(new_dataset_df.describe())

if __name__ == "__main__":
    T.cuda.empty_cache()
    print("GPU utilization after clearing the catche: ")
    gpu_usage()
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--simplification', type=bool, default=True)
    opt = parser.parse_args()
    
    # print("CUDA_VISIBLE_DEVICES =", os.environ['CUDA_VISIBLE_DEVICES'])
    time_1 = time.time()
    loss = clean_data()
    time_2 =  time.time()

