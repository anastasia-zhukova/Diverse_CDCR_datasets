import xml.etree.ElementTree as ET
import os
import json
import sys
import string
import spacy
import copy
import re
import pandas as pd
import numpy as np
from insert_whitespace import append_text
from nltk import Tree
from tqdm import tqdm
import warnings
from setup import *
from logger import LOGGER

path_sample = os.path.join(DATA_PATH, "_sample_doc.json")  # ->root/data/original/_sample_doc.json
WECENG_PARSING_FOLDER = os.path.join(os.getcwd())
OUT_PATH = os.path.join(WECENG_PARSING_FOLDER, OUTPUT_FOLDER_NAME)
source_path = os.path.join(WECENG_PARSING_FOLDER, WECENG_FOLDER_NAME)
result_path = os.path.join(OUT_PATH, 'test_parsing')

CONTEXT_COMB_SPAN = 5

# opens and loads the newsplease-format out of the json file: _sample_doc.json
with open(path_sample, "r") as file:
    newsplease_format = json.load(file)

def conv_files(path):
    """
        Converts the given dataset for a specified language into the desired format.
        :param paths: The paths desired to process (intra- & cross_intra annotations)
        :param result_path: the test_parsing folder for that language
        :param out_path: the output folder for that language
        :param language: the language to process
        :param nlp: the spacy model that fits the desired language
    """
    with open(os.path.join(path, 'WEC-Eng.json')) as f:
        wec_data = json.load(f)
        df = pd.DataFrame.from_dict(wec_data, orient="columns")

    df = df.groupby([DOC_ID])

    prev_doc_id = None
    doc_tokens_list = []
    for i, row in tqdm(df.iterrows()):
        print(row[DOC_ID])
        print(row[MENTION_CONTEXT])
        if row[DOC_ID] == prev_doc_id:
            last_few_tokens = doc_tokens_list[-CONTEXT_COMB_SPAN:]
            appearance = [(i, i+len(last_few_tokens)) for i in range(len(row[MENTION_CONTEXT])) if row[MENTION_CONTEXT][i:i+len(last_few_tokens)] == last_few_tokens]
            print(appearance)

        else:
            doc_tokens_list = row[MENTION_CONTEXT]

        prev_doc_id = row[DOC_ID]

if __name__ == '__main__':
    LOGGER.info(f"Processing WEC-Eng {source_path[-34:].split('_')[2]}.")
    conv_files(source_path)
