import io
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
GVC_PARSING_FOLDER = os.path.join(os.getcwd())
OUT_PATH = os.path.join(GVC_PARSING_FOLDER, OUTPUT_FOLDER_NAME)
source_path = os.path.join(GVC_PARSING_FOLDER, GVC_FOLDER_NAME)
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

    conll_df = pd.DataFrame(columns=[TOPIC_SUBTOPIC, DOC_ID, SENT_ID, TOKEN_ID, TOKEN, REFERENCE])

    LOGGER.info("Reading gold.conll...")
    with open(os.path.join(path, 'gold.conll'), encoding="utf-8") as f:
        conll_str = f.read()
        print(conll_str)
        conll_lines = conll_str.split("\n")
        for conll_line in tqdm(conll_lines):
            if "#begin document" in conll_line or "#end document" in conll_line:
                continue
            if conll_line.split("\t")[0].split(".")[1] == "DCT":
                continue
            conll_df = pd.concat([conll_df, pd.DataFrame({
                TOPIC_SUBTOPIC: conll_line.split("\t")[0].split(".")[0],
                SENT_ID: int(conll_line.split("\t")[0].split(".")[1][1:]),
                TOKEN_ID: int(conll_line.split("\t")[0].split(".")[2]),
                TOKEN: conll_line.split("\t")[1],
                REFERENCE: conll_line.split("\t")[3]
            }, index=[0])])

    print(conll_df.head(10))

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
    LOGGER.info(f"Processing GVC {source_path[-34:].split('_')[2]}.")
    conv_files(source_path)
