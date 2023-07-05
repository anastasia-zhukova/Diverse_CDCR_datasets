import xml.etree.ElementTree as ET
import os
import json
import sys
sys.path.append('..')
import string
import spacy
import copy
import re
import pandas as pd
import numpy as np
from utils import *
from nltk import Tree
from tqdm import tqdm
import warnings

from setup import *
# import wikipedia
from logger import LOGGER

path_sample = os.path.join(DATA_PATH, "_sample_doc.json")  # ->root/data/original/_sample_doc.json
WECENG_PARSING_FOLDER = os.path.join(os.getcwd())
OUT_PATH = os.path.join(WECENG_PARSING_FOLDER, OUTPUT_FOLDER_NAME)
source_path = os.path.join(WECENG_PARSING_FOLDER, WECENG_FOLDER_NAME)
result_path = os.path.join(OUT_PATH, 'test_parsing')

CONTEXT_COMB_SPAN = 5

# def get_full_wiki_article(article_title, sentences_length=10):
#     try:
#         if wikipedia.page(article_title).title != article_title:
#             return article_title
#         else:
#             return wikipedia.summary(article_title, sentences=sentences_length)
#     except:
#         return article_title

# # opens and loads the newsplease-format out of the json file: _sample_doc.json
# with open(path_sample, "r") as file:
#     newsplease_format = json.load(file)

def assign_reference(row):
    if row['token_id'] in row[TOKENS_NUMBER]:
        reference = str(row['coref_chain'])
        if min(row[TOKENS_NUMBER]) == row['token_id']:
            reference = '(' + reference
        if max(row[TOKENS_NUMBER]) == row['token_id']:
            reference = reference + ')'
    else:
        reference = '-'
    return reference

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

    df = df.sort_values([DOC_ID])
    df = df.rename(columns={DOC_ID:DOC})
                            #, 'tokens_number':'tokens_numbers'})
    df[DOC_ID] = df.doc.str.replace(
        '[^a-zA-Z\t\s0-9]','').str.replace(
        '\s|\t','_').str.lower()
    coref_val_counts = df.coref_chain.value_counts()
    df[IS_SINGLETON] = df.coref_chain.isin(coref_val_counts[coref_val_counts==1].index)
    df[IS_CONTINIOUS] = df[TOKENS_NUMBER].apply(lambda x: all(x[i] == x[i-1] + 1 for i in range(1, len(x))))
    df[TOKENS_TEXT] = df.tokens_str.str.split('[\s\t\n]+')
    df[SENT_ID] = df.index
    df[COREF_TYPE] = IDENTITY
    df[MENTION_HEAD_ID] = df.groupby(MENTION_HEAD).cumcount()
    df[CONLL_DOC_KEY] = '-/-/'+df[DOC_ID]

    df[SENT_ID] = df.index
    # df[COREF_TYPE] = IDENTITY
    # leaving topic field empty since WEC dataset doesn't have topics
    # TODO: Can we extract Topics from wikipedia urls?
    df[TOPIC] = '-'
    df[TOPIC_ID] = 0
    df[TOKENS_NUMBER]= df[TOKENS_NUMBER].astype(str)

    conll_df = df[[DOC_ID, CONLL_DOC_KEY, MENTION_CONTEXT, 
                   TOKENS_NUMBER, COREF_CHAIN, SENT_ID]
                  ].explode('mention_context').reset_index().rename(
        columns={'index' : 'df_index',
                 MENTION_CONTEXT:TOKEN,
                 CONLL_DOC_KEY:TOPIC_SUBTOPIC_DOC})
    conll_df['token_id'] = conll_df.groupby('df_index').cumcount().astype(str)
    conll_df['reference'] = conll_df.apply(lambda x: assign_reference(x), axis=1)
    conll_df = conll_df.drop(['df_index', 'coref_chain'], axis=1)
    all_mentions_df = df.copy()
    
    # make_save_conll(conll_df, df, OUT_PATH)
    
    
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    
    make_save_conll(conll_df, df, OUT_PATH)
    all_mentions_df.to_csv(OUT_PATH + '/' + MENTIONS_ALL_CSV)
    conll_df.to_csv(OUT_PATH + '/' + CONLL_CSV)
    with open(os.path.join(OUT_PATH, MENTIONS_EVENTS_JSON), "w") as file:
        json.dump(all_mentions_df.to_dict('records'), file)
    
    with open(os.path.join(OUT_PATH, MENTIONS_ENTITIES_JSON), "w") as file:
        json.dump(pd.DataFrame(columns=all_mentions_df.columns).to_dict(), file)

    outputdoc_str= create_conll_string(conll_df)

    with open(os.path.join(OUT_PATH, 'wec.conll'), "w", encoding='utf-8') as file:
        file.write(outputdoc_str)

if __name__ == '__main__':
    LOGGER.info(f"Processing WEC-Eng {source_path[-34:].split('_')[2]}.")
    conv_files(source_path)
