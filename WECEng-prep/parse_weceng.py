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
from nltk import Tree
from tqdm import tqdm
import warnings
import requests
from bs4 import BeautifulSoup

from setup import *
from utils import *
import wikipedia
from logger import LOGGER

WECENG_PARSING_FOLDER = os.path.join(os.getcwd())
OUT_PATH = os.path.join(WECENG_PARSING_FOLDER, OUTPUT_FOLDER_NAME)
source_path = os.path.join(WECENG_PARSING_FOLDER, WECENG_FOLDER_NAME)


def get_full_wiki_article(article_title, sentences_length=10):
    try:
        if wikipedia.page(article_title).title != article_title:
            return article_title
        else:
            return wikipedia.summary(article_title, sentences=sentences_length)
    except:
        return article_title


#
# def assign_reference(row):
#     if row['token_id'] in row[TOKENS_NUMBER]:
#         reference = str(row['coref_chain'])
#         if min(row[TOKENS_NUMBER]) == row['token_id']:
#             reference = '(' + reference
#         if max(row[TOKENS_NUMBER]) == row['token_id']:
#             reference = reference + ')'
#     else:
#         reference = '-'
#     return reference


def conv_files():
    with open("topics.json", "r") as file:
        topics_list = json.load(file)

    for split, file_name in zip(["val", "test", "train"],
                                ["Dev_Event_gold_mentions_validated.json",
                                "Test_Event_gold_mentions_validated.json",
                                "Train_Event_gold_mentions.json"]):
        with open(os.path.join(source_path, file_name)) as f:
            mentions_list_init = json.load(f)
        mentions_df = pd.DataFrame.from_dict(mentions_list_init, orient="columns")
        mentions_df.drop(columns=[MENTION_CONTEXT], inplace=True)

        # get information about topics
        topics_path = os.path.join(source_path, "topics", f"topics_{split}.csv")
        if os.path.exists(topics_path):
            doc_topics_df = pd.read_csv(topics_path)
        else:
            doc_topics_df = pd.DataFrame()
            for mention_id, mention in tqdm(enumerate(mentions_list_init)):
                doc_topics_df = pd.concat([doc_topics_df, pd.DataFrame({
                    DOC: mention[DOC_ID],
                    SUBTOPIC: mention["coref_link"],
                    TOPIC: np.nan
                }, index=[mention_id])])

            doc_topics_df.drop_duplicates(inplace=True)

            for subtopic in tqdm(set(doc_topics_df[SUBTOPIC].values)):
                try:
                    wiki_page = wikipedia.page(subtopic)
                    subtopics = wiki_page.categories

                except wikipedia.exceptions.PageError or wikipedia.exceptions.DisambiguationError as e:
                    # page was not found via API OR page was ambiguius => parse manually
                    try:
                        req = requests.get(f"https://en.wikipedia.org/wiki/{subtopic.replace(' ', '_')}")
                        b = str(req.content, "utf-8")
                        soup = BeautifulSoup(b, 'html.parser')
                        category_div = soup.find('div', id='mw-normal-catlinks')
                        subtopics = [item.get_text() for item in category_div.contents[-1].find_all("a")]
                    except Exception as e:
                        LOGGER.error(e)
                        continue

                except Exception as e:
                    LOGGER.error(e)
                    continue

                topic_sim_dict = {t: 0 for t in topics_list}

                for topic in topics_list:
                    topic_tokens = set(topic.lower().split(" "))

                    for category in subtopics + [subtopic]:
                        category_tokens = set(category.lower().split(" "))
                        overlap = len(topic_tokens.intersection(category_tokens))
                        topic_sim_dict[topic] += overlap

                topic_sim_dict = {k:v for k,v in sorted(topic_sim_dict.items(), reverse=True, key=lambda x: x[1])}
                if list(topic_sim_dict.keys())[1] > 0:
                    selected_topic = list(topic_sim_dict.keys())[0]
                    subtopic_df = doc_topics_df[doc_topics_df[SUBTOPIC] == subtopic]
                    doc_topics_df.loc[subtopic_df.index, TOPIC] = selected_topic

            doc_topics_df.fillna("Misc", inplace=True)
            misc_found = doc_topics_df[doc_topics_df[TOPIC] == "Misc"]
            if len(misc_found):
                LOGGER.warning(f'Found {len(misc_found)} document assigned to Misc in the {split} split. Manual review of the topics needed!')

            doc_topics_df.to_csv(topics_path)

        a = 1
        # merge contexts
        context_dict = {}
        for mention_id, mention in tqdm(enumerate(mentions_list_init), total=len(mentions_list_init)):
            if mention[DOC_ID] not in context_dict:
                context_dict[mention[DOC_ID]] = {}

            context_dict[mention["doc_id"]][mention[MENTION_ID]] = mention[MENTION_CONTEXT]

        a = 1

        # coref_chain	269037
        # coref_link	"14th Academy Awards"
        # doc_id	"Richard Connell"
        # mention_context	[…]
        # mention_head	"Award"
        # mention_head_lemma	"Award"
        # mention_head_pos	"PROPN"
        # mention_id	45350
        # mention_index	203
        # mention_ner	"UNK"
        # mention_type	"7"
        # tokens_number	[…]
        # tokens_str	"Academy Award"

        # mentions_df_init = pd.read_csv(os.path.join(source_path, 'WEC-Eng.json'))

        # mention = {COREF_CHAIN: chain_id,
        #            MENTION_NER: m["mention_ner"],
        #            MENTION_HEAD_POS: m["mention_head_pos"],
        #            MENTION_HEAD_LEMMA: m["mention_head_lemma"],
        #            MENTION_HEAD: m["mention_head"],
        #            MENTION_HEAD_ID: m["mention_head_id"],
        #            DOC_ID: m[DOC_ID],
        #            DOC: m[DOC],
        #            IS_CONTINIOUS: True if token_numbers == list(
        #                range(token_numbers[0], token_numbers[-1] + 1))
        #            else False,
        #            IS_SINGLETON: len(chain_vals["mentions"]) == 1,
        #            MENTION_ID: mention_id,
        #            MENTION_TYPE: chain_id[:3],
        #            MENTION_FULL_TYPE: mention_type,
        #            SCORE: -1.0,
        #            SENT_ID: m["sent_id"],
        #            MENTION_CONTEXT: m[MENTION_CONTEXT],
        #            TOKENS_NUMBER: token_numbers,
        #            TOKENS_STR: m["text"],
        #            TOKENS_TEXT: m[TOKENS_TEXT],
        #            TOPIC_ID: m[TOPIC].split("_")[0],
        #            TOPIC: m[TOPIC],
        #            SUBTOPIC_ID: m[TOPIC_SUBTOPIC_DOC].split("/")[1],
        #            SUBTOPIC: m[SUBTOPIC],
        #            COREF_TYPE: IDENTITY,
        #            DESCRIPTION: chain_vals["descr"],
        #            LANGUAGE: m[LANGUAGE],
        #            CONLL_DOC_KEY: m[TOPIC_SUBTOPIC_DOC],
        #            }



    # df = df.sort_values([DOC_ID])
    # df = df.rename(columns={DOC_ID:DOC})
    #                         #, 'tokens_number':'tokens_numbers'})
    # df[DOC_ID] = df.doc.str.replace(
    #     '[^a-zA-Z\t\s0-9]','').str.replace(
    #     '\s|\t','_').str.lower()
    # coref_val_counts = df.coref_chain.value_counts()
    # df[IS_SINGLETON] = df.coref_chain.isin(coref_val_counts[coref_val_counts==1].index)
    # df[IS_CONTINIOUS] = df[TOKENS_NUMBER].apply(lambda x: all(x[i] == x[i-1] + 1 for i in range(1, len(x))))
    # df[TOKENS_TEXT] = df.tokens_str.str.split('[\s\t\n]+')
    # df[SENT_ID] = df.index
    # df[COREF_TYPE] = IDENTITY
    # df[MENTION_HEAD_ID] = df.groupby(MENTION_HEAD).cumcount()
    # df[CONLL_DOC_KEY] = '-/-/'+df[DOC_ID]
    #
    # df[SENT_ID] = df.index
    # # df[COREF_TYPE] = IDENTITY
    # # leaving topic field empty since WEC dataset doesn't have topics
    # # TODO: Can we extract Topics from wikipedia urls?
    # df[TOPIC] = '-'
    # df[TOPIC_ID] = 0
    # df[TOKENS_NUMBER]= df[TOKENS_NUMBER].astype(str)
    #
    # conll_df = df[[DOC_ID, CONLL_DOC_KEY, MENTION_CONTEXT,
    #                TOKENS_NUMBER, COREF_CHAIN, SENT_ID]
    #               ].explode('mention_context').reset_index().rename(
    #     columns={'index' : 'df_index',
    #              MENTION_CONTEXT:TOKEN,
    #              CONLL_DOC_KEY:TOPIC_SUBTOPIC_DOC})
    # conll_df['token_id'] = conll_df.groupby('df_index').cumcount().astype(str)
    # conll_df['reference'] = conll_df.apply(lambda x: assign_reference(x), axis=1)
    # conll_df = conll_df.drop(['df_index', 'coref_chain'], axis=1)
    # all_mentions_df = df.copy()
    
    # make_save_conll(conll_df, df, OUT_PATH)
    
    
    # if not os.path.exists(OUT_PATH):
    #     os.mkdir(OUT_PATH)
    #
    # make_save_conll(conll_df, df, OUT_PATH)
    # all_mentions_df.to_csv(OUT_PATH + '/' + MENTIONS_ALL_CSV)
    # # conll_df.to_csv(OUT_PATH + '/' + CONLL_CSV)
    # with open(os.path.join(OUT_PATH, MENTIONS_EVENTS_JSON), "w") as file:
    #     json.dump(all_mentions_df.to_dict('records'), file)
    #
    # with open(os.path.join(OUT_PATH, MENTIONS_ENTITIES_JSON), "w") as file:
    #     json.dump(pd.DataFrame(columns=all_mentions_df.columns).to_dict(), file)

    # outputdoc_str= create_conll_string(conll_df)

    # with open(os.path.join(OUT_PATH, 'wec.conll'), "w", encoding='utf-8') as file:
    #     file.write(outputdoc_str)

if __name__ == '__main__':
    LOGGER.info(f"Processing WEC-Eng {source_path[-34:].split('_')[2]}.")
    conv_files()
