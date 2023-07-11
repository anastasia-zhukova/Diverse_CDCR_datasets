import os
import json
import sys
import shortuuid
import string
import spacy
import copy
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import requests
import wikipedia
from bs4 import BeautifulSoup

from setup import *
from utils import *
from logger import LOGGER

WECENG_PARSING_FOLDER = os.path.join(os.getcwd())
OUT_PATH = os.path.join(WECENG_PARSING_FOLDER, OUTPUT_FOLDER_NAME)
source_path = os.path.join(WECENG_PARSING_FOLDER, WECENG_FOLDER_NAME)


def conv_files():
    with open(os.path.join(source_path, "topics", "topics.json"), "r") as file:
        topics_list = json.load(file)

    conll_df = pd.DataFrame()
    event_mentions = []

    for split, file_name in zip(["val", "test", "train"],
                                ["Dev_Event_gold_mentions_validated.json",
                                "Test_Event_gold_mentions_validated.json",
                                "Train_Event_gold_mentions.json"]):
        with open(os.path.join(source_path, file_name)) as f:
            mentions_list_init = json.load(f)

        event_mentions_local = []
        conll_df_local = pd.DataFrame()

        mentions_df = pd.DataFrame.from_dict(mentions_list_init, orient="columns")
        mentions_df.drop(columns=[MENTION_CONTEXT], inplace=True)
        mentions_df.set_index("mention_index", inplace=True)

        # get information about topics
        topics_path = os.path.join(source_path, "topics", f"topics_{split}.csv")

        if os.path.exists(topics_path):
            doc_topics_df = pd.read_csv(topics_path, index_col=[0])
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

        # create ids
        doc_topics_df.loc[:, DOC_ID] = [shortuuid.uuid(v) for v in doc_topics_df[DOC].values]
        doc_topics_df.loc[:, SUBTOPIC_ID] = [shortuuid.uuid(v) for v in doc_topics_df[SUBTOPIC].values]
        topic_dict = {t: str(i) for i, t in enumerate(sorted(topics_list))}
        doc_topics_df.loc[:, TOPIC_ID] = [topic_dict[v] for v in doc_topics_df[TOPIC].values]
        doc_topics_df.drop_duplicates(subset=[DOC], inplace=True)
        doc_topics_df.set_index(DOC, inplace=True)

        # merge contexts
        context_dict_global = {}
        chain_dict = {}
        for mention_id, mention in tqdm(enumerate(mentions_list_init), total=len(mentions_list_init)):
            if mention[DOC_ID] not in context_dict_global:
                context_dict_global[mention[DOC_ID]] = {}

            context_dict_global[mention[DOC_ID]][mention["mention_index"]] = mention[MENTION_CONTEXT]

            if mention[COREF_CHAIN] not in chain_dict:
                chain_dict[mention[COREF_CHAIN]] = []
            chain_dict[mention[COREF_CHAIN]].append(mention[MENTION_ID])

        mentions_df.loc[:, SENT_ID] = 0

        LOGGER.info(f"Creating conll for the documents based on the mentions' context...")
        for doc_name, context_dict in tqdm(context_dict_global.items(), total=len(context_dict_global)):
            # overlap_df = pd.DataFrame(np.zeros((len(context_dict), len(context_dict))),
            #                                    index=list(context_dict), columns=list(context_dict))
            overlap_dict = {}
            used_context = set()
            for i, (m_id_1, context_1) in enumerate(context_dict.items()):
                if m_id_1 in used_context:
                    continue

                for j, (m_id_2, context_2) in enumerate(context_dict.items()):
                    if j >= i:
                        continue

                    # same wiki paragraph
                    if context_1 == context_2:
                        if m_id_1 not in overlap_dict:
                            overlap_dict[m_id_1] = []

                        overlap_dict[m_id_1].append(m_id_2)
                        used_context.add(m_id_2)
                        used_context.add(m_id_1)

            # add mentions that had unique paragraphs
            overlap_dict.update({k: [] for k in set(context_dict) - used_context})
            # sort by mention_index to maintain reproducibility of the same results
            overlap_dict = {k:v for k,v in sorted(overlap_dict.items(), reverse=False, key=lambda x: x[0])}

            # form one document based on the merged context/paragraphs of the mentions
            for sent_id, (mention_key, mention_same_doc) in enumerate(overlap_dict.items()):
                all_mentions = {mention_key}.union(set(mention_same_doc))
                for m in all_mentions:
                    mentions_df.loc[m, SENT_ID] = int(sent_id)

                topic_id = doc_topics_df.loc[doc_name, TOPIC_ID]
                subtopic_id = doc_topics_df.loc[doc_name, SUBTOPIC_ID]
                doc_id = doc_topics_df.loc[doc_name, DOC_ID]

                for token_id, token in enumerate(context_dict[mention_key]):
                    conll_df_local = pd.concat([conll_df_local, pd.DataFrame({
                        TOPIC_SUBTOPIC_DOC: f"{topic_id}/{subtopic_id}/{doc_id}",
                        DOC_ID: doc_id,
                        SENT_ID: int(sent_id),
                        TOKEN_ID: int(token_id),
                        TOKEN: token,
                        REFERENCE: "-"
                    }, index=[f'{doc_id}/{sent_id}/{token_id}'])])

                    # TODO remove after no longer needed for the similar datasets with context only
                        # overlap_df.loc[m_id_1, m_id_2] = 1
                    # matcher = difflib.SequenceMatcher(None, context_1, context_2)
                    # matches = matcher.get_matching_blocks()
                    # for match in matches:
                    #     apos, bpos, size = match
                    #     # if bpos == 0 and size > 5:
                    #     if bpos == 0 and apos + size == len(context_1):
                    #         overlap_df.loc[m_id_1, m_id_2] = size
                    #         print(context_1[apos:apos + size], apos, bpos, size)
                    #         # break
                    #     if apos == 0 and bpos + size == len(context_2):
                    #     # if apos == 0 and size > 5:
                    #         overlap_df.loc[m_id_1, m_id_2] = size
                    #         print(context_2[apos:apos + size], apos, bpos, size)
                            # break

        LOGGER.info(f"Creating mentions with missing attributes...")
        for mention_orig in mentions_list_init:
            mention_type = "OCCURRENCE"
            doc_id = doc_topics_df.loc[mention_orig[DOC_ID], DOC_ID]
            subtopic_id = doc_topics_df.loc[mention_orig[DOC_ID], SUBTOPIC_ID]
            topic_id = doc_topics_df.loc[mention_orig[DOC_ID], TOPIC_ID]
            tokens_text = [mention_orig[MENTION_CONTEXT][i] for i in mention_orig[TOKENS_NUMBER]]
            tokens_number = mention_orig[TOKENS_NUMBER]
            mention_head_id = tokens_number[tokens_text.index(mention_orig[MENTION_HEAD])]
            sent_id = int(mentions_df.loc[mention_orig["mention_index"], SENT_ID])
            context_min_id = 0 if tokens_number[0] - CONTEXT_RANGE < 0 else tokens_number[0] - CONTEXT_RANGE
            context_max_id = min(tokens_number[0] + CONTEXT_RANGE, len(mention_orig[MENTION_CONTEXT]))

            mention = {COREF_CHAIN: mention_orig[COREF_CHAIN],
                       MENTION_NER: mention_orig[MENTION_NER],
                       MENTION_HEAD_POS: mention_orig[MENTION_HEAD_POS],
                       MENTION_HEAD_LEMMA: mention_orig[MENTION_HEAD_LEMMA],
                       MENTION_HEAD: mention_orig[MENTION_HEAD],
                       MENTION_HEAD_ID: mention_head_id,
                       DOC_ID: doc_id,
                       DOC: mention_orig[DOC_ID],
                       IS_CONTINIOUS: True if mention_orig[TOKENS_NUMBER] == list(range(mention_orig[TOKENS_NUMBER] [0],
                                                                                        mention_orig[TOKENS_NUMBER] [-1] + 1))
                                        else False,
                       IS_SINGLETON: len(chain_dict[mention_orig[COREF_CHAIN]]) == 1,
                       MENTION_ID: mention_orig[MENTION_ID],
                       MENTION_TYPE: mention_type[:3],
                       MENTION_FULL_TYPE: mention_type,
                       SCORE: -1.0,
                       SENT_ID: sent_id,
                       MENTION_CONTEXT: mention_orig[MENTION_CONTEXT][context_min_id: context_max_id],
                       TOKENS_NUMBER: tokens_number,
                       TOKENS_STR: mention_orig[TOKENS_STR],
                       TOKENS_TEXT: tokens_text,
                       TOPIC_ID: topic_id,
                       TOPIC: doc_topics_df.loc[mention_orig[DOC_ID], [TOPIC]],
                       SUBTOPIC_ID: subtopic_id,
                       SUBTOPIC: doc_topics_df.loc[mention_orig[DOC_ID], [SUBTOPIC]],
                       COREF_TYPE: IDENTITY,
                       DESCRIPTION: mention_orig["coref_link"],
                       CONLL_DOC_KEY: f'{topic_id}/{subtopic_id}/{doc_id}',
                       }
            # sanity check
            tokens_from_conll = list(conll_df_local.loc[f'{doc_id}/{sent_id}/{tokens_number[0]}':
                                                        f'{doc_id}/{sent_id}/{tokens_number[-1]}', TOKEN])
            try:
                assert tokens_text == tokens_from_conll
                event_mentions_local.append(mention)
            except AssertionError:
                LOGGER.warning(f'Mention {mention_orig[TOKENS_STR]} from a document \"{mention_orig[DOC_ID]}\" was not '
                               f'correctly mapped to conll and will be skipped. ')

        save_path = os.path.join(OUT_PATH, split)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        with open(os.path.join(save_path, MENTIONS_EVENTS_JSON), "w") as file:
            json.dump(event_mentions_local, file)

        with open(os.path.join(save_path, MENTIONS_ENTITIES_JSON), "w") as file:
            json.dump([], file)

        conll_df_local_labeled = make_save_conll(conll_df_local, event_mentions_local, save_path)
        conll_df = pd.concat([conll_df, conll_df_local_labeled])
        event_mentions.extend(event_mentions_local)

    with open(os.path.join(OUT_PATH, MENTIONS_EVENTS_JSON), "w") as file:
        json.dump(event_mentions, file)

    with open(os.path.join(OUT_PATH, MENTIONS_ENTITIES_JSON), "w") as file:
        json.dump([], file)

    df_all_mentions = pd.DataFrame()
    for mention in event_mentions:
        df_all_mentions = pd.concat([df_all_mentions, pd.DataFrame({
            attr: str(value) if type(value) == list else value for attr, value in mention.items()
        }, index=[mention[MENTION_ID]])], axis=0)
    df_all_mentions.to_csv(os.path.join(OUT_PATH, MENTIONS_ALL_CSV))

    make_save_conll(conll_df, df_all_mentions, OUT_PATH, assign_reference_labels=False)

    LOGGER.info(
        f'\nNumber of unique mentions: {len(event_mentions)} '
        f'\nNumber of unique chains: {len(set(df_all_mentions[COREF_CHAIN].values))} ')
    LOGGER.info(f'Parsing of WEC-Eng is done!')


if __name__ == '__main__':
    LOGGER.info(f"Processing WEC-Eng {source_path[-34:].split('_')[2]}.")
    conv_files()
