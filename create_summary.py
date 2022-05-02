from setup import *
from logger import LOGGER

import pandas as pd
import os
import json
import nltk
import string
import numpy as np
from typing import Dict, List
from nltk.corpus import stopwords
from tqdm import tqdm


DIRECTORIES_TO_SUMMARIZE = [NEWSWCL50, ECB_PLUS]

nltk.download('stopwords')


def phrasing_diversity_calc(mentions: List[Dict]) -> float:
    """
    Calculates a metric that represents phrasing complexity of an entity, i.e., how various is the wording of the
    phrases referring to this entity. For more details see LREC paper
    Returns:
        Internal headword-based structure, value of the phrasing complexity

    a format of members from the json for vis:
    [
        {
            "sentence": 0,
            "mention_id": ll0009,
            "doc_id: 0,
            "tokens_number": [6, 7, 8, 9, 10, 11, 12, 13, 14],
            "tokens_text": ["Trump", "in", "broad", "quest", "on", "Russia", "ties", "and", "obstruction"],
            "tokens_str": "Trump in broad quest on Russia ties and obstruction",
            "mention_head_id": 6,
            "mention_head": "Trump"
        }
    ]
    """
    headwords_phrase_tree = {}
    for mention in mentions:
        mention_wo_stopwords = [w for w in mention[TOKENS_TEXT]
                                 if w not in string.punctuation and w not in stopwords.words("english")]

        if not len(mention_wo_stopwords):
            continue

        mention_wordset = frozenset(mention_wo_stopwords)

        if mention[MENTION_HEAD] not in headwords_phrase_tree:
            headwords_phrase_tree[mention[MENTION_HEAD]] = {"set": {mention_wordset},
                                                                 "list": [mention_wo_stopwords]}
        else:
            # set ensures unique phrases
            headwords_phrase_tree[mention[MENTION_HEAD]]["set"].add(mention_wordset)
            # list keeps actual number of phrase occurence
            headwords_phrase_tree[mention[MENTION_HEAD]]["list"].append(mention_wo_stopwords)

    sets = []
    fractions = []
    for head, head_properties in headwords_phrase_tree.items():
        fractions.append(len(head_properties["set"]) / (len(head_properties["list"])))
        sets.append(len(head_properties["set"]))

    score = np.sum(np.array(fractions)) * np.sum(np.array(sets)) / len(mentions) if len(mentions) > 1 else 1
    return float(format(score, '.3f'))


if __name__ == '__main__':

    summary_df = pd.DataFrame()
    chain_df_all = pd.DataFrame()
    all_mentions_list = []

    # read annotated mentions and corresponding texts
    for dataset_folder in DIRECTORIES_TO_SUMMARIZE:
        mentions_df = pd.DataFrame()

        LOGGER.info(f'Reading files with mentions for {dataset_folder} dataset..')
        for mention_type, file_name in zip([EVENT, ENTITY], [MENTIONS_EVENTS_JSON, MENTIONS_ENTITIES_JSON]):
            full_filename = os.path.join(os.getcwd(), dataset_folder, OUTPUT_FOLDER_NAME, file_name)
            with open(full_filename, encoding='utf-8', mode='r') as file:
                mentions_read_list = json.load(file)

            # remove list attributes that can't fit a dataframe
            mentions_read_list_filt = [v for v in mentions_read_list if type(v) != list]
            df_tmp = pd.DataFrame(mentions_read_list_filt, index=list(range(len(mentions_read_list_filt))))
            df_tmp[TYPE] = [mention_type] * len(df_tmp)
            # kick out "prep" from the name
            df_tmp[DATASET_NAME] = [dataset_folder.split("-")[0]] * len(df_tmp)
            mentions_df = pd.concat([mentions_df, df_tmp], ignore_index=True, axis = 0)
            all_mentions_list.extend(mentions_read_list)

        # read texts (conll format)
        df_conll = pd.read_csv(os.path.join(os.getcwd(), dataset_folder, OUTPUT_FOLDER_NAME, CONLL_CSV), index_col=[0])

        # calculate statistics about the chains
        chain_df = mentions_df[[COREF_CHAIN, DATASET_NAME, DOC_ID]].groupby([COREF_CHAIN,
                         DATASET_NAME]).count().rename(columns={DOC_ID: MENTIONS}).reset_index().set_index(COREF_CHAIN)
        chain_df[PHRASING_DIVERSITY] = [0] * len(chain_df)
        chain_df[UNIQUE_LEMMAS] = [0] * len(chain_df)
        for chain in set(mentions_df[COREF_CHAIN].values):
            chain_mentions = [v for v in all_mentions_list if v[COREF_CHAIN] == chain]
            chain_df.loc[chain, PHRASING_DIVERSITY] = phrasing_diversity_calc(chain_mentions)
            chain_df.loc[chain, UNIQUE_LEMMAS] = len(
                set([v.lower() for v in mentions_df[mentions_df[COREF_CHAIN] == chain][MENTION_HEAD_LEMMA].values]))

        # form dataset as full and split into topics to process
        # full dataset
        process_list = [(dataset, "", None, dataset_df) for dataset, dataset_df in mentions_df.groupby([DATASET_NAME])]
        # per topic in the dataset
        process_list.extend([(dataset, topic, topic_id, group_df)
                             for (dataset, topic, topic_id), group_df in mentions_df.groupby([DATASET_NAME, TOPIC, TOPIC_ID])])

        # collect statistics about the dataset
        for dataset, topic, topic_id, group_df in tqdm(process_list):
            coref_chains = list(set(group_df[COREF_CHAIN].values))

            # general statistics
            summary_dict = {
                DATASET_NAME: dataset,
                TOPIC: topic,
                ARTICLES: len(set(group_df[DOC_ID].values)),
                TOKENS: len(df_conll[df_conll[TOPIC_SUBTOPIC].str.startswith(str(topic_id))]) if topic_id is not None else len(df_conll),
                COREF_CHAIN: len(coref_chains),
                MENTIONS: len(group_df),
                f'{EVENT}_{MENTIONS}': len(group_df[group_df[TYPE] == EVENT]),
                f'{ENTITY}_{MENTIONS}': len(group_df[group_df[TYPE] == ENTITY]),
                SINGLETONS: len(group_df[group_df[IS_SINGLETON]]),
                AVERAGE_SIZE: float(format(np.mean(chain_df.loc[coref_chains][MENTIONS].values), '.3f')),
            }

            # various for, of lexical diversity that depend on the presence/absence of singletons
            for suff, filt_criteria in zip([ALL, WO_SINGL], [0, 1]):
                selected_chains_df = chain_df[(chain_df[MENTIONS] > filt_criteria) & (chain_df.index.isin(coref_chains))]
                summary_dict[PHRASING_DIVERSITY + WEIGHTED + suff] = float(format(sum([row[PHRASING_DIVERSITY] * row[MENTIONS]
                                                                  for index, row in selected_chains_df.iterrows()]) / \
                                                                     sum(selected_chains_df[MENTIONS].values), '.3f'))
                summary_dict[PHRASING_DIVERSITY + MEAN + suff] = float(format(np.mean(selected_chains_df[PHRASING_DIVERSITY].values), '.3f'))
                summary_dict[UNIQUE_LEMMAS + suff] = float(format(np.mean(selected_chains_df[UNIQUE_LEMMAS].values), '.3f'))

            summary_df = pd.concat([summary_df, pd.DataFrame(summary_dict, index=[f'{dataset}\\{topic}'])], axis=0)
        chain_df_all = pd.concat([chain_df_all, chain_df], axis=0)

        #outpput the chains statistics
    chain_df_all.reset_index().to_csv(os.path.join(os.getcwd(), SUMMARY_FOLDER, SUMMARY_CHAINS_CSV))
    summary_df.to_csv(os.path.join(os.getcwd(), SUMMARY_FOLDER, SUMMARY_TOPICS_CSV))

    LOGGER.info(f'Summary computation over {len(DIRECTORIES_TO_SUMMARIZE)} datasets ({DIRECTORIES_TO_SUMMARIZE}) '
                f'is completed.')
