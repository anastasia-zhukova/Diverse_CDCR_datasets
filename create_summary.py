from setup import *
from logger import LOGGER

import pandas as pd
import os
import json
import subprocess
import nltk
import string
import numpy as np
from typing import Dict, List
from nltk.corpus import stopwords
from tqdm import tqdm


DIRECTORIES_TO_SUMMARIZE = [NEWSWCL50, ECB_PLUS]

nltk.download('stopwords')

MUC = "_MUC"
B3 = "_B3"
CEAF_M = "_CEAF_M"
CEAF_E = "_CEAF_E"
BLANC = "_BLANC"
CONLL = "_CONLL"
P = "P"
R = "R"
F1 = "F1"
TRUE_LABEL, PRED_LABEL = "label_true", "label_pred"

with open(os.path.join(os.getcwd(), SAMPLE_MENTION_JSON), "r") as file:
    sample_mention = json.load(file)


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


def conll_lemma_baseline(mentions: List[dict]) -> float:
    """
    Calculates ConLL F1 for a simple same-lemma CDCR baseline. Uses official perl script from CoNLL.
    Args:
        mentions: a list of mentions

    Returns: float, calculated F1 CoNll
    """
    resolved_mentions_dict = {}
    pred_true_df = pd.DataFrame()

    for mention in mentions:
        if mention[MENTION_HEAD_LEMMA] not in resolved_mentions_dict:
            resolved_mentions_dict[mention[MENTION_HEAD_LEMMA]] = []
        resolved_mentions_dict[mention[MENTION_HEAD_LEMMA]].append(mention[MENTION_ID])
        pred_true_df = pd.concat([pred_true_df, pd.DataFrame({
            TRUE_LABEL: mention[COREF_CHAIN]
        }, index=[mention[MENTION_ID]])], axis=0)

    pred_true_df[PRED_LABEL] = [0] * len(pred_true_df)
    for i, (key, values) in enumerate(resolved_mentions_dict.items()):
        for v in values:
            pred_true_df.loc[v, PRED_LABEL] = key

    # case for only gold mentions
    general_tag = "CDCR/cdcr_topic"
    in_str = f'#begin document ({general_tag}); part 000\n'
    out_str = f'#begin document ({general_tag}); part 000\n'

    true_label_ids_dict = {v: i for i, v in enumerate(set(pred_true_df[TRUE_LABEL].values))}
    pred_label_ids_dict = {v: i for i, v in enumerate(set(pred_true_df[PRED_LABEL].values))}
    for index_, row in pred_true_df.iterrows():
        in_str += f'{general_tag}\t({str(true_label_ids_dict[row[TRUE_LABEL]])})\n'
        out_str += f'{general_tag}\t({str(pred_label_ids_dict[row[PRED_LABEL]])})\n'

    in_str += "#end document\n"
    out_str += "#end document\n"
    with open(os.path.join(TMP_PATH, "key.key_conll"), "w") as file:
        file.write(in_str)
    with open(os.path.join(TMP_PATH, "response.response_conll"), "w") as file:
        file.write(out_str)
    result_file = os.path.join(TMP_PATH, "conll_res.txt")

    scorer_command = (f'perl {os.path.join(os.getcwd(), "scorer", "scorer.pl")} '
                      f'all {os.path.join(TMP_PATH, "key.key_conll")} '
                      f'{os.path.join(TMP_PATH, "response.response_conll")} none > {result_file} \n')

    processes = []
    # LOGGER.info('Run CoNLL scorer perl command for CDCR')
    processes.append(subprocess.Popen(scorer_command, shell=True))

    while processes:
        status = processes[0].poll()
        if status is not None:
            processes.pop(0)

    # LOGGER.info('Running CoNLL scorers has been completed.')

    f1_dict = {}
    metrics_list = [MUC, B3, CEAF_M, CEAF_E, BLANC]
    params = [R, P, F1]
    i = 0
    with open(result_file, "r") as ins:
        for line in ins:
            new_line = line.strip()
            if new_line.find('F1:') != -1:
                if i >= len(metrics_list):
                    break
                f1_dict[metrics_list[i]] = {}
                if new_line.find('Coreference') != -1:
                    j = 0
                    for value in new_line.replace("\t", " ").split(' '):
                        if "%" not in value:
                            continue
                        param = params[j]
                        f1_dict[metrics_list[i]][param] = round(float(value[:-1]), 2)
                        j += 1
                    i += 1

    # metrics_df = pd.DataFrame()
    avg_f1 = []
    # conll_count = 0

    for metrics_name, metrics_vals in f1_dict.items():
        if metrics_name not in [MUC, B3, CEAF_E]:
            continue

        avg_f1.append(metrics_vals[F1])
    if len(avg_f1):
        f1_conll = np.mean(avg_f1)
    else:
        LOGGER.warning(f'CoNLL score was not calculated. Most likely, perl is not installed.')
        f1_conll = 0
    return float(format(f1_conll, '.3f'))


if __name__ == '__main__':
    for i, dataset in enumerate(DIRECTORIES_TO_SUMMARIZE):
        print(f'{i}: {dataset.split("-")[0]}')

    input_str = input("List dataset IDs with comma separation which to include into the summary or print \"all\" to summarize all: ")
    if "all" in input_str:
        selected_dir_to_summarize = DIRECTORIES_TO_SUMMARIZE
    else:
        selected_dataset_ids = [int(part.strip()) for part in input_str.split(',')]
        selected_dir_to_summarize = [DIRECTORIES_TO_SUMMARIZE[i] for i in selected_dataset_ids]

    LOGGER.info(f'Selected dataset to summarize: {selected_dir_to_summarize}')

    summary_df = pd.DataFrame()
    chain_df_all = pd.DataFrame()
    all_mentions_list = []

    # read annotated mentions and corresponding texts
    for dataset_folder in selected_dir_to_summarize:
        mentions_df = pd.DataFrame()

        LOGGER.info(f'Reading files with mentions for {dataset_folder} dataset..')
        for mention_type, file_name in zip([EVENT, ENTITY], [MENTIONS_EVENTS_JSON, MENTIONS_ENTITIES_JSON]):
            full_filename = os.path.join(os.getcwd(), dataset_folder, OUTPUT_FOLDER_NAME, file_name)
            with open(full_filename, encoding='utf-8', mode='r') as file:
                mentions_read_list = json.load(file)

            for v in mentions_read_list:
                missed_attributes = set(sample_mention.keys()) - set(v.keys())
                if len(missed_attributes):
                    LOGGER.warning(f'Dataset {dataset_folder.split("-")[0]} misses mentions\' attributes {missed_attributes} '
                                   f'and this may cause troubles in the script execution')

            df_tmp = pd.DataFrame(mentions_read_list, index=list(range(len(mentions_read_list))))
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
        # per topic in the dataset
        process_list = [(dataset, topic_id, subtopic, group_df)
                             for (dataset, topic_id, subtopic), group_df in mentions_df.groupby([DATASET_NAME, TOPIC_ID, SUBTOPIC])]
        # full dataset
        process_list.extend([(dataset, "", None, dataset_df) for dataset, dataset_df in mentions_df.groupby([DATASET_NAME])])

        conll_f1_dict = {}

        # collect statistics about the dataset
        for dataset, topic_id, subtopic, group_df in tqdm(process_list):
            if dataset not in conll_f1_dict:
                conll_f1_dict[dataset] = {}

            coref_chains = list(set(group_df[COREF_CHAIN].values))

            # general statistics
            summary_dict = {
                DATASET_NAME: dataset,
                TOPIC: f'{topic_id}/{subtopic}',
                TOPICS: len(set(group_df[TOPIC].values)),
                ARTICLES: len(set(group_df[DOC_ID].values)),
                TOKENS: len(df_conll[df_conll[TOPIC_SUBTOPIC].str.contains(f'{topic_id}/{subtopic}')]) if subtopic is not None else len(df_conll),
                COREF_CHAIN: len(coref_chains),
                MENTIONS: len(group_df),
                f'{EVENT}_{MENTIONS}': len(group_df[group_df[TYPE] == EVENT]),
                f'{ENTITY}_{MENTIONS}': len(group_df[group_df[TYPE] == ENTITY]),
                SINGLETONS: len(group_df[group_df[IS_SINGLETON]]),
            }

            # various for, of lexical diversity that depend on the presence/absence of singletons
            for suff, filt_criteria in zip([ALL, WO_SINGL], [0, 1]):
                selected_chains_df = chain_df[(chain_df[MENTIONS] > filt_criteria) & (chain_df.index.isin(coref_chains))]
                if not len(selected_chains_df):
                    continue
                summary_dict[AVERAGE_SIZE + suff] = float(format(np.mean(selected_chains_df[MENTIONS].values), '.3f'))
                summary_dict[PHRASING_DIVERSITY + WEIGHTED + suff] = float(format(sum([row[PHRASING_DIVERSITY] * row[MENTIONS]
                                                                  for index, row in selected_chains_df.iterrows()]) / \
                                                                     sum(selected_chains_df[MENTIONS].values), '.3f'))
                summary_dict[UNIQUE_LEMMAS + suff] = float(format(np.mean(selected_chains_df[UNIQUE_LEMMAS].values), '.3f'))

                if subtopic:
                    conll_f1 = conll_lemma_baseline([v for v in all_mentions_list if v[COREF_CHAIN] in list(selected_chains_df.index)])
                    summary_dict[F1 + CONLL + suff] = conll_f1
                    if suff not in conll_f1_dict[dataset]:
                        conll_f1_dict[dataset][suff] = []
                    conll_f1_dict[dataset][suff].append(conll_f1)
                else:
                    summary_dict[F1 + CONLL + suff] = float(format(np.mean(conll_f1_dict[dataset][suff]), '.3f'))

            summary_df = pd.concat([summary_df, pd.DataFrame(summary_dict, index=[f'{dataset}\\{topic_id}\\subtopic'])], axis=0)
        chain_df_all = pd.concat([chain_df_all, chain_df], axis=0)

        #output the chains statistics
    chain_df_all.reset_index().to_csv(os.path.join(os.getcwd(), SUMMARY_FOLDER, SUMMARY_CHAINS_CSV))
    summary_df.to_csv(os.path.join(os.getcwd(), SUMMARY_FOLDER, SUMMARY_TOPICS_CSV))

    LOGGER.info(f'Summary computed over {len(DIRECTORIES_TO_SUMMARIZE)} datasets ({DIRECTORIES_TO_SUMMARIZE}) '
                f'is completed.')
