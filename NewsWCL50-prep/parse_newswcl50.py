from setup import *
from logger import LOGGER
from utils import make_save_conll
AGGR_FILENAME = 'aggr_m_conceptcategorization.csv'

import spacy
import os
import json
import pandas as pd
import shortuuid
import re
from tqdm import tqdm
from typing import List

DATASET_PATH = os.path.join(os.getcwd(), NEWSWCL50_FOLDER_NAME)
TOPICS_FOLDER = os.path.join(DATASET_PATH, "topics")
ANNOTATIONS_FOLDER = os.path.join(DATASET_PATH, "annotations")

# loading spacy NLP
nlp = spacy.load('en_core_web_sm')


def check_continuous(token_numbers: List[int]):
    """
    Checks whether a list of token ints is continuous
    Args:
        token_numbers:

    Returns: boolean
    """
    return token_numbers == list(range(token_numbers[0], token_numbers[-1] + 1))


if __name__ == '__main__':
    LOGGER.info("Starting routine... Retrieving data from specified directories.")
    with open(os.path.join(os.getcwd(), "..", SAMPLE_DOC_JSON), "r") as file:
        sample_fields = list(json.load(file).keys())
    topic_df = pd.DataFrame(columns=sample_fields)
    topic_counter = 0

    # read topic files
    for subfolder_name in os.listdir(TOPICS_FOLDER):
        LOGGER.info(f'Reading files from {subfolder_name}.')

        # iterate over every file
        for file_name in os.listdir(os.path.join(TOPICS_FOLDER, subfolder_name)):
            full_filename = os.path.join(TOPICS_FOLDER, subfolder_name, file_name)

            with open(full_filename, encoding='utf-8', mode='r') as file:
                topic_dict = {}
                for k,v in json.load(file).items():
                    topic_dict[k] = v if type(v) != list else ", ".join(v)
                topic_df = pd.concat([topic_df, pd.DataFrame(topic_dict, index=[topic_counter])], ignore_index=True, axis=0)
            topic_counter =+ 1

    docs = {}
    topic_df[CONCAT_TEXT] = ["\n".join([topic_df.loc[index, field] for field in [TITLE, DESCRIPTION, TEXT]
                                          if topic_df.loc[index, field]])
                               for index in list(topic_df.index)]

    # appending all articles as spacy docs to the dataframe
    for index, row in tqdm(topic_df.iterrows(), total=topic_df.shape[0]):
        doc = {}
        text = row[CONCAT_TEXT]
        doc_preproc = nlp(text)
        par_id = -1

        for sent_id, sent in enumerate(doc_preproc.sents):
            if "\n" in sent.orth_[:3] or sent_id == 0:
                par_id += 1
                doc[par_id] = {}
            doc[par_id][sent_id] = sent
            if sent.orth_[-1] == "\n":
                par_id += 1
                doc[par_id] = {}
        docs[row[SOURCE_DOMAIN]] = doc

    #read other csvs for the annotations
    df_annotations = pd.DataFrame()
    topics_dict = {}

    for file_name in os.listdir(ANNOTATIONS_FOLDER):
        full_filename = os.path.join(ANNOTATIONS_FOLDER, file_name)
        LOGGER.info(f'Executing code for {str(full_filename)}')
        df_tmp = pd.read_csv(full_filename)
        df_tmp = df_tmp[~df_tmp[CODE].str.contains("Properties")]
        topic_id = int(file_name.split("_")[0])
        df_tmp[TOPIC_ID] = topic_id

        topics_dict[topic_id] = file_name.split(".")[0]
        df_annotations = pd.concat([df_annotations, df_tmp], ignore_index=True, axis=0)

    # Open the aggregated file to get the entity types per code
    concept_df = pd.read_csv(os.path.join(os.getcwd(), NEWSWCL50_FOLDER_NAME, AGGR_FILENAME))

    # assign every segment that gets mentioned to a specific code and entity type
    df_annotations = pd.merge(df_annotations, concept_df, how="left", on=[TOPIC_ID, CODE])

    # make sure no NA values are present after merging
    df_annotations = df_annotations[df_annotations.type.notna()]
    df_annotations.reset_index(drop=True, inplace=True)

    # create coref_chain ids which show the connection/corellation of many segment mentions within the same topic
    coref_chains = {}
    # chains_list = []
    df_annotations[COREF_CHAIN] = [""] * len(df_annotations)
    for index, row in df_annotations.iterrows():
        unique_chain_name = "_".join(str(row[col]) for col in [TOPIC_ID, CODE, TYPE])
        unique_chain_name = unique_chain_name.replace("\\", "_").replace(" ", "_")
        df_annotations.loc[index, COREF_CHAIN] = unique_chain_name
        coref_chains[unique_chain_name] = {MENTION_FULL_TYPE: row[TYPE],
                                           MENTION_TYPE: row[TYPE],
                                           COREF_CHAIN: unique_chain_name}

    LOGGER.info(f'Parsing and matching annotations in the texts...')
    mentions_df = pd.DataFrame()
    mentions_dict = {}
    not_found_list = {}

    for (coref_chain, doc_name, mention_orig, paragraph_orig, topic_id, code), group_df in\
            tqdm(df_annotations.groupby([COREF_CHAIN, DOCUMENT_NAME, SEGMENT, BEGINNING, TOPIC_ID, CODE])):
        # tokenize the segment
        mention_orig = re.sub("’", "'", mention_orig)
        mention_orig = re.sub("‘", "'", mention_orig)
        mention_orig = re.sub("“", "\"", mention_orig)
        mention_orig = re.sub("”", "\"", mention_orig)
        segment_doc = nlp(mention_orig)
        segment_tokenized = [t for t in segment_doc]
        found_mentions_counter = 0

        for paragraph_correction in [-1, -2, 0, -3, -4, 1, 2]:
            modified_par = paragraph_orig + paragraph_correction
            try:
                paragraph = docs[doc_name][modified_par]
            except KeyError:
                continue

            # iterate over every token of the sentence
            for sent_id, sentence in paragraph.items():
                # a counter to show with which token in the tokenized segment to match
                start_token_ids = []
                for i, token in enumerate(sentence):
                    if token.norm_.lower() == segment_tokenized[0].norm_.lower():
                        start_token_ids.append(i)
                if not start_token_ids:
                    continue

                for start_token_id in start_token_ids:
                    found_tokens = []

                    for i in range(len(segment_tokenized)):
                        try:
                            if sentence[i + start_token_id].norm_.lower() == segment_tokenized[i].norm_.lower():
                                found_tokens.append(sentence[i + start_token_id])
                        except IndexError:
                            continue

                    norm_annot = re.sub(r'\W+', "",  " ".join([t.norm_ for t in segment_tokenized])).lower()
                    norm_found = re.sub(r'\W+', "",  " ".join([t.norm_ for t in found_tokens])).lower()
                    if norm_annot != norm_found:
                        a = 1
                        continue
                    else:
                        found_mentions_counter += 1

                        # determine the head of the mention tokens
                        found_token_ids = list(range(start_token_id, start_token_id + len(found_tokens)))
                        found_tokens_global_ids = [t.i for t in found_tokens]
                        tokens_text = [t.text for t in found_tokens]
                        mention_head_token = None
                        mention_head_token_id = -1

                        for t_id, token in zip(found_token_ids, found_tokens):
                            # mention head's ancestors should not be in the found tokens
                            if all([a.i not in found_tokens_global_ids for a in token.ancestors]):
                                if token.pos_ in ["DET", "PUNCT"]:
                                    continue
                                mention_head_token = token
                                mention_head_token_id = t_id

                        mention_id = "_".join([doc_name, str(sent_id), str(mention_head_token_id), shortuuid.uuid()[:4]])

                        context_min_id = max(min(found_tokens_global_ids) - CONTEXT_RANGE, 0)
                        context_max_id = min(max(found_tokens_global_ids) + CONTEXT_RANGE, len(mention_head_token.doc) - 1)
                        mention_context_str = [t.text for t in mention_head_token.doc[context_min_id:context_max_id]]
                        mentions_dict[mention_id] = {COREF_CHAIN: coref_chain,
                                                     TOKENS_NUMBER: found_token_ids,
                                                     DOC_ID: doc_name,
                                                     DOC: doc_name,
                                                     SCORE: -1,
                                                     SENT_ID: sent_id,
                                                     MENTION_TYPE: coref_chains[coref_chain][MENTION_TYPE],
                                                     MENTION_FULL_TYPE: coref_chains[coref_chain][MENTION_FULL_TYPE],
                                                     MENTION_ID: mention_id,
                                                     TOPIC_ID: str(topic_id),
                                                     TOPIC: str(topic_id),
                                                     SUBTOPIC: topics_dict[topic_id],
                                                     SUBTOPIC_ID: str(topic_id),
                                                     DESCRIPTION: code,
                                                     COREF_TYPE: IDENTITY,
                                                     MENTION_NER: mention_head_token.ent_type_ if mention_head_token.ent_type_ else "O",
                                                     MENTION_HEAD_POS: mention_head_token.pos_,
                                                     MENTION_HEAD_LEMMA: mention_head_token.lemma_,
                                                     MENTION_HEAD: mention_head_token.text,
                                                     MENTION_HEAD_ID: mention_head_token_id,
                                                     IS_CONTINIOUS: bool(check_continuous([t.i for t in found_tokens])),
                                                     IS_SINGLETON: False,
                                                     MENTION_CONTEXT: mention_context_str,
                                                     TOKENS_STR: "".join([t.text_with_ws for t in found_tokens]),
                                                     TOKENS_TEXT: [t.text for t in found_tokens],
                                                     CONLL_DOC_KEY: f'{topic_id}/{topic_id}/{doc_name}'
                                                     }

                        mentions_df = pd.concat([mentions_df, pd.DataFrame({
                                               COREF_CHAIN: coref_chain,
                                               DOC_ID: doc_name,
                                               SENT_ID: sent_id,
                                               MENTION_HEAD_ID: mention_head_token_id,
                                               "char_length": len("".join([t.text_with_ws for t in found_tokens]))},
                            index=[mention_id])], axis=0)
            if found_mentions_counter:
                break

        if not found_mentions_counter:
            LOGGER.warning(f'A mention \"{mention_orig}\" was not found in document {doc_name} and will be skipped. ')

    LOGGER.warning(f'Not found annotations in the text ({len(not_found_list)}): \n{list(not_found_list)}')

    mentions_df = mentions_df.sort_values(by=[COREF_CHAIN, DOC_ID, SENT_ID, MENTION_HEAD_ID, "char_length"],
                                          ascending=[True, True, True, True, False])
    mentions_df_unique = mentions_df.drop_duplicates([COREF_CHAIN, DOC_ID, SENT_ID, MENTION_HEAD_ID], keep="first")
    mentions_unique_dict = {k:v for k, v in mentions_dict.items() if k in list(mentions_df_unique.index)}

    events = ["ACTION", "EVENT", "MISC"]
    mentions_events_list = []
    mentions_entities_list = []
    chain_df = mentions_df_unique[[DOC_ID, COREF_CHAIN]].groupby(COREF_CHAIN).count()
    for index, row in mentions_df_unique.iterrows():
        if mentions_unique_dict[index][MENTION_FULL_TYPE] in ["MISC", "ACTOR-I"]:
            continue
        mentions_unique_dict[index][IS_SINGLETON] = bool(chain_df.loc[row[COREF_CHAIN], DOC_ID] == 1)
        if mentions_unique_dict[index][MENTION_FULL_TYPE] in events:
            mentions_events_list.append(mentions_unique_dict[index])
        else:
            mentions_entities_list.append(mentions_unique_dict[index])

    output_path = os.path.join(os.getcwd(), OUTPUT_FOLDER_NAME)

    with open(os.path.join(output_path, MENTIONS_EVENTS_JSON), 'w', encoding='utf-8') as file:
        json.dump(mentions_events_list, file)

    with open(os.path.join(output_path, MENTIONS_ENTITIES_JSON), 'w', encoding='utf-8') as file:
        json.dump(mentions_entities_list, file)

    # save all mentions as csv
    df_all_mentions = pd.DataFrame()
    for mention in mentions_entities_list + mentions_events_list:
        df_all_mentions = pd.concat([df_all_mentions, pd.DataFrame({
            attr: str(value) if type(value) == list else value for attr, value in mention.items()
        }, index=[mention[MENTION_ID]])], axis=0)
    df_all_mentions.to_csv(os.path.join(os.getcwd(), OUTPUT_FOLDER_NAME, MENTIONS_ALL_CSV))

    LOGGER.info("Generating conll...")
    conll_list = []
    token_keys_conll = []
    annot_token_dict = {}

    # first create a conll dataframe that gets a new row for each token
    for doc_id, doc in tqdm(docs.items()):
        if doc_id not in annot_token_dict:
            annot_token_dict[doc_id] = {}
        for par_id, par in doc.items():
            for sentence_id, sentence in par.items():
                if sentence_id not in annot_token_dict[doc_id]:
                    annot_token_dict[doc_id][sentence_id] = {}

                for token_id, token in enumerate(sentence):
                    if token_id not in annot_token_dict[doc_id][sentence_id]:
                        annot_token_dict[doc_id][sentence_id][token_id] = pd.DataFrame()

                    token_text = token.text
                    if "\n" in token_text:
                        token_text = token_text.replace("\n", "\\n")  # avoid unwanted like breaks in the conll file

                    conll_list.append(
                        {TOPIC_SUBTOPIC_DOC: f'{doc_id.split("_")[0]}/{doc_id.split("_")[0]}/{doc_id}',
                         DOC_ID: doc_id,
                         SENT_ID: sentence_id,
                         TOKEN_ID: token_id,
                         TOKEN: token_text,
                         REFERENCE: "-"})
                    token_keys_conll.append("_".join([doc_id, str(sentence_id), str(token_id)]))

    df_conll = pd.DataFrame(conll_list, index=token_keys_conll)

    # then annotate each token (i.e. row) in the conll df with the coref_chain id
    added_corefs = []

    LOGGER.info(f'Processing {len(mentions_df_unique)} mentions rows and assigning to the conll text...')
    for mention_values in tqdm(mentions_entities_list + mentions_events_list):
        mention_id = mention_values[MENTION_ID]
        if len(mention_values[TOKENS_NUMBER]) == 1:
            annot_token_dict[mention_values[DOC_ID]][mention_values[SENT_ID]][mention_values[MENTION_HEAD_ID]] = \
                pd.concat([annot_token_dict[mention_values[DOC_ID]][mention_values[SENT_ID]][mention_values[MENTION_HEAD_ID]],
                           pd.DataFrame({
                                COREF_CHAIN: f'({mention_values[COREF_CHAIN]})',
                                FIRST_TOKEN: mention_values[TOKENS_NUMBER][0],
                                LAST_TOKEN:mention_values[TOKENS_NUMBER][-1]
                           }, index=[mention_id])])
        else:
            # first token
            annot_token_dict[mention_values[DOC_ID]][mention_values[SENT_ID]][mention_values[TOKENS_NUMBER][0]] = \
                pd.concat([annot_token_dict[mention_values[DOC_ID]][mention_values[SENT_ID]][mention_values[TOKENS_NUMBER][0]],
                           pd.DataFrame({
                                COREF_CHAIN: f'({mention_values[COREF_CHAIN]}',
                                FIRST_TOKEN: mention_values[TOKENS_NUMBER][0],
                                LAST_TOKEN: mention_values[TOKENS_NUMBER][-1]
                           }, index=[mention_id])])
            # last token
            annot_token_dict[mention_values[DOC_ID]][mention_values[SENT_ID]][mention_values[TOKENS_NUMBER][-1]] = \
                pd.concat([annot_token_dict[mention_values[DOC_ID]][mention_values[SENT_ID]][mention_values[TOKENS_NUMBER][-1]],
                           pd.DataFrame({
                                COREF_CHAIN: f'{mention_values[COREF_CHAIN]})',
                                FIRST_TOKEN: mention_values[TOKENS_NUMBER][0],
                                LAST_TOKEN: mention_values[TOKENS_NUMBER][-1]
                           }, index=[mention_id])])

    # output conll data as json file for the dataset_summary to handle more easily
    for doc_id, doc in tqdm(annot_token_dict.items()):
        for sent_id, sent in doc.items():
            for token_id, token_values in sent.items():
                if not len(token_values):
                    continue

                token_values_sort = token_values.sort_values(by=[FIRST_TOKEN, LAST_TOKEN], ascending=[False, False])
                df_conll.loc["_".join([doc_id, str(sent_id), str(token_id)]), REFERENCE] = "| ".join(token_values_sort[COREF_CHAIN].values)

    make_save_conll(df_conll, df_all_mentions, output_path)

    LOGGER.info(f'Done! \nNumber of unique mentions: {len(mentions_df_unique)} \nNumber of unique chains: {len(chain_df)} ')
