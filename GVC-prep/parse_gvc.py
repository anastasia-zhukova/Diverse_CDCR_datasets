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
sys.path.insert(0, '..')
from utils import *
import shortuuid
from nltk import Tree
from tqdm import tqdm
import warnings
from setup import *
from logger import LOGGER

GVC_PARSING_FOLDER = os.path.join(os.getcwd())
OUT_PATH = os.path.join(GVC_PARSING_FOLDER, OUTPUT_FOLDER_NAME)
source_path = os.path.join(GVC_PARSING_FOLDER, GVC_FOLDER_NAME)

nlp = spacy.load('en_core_web_sm')


def conv_files():
    topic_name = "0_gun_violence"
    topic_id = "0"
    train_dev_test_split_dict = {}
    conll_df = pd.DataFrame()
    need_manual_review_mention_head = {}
    event_mentions = []

    for file_name in ["dev", "test", "train"]:
        df = pd.read_csv(os.path.join(source_path, f'{file_name}.csv'), header=None)
        train_dev_test_split_dict[file_name] = list(df[1])

    subtopic_structure_dict = {}
    for index, row in pd.read_csv(os.path.join(source_path, "gvc_doc_to_event.csv")).iterrows():
        subtopic_structure_dict[row["doc-id"]] = row["event-id"]

    #
    with open(os.path.join(source_path, 'verbose.conll'), encoding="utf-8") as f:
        conll_str = f.read()

    conll_lines = conll_str.split("\n")
    doc_id_prev = ""
    orig_sent_id_prev = ""
    sent_id = 0
    token_id = 0
    mentions_dict = {}
    mention_id = ""
    doc_title_dict = {}
    coref_dict = {}

    for i, conll_line in tqdm(enumerate(conll_lines), total=len(conll_lines)):
        if i+1 == len(conll_lines):
            break

        if "#begin document" in conll_line or "#end document" in conll_line:
            continue

        original_key, token, part_of_text, chain_description, chain_value = conll_line.split("\t")
        try:
            doc_id, orig_sent_id, _ = original_key.split(".")
        except ValueError:
            doc_id, orig_sent_id = original_key.split(".")

        chain_id = re.sub("\D+", "", chain_value)
        coref_dict[chain_id] = coref_dict.get(chain_id, 0) + 1
        subtopic_id = subtopic_structure_dict[doc_id]
        topic_subtopic_doc = f'{topic_id}/{subtopic_id}/{doc_id}'

        if doc_id != doc_id_prev:
            sent_id = 0
            token_id = 0
        else:
            if orig_sent_id_prev != orig_sent_id:
                sent_id += 1
                token_id = 0

        if part_of_text == "TITLE":
            if doc_id not in doc_title_dict:
                doc_title_dict[doc_id] = []
            doc_title_dict[doc_id].append(token)

        if chain_value.strip() == f'({chain_id}' or chain_value.strip() == f'({chain_id})':
            mention_id = shortuuid.uuid(original_key)
            if chain_id == "0":
                chain_id = mention_id
                coref_dict[chain_id] = coref_dict.get(chain_id, 0) + 1

            mentions_dict[mention_id] = {
                COREF_CHAIN: chain_id,
                DESCRIPTION: chain_description,
                MENTION_ID: mention_id,
                DOC_ID: doc_id,
                DOC: "",
                SENT_ID: int(sent_id),
                SUBTOPIC_ID: str(subtopic_id),
                SUBTOPIC: str(subtopic_id),
                TOPIC_ID: topic_id,
                TOPIC: topic_name,
                COREF_TYPE: IDENTITY,
                CONLL_DOC_KEY: topic_subtopic_doc,
                "words": []}
            mentions_dict[mention_id]["words"].append((token, token_id))
            if chain_value.strip() == f'({chain_id})':
                mention_id = ""

        elif mention_id and chain_value == chain_id:
            mentions_dict[mention_id]["words"].append((token, token_id))

        elif chain_value.strip() == f'{chain_id})':
            mentions_dict[mention_id]["words"].append((token, token_id))
            mention_id = ""

        conll_df = pd.concat([conll_df, pd.DataFrame({
            TOPIC_SUBTOPIC_DOC: topic_subtopic_doc,
            DOC_ID: conll_line.split("\t")[0].split(".")[0],
            SENT_ID: sent_id,
            TOKEN_ID: token_id,
            TOKEN: token.replace("NEWLINE", "//n"),
            REFERENCE: "-"
        }, index=[f'{doc_id}/{sent_id}/{token_id}'])])
        token_id += 1
        doc_id_prev = doc_id
        orig_sent_id_prev = orig_sent_id

    # parse information about mentions
    for mention_id, mention in tqdm(mentions_dict.items()):
        word_start = mention["words"][0][1]
        word_end = mention["words"][-1][1]
        # coref_id = mention_orig["entity"]
        doc_id = mention[DOC_ID]
        sent_id = mention[SENT_ID]
        markable_df = conll_df.loc[f'{doc_id}/{sent_id}/{word_start}': f'{doc_id}/{sent_id}/{word_end}']
        if not len(markable_df):
            continue

        # mention attributes
        token_ids = [int(t) for t in list(markable_df[TOKEN_ID].values)]
        token_str = ""
        tokens_text = list(markable_df[TOKEN].values)
        for token in tokens_text:
            token_str, word_fixed, no_whitespace = append_text(token_str, token)

        sent_id = list(markable_df[SENT_ID].values)[0]

        # determine the sentences as a string
        sentence_str = ""
        sent_tokens = list(conll_df[(conll_df[DOC_ID] == doc_id) & (conll_df[SENT_ID] == sent_id)][TOKEN])
        for t in sent_tokens:
            sentence_str, _, _ = append_text(sentence_str, t)

        # pass the string into spacy
        doc = nlp(sentence_str)

        tolerance = 0
        token_found = {t: None for t in token_ids}
        prev_id = 0
        for t_id, t in zip(token_ids, tokens_text):
            to_break = False
            for tolerance in range(10):
                for token in doc[max(0, t_id - tolerance): t_id + tolerance + 1]:
                    if token.i < prev_id:
                        continue
                    if token.text == t:
                        token_found[t_id] = token.i
                        prev_id = token.i
                        to_break = True
                        break
                    elif t.startswith(token.text):
                        token_found[t_id] = token.i
                        prev_id = token.i
                        to_break = True
                        break
                    elif token.text.startswith(t):
                        token_found[t_id] = token.i
                        prev_id = token.i
                        to_break = True
                        break
                    elif t.endswith(token.text):
                        token_found[t_id] = token.i
                        prev_id = token.i
                        to_break = True
                        break
                if to_break:
                    break
        # whole mention string processed, look for the head
        mention_head_id = None
        mention_head = None
        if f'{doc_id}/{mention_id}' not in need_manual_review_mention_head:
            found_mentions_tokens = doc[min([t for t in token_found.values()]): max(
                [t for t in token_found.values()]) + 1]
            if len(found_mentions_tokens) == 1:
                mention_head = found_mentions_tokens[0]
                # remap the mention head back to the np4e original tokenization to get the ID for the output
                for t_orig, t_mapped in token_found.items():
                    if t_mapped == mention_head.i:
                        mention_head_id = t_orig
                        break

            if mention_head is None:
                found_mentions_tokens_ids = list(token_found.values())
                # found_mentions_tokens_ids = set([t.i for t in found_mentions_tokens])
                for i, t in enumerate(found_mentions_tokens):
                    if t.head.i == t.i:
                        # if a token is a root, it is a candidate for the head
                        pass

                    elif t.head.i >= min(found_mentions_tokens_ids) and t.head.i <= max(found_mentions_tokens_ids):
                        # check if a head the candidate head is outside the mention's boundaries
                        if t.head.text in tokens_text:
                            # a head of a candiate head cannot be in the text of the mention
                            continue

                    mention_head = t
                    if mention_head.pos_ == "DET":
                        mention_head = None
                        continue

                    to_break = False
                    # remap the mention head back to the np4e original tokenization to get the ID for the output
                    for t_orig, t_mapped in token_found.items():
                        if t_mapped == mention_head.i:
                            mention_head_id = t_orig
                            to_break = True
                            break
                    if to_break:
                        break

        # add to manual review if the resulting token is not inside the mention
        # (error must have happened)
        if mention_head_id is None:  # also "if is None"
            if f'{doc_id}/{mention_id}' not in need_manual_review_mention_head:
                need_manual_review_mention_head[f'{doc_id}/{mention_id}'] = \
                    {
                        "mention_text": list(zip(token_ids, tokens_text)),
                        "sentence_tokens": list(enumerate(sent_tokens)),
                        "spacy_sentence_tokens": [(i, t.text) for i, t in enumerate(doc)],
                        "tolerance": tolerance
                    }

                LOGGER.warning(
                    f"Mention with ID {doc_id}/{mention_id} ({token_str}) needs manual review. Could not "
                    f"determine the mention head automatically. {str(tolerance)}")

        doc_df = conll_df[conll_df[DOC_ID] == doc_id]
        token_mention_start_id = list(doc_df.index).index(f'{doc_id}/{sent_id}/{word_start}')
        context_min_id = 0 if token_mention_start_id - CONTEXT_RANGE < 0 else token_mention_start_id - CONTEXT_RANGE
        context_max_id = min(token_mention_start_id + CONTEXT_RANGE, len(doc_df))

        mention_context_str = list(doc_df.iloc[context_min_id:context_max_id][TOKEN].values)

        # add to mentions if the variables are correct ( do not add for manual review needed )
        if f'{doc_id}/{mention_id}' not in need_manual_review_mention_head:
            mention_ner = mention_head.ent_type_ if mention_head.ent_type_ != "" else "O"
            mention_type = "OCCURRENCE"
            mention.update({
                MENTION_NER: mention_ner,
                MENTION_HEAD_POS: mention_head.pos_,
                MENTION_HEAD_LEMMA: mention_head.lemma_,
                MENTION_HEAD: mention_head.text,
                MENTION_HEAD_ID: int(mention_head_id),
                DOC: "_".join(doc_title_dict[doc_id]),
                IS_CONTINIOUS: token_ids == list(
                    range(token_ids[0], token_ids[-1] + 1)),
                IS_SINGLETON: coref_dict[mention[COREF_CHAIN]] == 1,
                MENTION_TYPE: mention_type[:3],
                MENTION_FULL_TYPE: mention_type,
                SCORE: -1.0,
                MENTION_CONTEXT: mention_context_str,
                TOKENS_NUMBER: token_ids,
                TOKENS_STR: token_str,
                TOKENS_TEXT: tokens_text
            })
            mention.pop("words")
            event_mentions.append(mention)

    entity_mentions = []
    df_all_mentions = pd.DataFrame()
    for mention in tqdm(entity_mentions + event_mentions):
        df_all_mentions = pd.concat([df_all_mentions, pd.DataFrame({
            attr: str(value) if type(value) == list else value for attr, value in mention.items()
        }, index=[mention[MENTION_ID]])], axis=0)
    df_all_mentions.to_csv(os.path.join(OUT_PATH, MENTIONS_ALL_CSV))

    conll_df = conll_df.reset_index(drop=True)
    conll_df_labels = make_save_conll(conll_df, df_all_mentions, OUT_PATH)

    with open(os.path.join(OUT_PATH, MENTIONS_ENTITIES_JSON), "w") as file:
        json.dump(entity_mentions, file)

    with open(os.path.join(OUT_PATH, MENTIONS_EVENTS_JSON), "w") as file:
        json.dump(event_mentions, file)

    LOGGER.info(
        "Mentions that need manual review to define the head and its attributes have been saved to: " +
        MANUAL_REVIEW_FILE + " - Total: " + str(len(need_manual_review_mention_head)))

    with open(os.path.join(TMP_PATH, MANUAL_REVIEW_FILE), "w", encoding='utf-8') as file:
        json.dump(need_manual_review_mention_head, file)

    LOGGER.info(f'Done! \nNumber of unique mentions: {len(df_all_mentions)} '
                f'\nNumber of unique chains: {len(set(df_all_mentions[COREF_CHAIN].values))} ')

    LOGGER.info(f'Splitting GVC into train/dev/test subsets...')
    for subset, subtopic_ids in train_dev_test_split_dict.items():
        LOGGER.info(f'Creating data for {subset} subset...')
        split_folder = os.path.join(OUT_PATH, subset)
        if subset not in os.listdir(OUT_PATH):
            os.mkdir(split_folder)

        selected_entity_mentions = []
        for mention in entity_mentions:
            if int(mention[SUBTOPIC_ID]) in subtopic_ids:
                selected_entity_mentions.append(mention)

        selected_event_mentions = []
        for mention in event_mentions:
            if int(mention[SUBTOPIC_ID]) in subtopic_ids:
                selected_event_mentions.append(mention)

        with open(os.path.join(split_folder, MENTIONS_ENTITIES_JSON), "w", encoding='utf-8') as file:
            json.dump(selected_entity_mentions, file)

        with open(os.path.join(split_folder, MENTIONS_EVENTS_JSON), "w", encoding='utf-8') as file:
            json.dump(selected_event_mentions, file)

        conll_df_split = pd.DataFrame()
        for t_id in subtopic_ids:
            conll_df_split = pd.concat([conll_df_split,
                                        conll_df_labels[conll_df_labels[TOPIC_SUBTOPIC_DOC].str.contains(f'{t_id}/')]], axis=0)
        make_save_conll(conll_df_split, selected_event_mentions + selected_entity_mentions, split_folder, False)

    LOGGER.info(f'Parsing of GVC done!')


if __name__ == '__main__':
    LOGGER.info(f"Processing GVC {source_path[-34:].split('_')[2]}.")
    conv_files()
