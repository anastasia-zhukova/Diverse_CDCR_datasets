import xml.etree.ElementTree as ET
import os
import json
import string
import copy
import re
import pandas as pd
import numpy as np
from nltk import Tree
import spacy
import sys
import shortuuid
from tqdm import tqdm
from setup import *
from utils import *
from logger import LOGGER


path_sample = os.path.join(DATA_PATH, "_sample_doc.json")  # ->root/data/original/_sample_doc.json
NP4E_PARSING_FOLDER = os.path.join(os.getcwd())
OUT_PATH = os.path.join(NP4E_PARSING_FOLDER, OUTPUT_FOLDER_NAME)

nlp = spacy.load('en_core_web_sm')

# opens and loads the newsplease-format out of the json file: _sample_doc.json
with open(path_sample, "r") as file:
    newsplease_format = json.load(file)


source_path = os.path.join(NP4E_PARSING_FOLDER, NP4E_FOLDER_NAME)
result_path = os.path.join(OUT_PATH, 'test_parsing')
out_path = os.path.join(OUT_PATH)

subtopic_fullname_dict = {
    "bukavu": "Bukavu_bombing",
    "peru": "Peru_hostages",
    "tajikistan": "Tajikistan_hostages",
    "israel": "Israel_suicide_bomb",
    "china": "China-Taiwan_hijack"
}


def conv_files():
    doc_files = {}
    entity_mentions = []
    event_mentions = []
    conll_df = pd.DataFrame(columns=[TOPIC_SUBTOPIC_DOC, DOC_ID, SENT_ID, TOKEN_ID, TOKEN, REFERENCE])
    need_manual_review_mention_head = {}
    topic_name = "0_bomb_explosion_kidnap"

    for subtopic_id, subtopic in enumerate(os.listdir(source_path)):
        entity_mentions_local = []

        LOGGER.info(f"Parsing of NP4E topic {subtopic}...")
        subtopic_name_composite_full = f'{subtopic_id}_{subtopic_fullname_dict[subtopic]}'
        subtopic_name_composite = f'{subtopic_id}_{subtopic}'
        topic_folder = os.path.join(source_path, subtopic)

        # for file in tqdm(os.listdir(topic_folder)):
        #     if file == "Basedata":
        docs_folder = os.path.join(topic_folder, "Basedata")
        # docs_dict = {}
        coref_pre_dict = {}
        coref_pre_df = pd.DataFrame()

        for doc_text_name in os.listdir(docs_folder):
            # info_t_name = str(doc_text_name.split("_")[0])
            doc_id = re.sub(r'\D+', "", str(doc_text_name))
            topic_subtopic_doc = f'{topic_name.split("_")[0]}/{subtopic_name_composite}/{doc_id}'

            if doc_text_name.split(".")[-1] != "xml":
                continue

            doc_text_file_path = os.path.join(docs_folder, doc_text_name)
            tree = ET.parse(doc_text_file_path)
            root = tree.getroot()

            for t_id, elem in enumerate(root):
                conll_df = pd.concat([conll_df, pd.DataFrame({
                    TOPIC_SUBTOPIC_DOC: topic_subtopic_doc,
                    DOC_ID: doc_id,
                    SENT_ID: 0,
                    TOKEN_ID: t_id,
                    TOKEN: elem.text,
                    REFERENCE: "-"
                }, index=[f'{doc_id}/{elem.attrib["id"]}'])], axis=0)

            sentence_markables = ET.parse(os.path.join(topic_folder, "markables", doc_text_name.split("_")[0]+ "_sentence_level.xml"))
            root = sentence_markables.getroot()
            for sent_id, elem in enumerate(root):
                # span="word_243..word_284"
                try:
                    word_start, word_end = elem.attrib["span"].split("..")
                except ValueError:
                    word_start, word_end = elem.attrib["span"], elem.attrib["span"]
                conll_df.loc[f'{doc_id}/{word_start}': f'{doc_id}/{word_end}', SENT_ID] = sent_id
                local_df = conll_df.loc[f'{doc_id}/{word_start}': f'{doc_id}/{word_end}']
                conll_df.loc[f'{doc_id}/{word_start}': f'{doc_id}/{word_end}', TOKEN_ID] = list(range(len(local_df)))

            doc_df = conll_df[conll_df[DOC_ID] == doc_id]

            markables = ET.parse(os.path.join(topic_folder, "markables", doc_text_name.split("_")[0]+"_coref_level.xml"))
            root = markables.getroot()

            for markable in root:
                marker_id = markable.get("id")
                span_str = markable.get("span")
                coref_id = markable.get("coref_set")

                mention_id = shortuuid.uuid()

                if span_str == "p":
                    continue

                try:
                    word_start, word_end = span_str.split("..")
                except ValueError:
                    word_start, word_end = span_str, span_str

                markable_df = conll_df.loc[f'{doc_id}/{word_start}': f'{doc_id}/{word_end}']
                if not len(markable_df):
                    continue

                # mention attributes
                token_ids = list(markable_df[TOKEN_ID].values)
                tokens = {}
                token_str = ""
                tokens_text = list(markable_df[TOKEN].values)
                for token in tokens_text:
                    # tokens[t_id] = {"text": token_dict[t_id]["text"], "sent": token_dict[t_id]["sent"], "id": token_dict[t_id]["id"]}
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
                        for token in doc[max(0, t_id-tolerance): t_id+tolerance+1]:
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
                    found_mentions_tokens = doc[min([t for t in token_found.values()]): max([t for t in token_found.values()]) + 1]
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
                                "spacy_sentence_tokens": [(i,t.text) for i,t in enumerate(doc)],
                                "tolerance": tolerance
                            }

                        LOGGER.warning(
                            f"Mention with ID {doc_id}/{mention_id} ({token_str}) needs manual review. Could not "
                            f"determine the mention head automatically. {str(tolerance)}")

                token_mention_start_id = list(doc_df.index).index(f'{doc_id}/{word_start}')
                context_min_id = 0 if token_mention_start_id-CONTEXT_RANGE < 0 else token_mention_start_id-CONTEXT_RANGE
                context_max_id = min(token_mention_start_id + CONTEXT_RANGE, len(doc_df))

                mention_context_str = list(doc_df.iloc[context_min_id:context_max_id][TOKEN].values)

                # add to mentions if the variables are correct ( do not add for manual review needed )
                if f'{doc_id}/{mention_id}' not in need_manual_review_mention_head:
                    mention_ner =  mention_head.ent_type_ if mention_head.ent_type_ != "" else "O"
                    coref_pre_dict[f'{doc_id}/{coref_id}/{mention_id}'] = {
                        COREF_CHAIN: None,
                               MENTION_NER: mention_ner,
                               MENTION_HEAD_POS: mention_head.pos_,
                               MENTION_HEAD_LEMMA: mention_head.lemma_,
                               MENTION_HEAD: mention_head.text,
                               MENTION_HEAD_ID: mention_head_id,
                               DOC_ID: doc_id,
                               DOC: doc_id,
                               IS_CONTINIOUS: token_ids == list(
                                   range(token_ids[0], token_ids[-1] + 1)),
                               IS_SINGLETON: len(tokens) == 1,
                               MENTION_ID: mention_id,
                               MENTION_TYPE: None,
                               MENTION_FULL_TYPE: None,
                               SCORE: -1.0,
                               SENT_ID: sent_id,
                               MENTION_CONTEXT: mention_context_str,
                               TOKENS_NUMBER: token_ids,
                               TOKENS_STR: token_str,
                               TOKENS_TEXT: tokens_text,
                               TOPIC_ID: topic_subtopic_doc.split("/")[0],
                               TOPIC: topic_name,
                               SUBTOPIC_ID: topic_subtopic_doc.split("/")[1],
                               SUBTOPIC: subtopic_name_composite_full,
                               COREF_TYPE: IDENTITY,
                               DESCRIPTION: None,
                               CONLL_DOC_KEY: topic_subtopic_doc
                               }
                    coref_pre_df = pd.concat([coref_pre_df, pd.DataFrame({
                        COREF_CHAIN: coref_id,
                        DOC_ID: doc_id,
                        MENTION_ID: mention_id,
                        TOKENS_STR: token_str,
                        MENTION_HEAD: mention_head.text,
                        MENTION_HEAD_POS: mention_head.pos_,
                        MENTION_NER: mention_ner
                    }, index=[mention_id])], axis=0)

        grouped_dfs = coref_pre_df.groupby([COREF_CHAIN, DOC_ID])
        cand_chains_df = pd.DataFrame(np.zeros((len(grouped_dfs), len(grouped_dfs))),
                                      index=[f'{doc_id_1}/{coref_chain_orig_id_1}'
                                             for (coref_chain_orig_id_1, doc_id_1), group_df_1 in grouped_dfs],
                                      columns=[f'{doc_id_1}/{coref_chain_orig_id_1}'
                                               for (coref_chain_orig_id_1, doc_id_1), group_df_1 in grouped_dfs])

        LOGGER.info("Building matrix of the cross-document chains overlap...")
        for i, ((coref_chain_orig_id_1, doc_id_1), group_df_1) in tqdm(list(enumerate(grouped_dfs))):
            # no pronouns
            cand_df_1 = group_df_1[group_df_1[MENTION_HEAD_POS] != "PRON"]
            # check full mentions and their heads
            mentions_n_heads_1 = set(cand_df_1[TOKENS_STR].values).union(set(cand_df_1[MENTION_HEAD].values))

            for j, ((coref_chain_orig_id_2, doc_id_2), group_df_2) in enumerate(grouped_dfs):
                cand_df_2 = group_df_2[group_df_2[MENTION_HEAD_POS] != "PRON"]
                if j <= i:
                    continue

                # check full mentions and their heads
                mentions_n_heads_2 = set(cand_df_2[TOKENS_STR].values).union(set(cand_df_2[MENTION_HEAD].values))
                overlap = mentions_n_heads_1.intersection(mentions_n_heads_2)
                overlap_size = len(overlap)

                if len(overlap) == 1:
                    # handle a special case when there is only one overlap and it is a proper noun: give a bit bigger
                    # size to pass the upcoming check
                    if list(overlap)[0][0].isupper():
                        overlap_size = 1.5
                cand_chains_df.loc[f'{doc_id_1}/{coref_chain_orig_id_1}', f'{doc_id_2}/{coref_chain_orig_id_2}'] = overlap_size
                cand_chains_df.loc[ f'{doc_id_2}/{coref_chain_orig_id_2}', f'{doc_id_1}/{coref_chain_orig_id_1}',] = overlap_size

        chain_dict = {}
        checked_sets = set()
        for col in cand_chains_df.columns:
            if col in checked_sets:
                continue

            to_check = {col}
            # first: candidates that are similar to the seed chain
            to_check = to_check.union(set(cand_chains_df[cand_chains_df[col] > 1].index))
            # second: candidates that are similar to the similar chains to the seed chain
            to_check = to_check.union(set(cand_chains_df.loc[list(to_check)][cand_chains_df.loc[list(to_check)][col] > 1].index))

            sim_mentions_df = pd.DataFrame()
            for cand_key in to_check:
                if cand_key in checked_sets:
                    continue

                doc_id, coref_chain = cand_key.split("/")
                sim_mentions_df = pd.concat([sim_mentions_df,
                                             coref_pre_df[(coref_pre_df[DOC_ID] == doc_id) & (coref_pre_df[COREF_CHAIN] == coref_chain)]], axis=0)
            # ignore chains that consist only of pronouns
            non_pron_df = sim_mentions_df[sim_mentions_df[MENTION_HEAD_POS] != "PRON"]
            if not len(non_pron_df):
                continue

            mention_type_df = non_pron_df.groupby(MENTION_NER).count()
            if len(mention_type_df) == 1:
                mention_type = mention_type_df[COREF_CHAIN].idxmax()
            else:
                mention_type_df = mention_type_df.sort_values(COREF_CHAIN, ascending=False)
                if mention_type_df.index[0] != "O":
                    mention_type = mention_type_df.index[0]
                else:
                    # next best
                    mention_type = mention_type_df.index[1]

            mention_type = mention_type if mention_type != "O" else "OTHER"
            description = non_pron_df.groupby(TOKENS_STR).count()[COREF_CHAIN].idxmax()
            chain_id = f'{mention_type[:3]}{shortuuid.uuid()}'
            for mention_index, mention_row in sim_mentions_df.iterrows():
                mention = coref_pre_dict[f'{mention_row[DOC_ID]}/{mention_row[COREF_CHAIN]}/{mention_index}']
                # overwrite attributes related to the chain properties
                mention[MENTION_TYPE] = mention_type[:3]
                mention[MENTION_FULL_TYPE] = mention_type
                mention[DESCRIPTION] = description
                mention[IS_SINGLETON] = len(sim_mentions_df) == 1
                mention[COREF_CHAIN] = chain_id
                if chain_id not in chain_dict:
                    chain_dict[chain_id] = []
                chain_dict[chain_id].append(mention)

                # np4e only has entities
                entity_mentions_local.append(mention)
            checked_sets = checked_sets.union(to_check)


        entity_mentions.extend(entity_mentions_local)

    df_all_mentions = pd.DataFrame()
    for mention in tqdm(entity_mentions + event_mentions):
        df_all_mentions = pd.concat([df_all_mentions, pd.DataFrame({
            attr: str(value) if type(value) == list else value for attr, value in mention.items()
        }, index=[mention[MENTION_ID]])], axis=0)
    df_all_mentions.to_csv(os.path.join(OUT_PATH, MENTIONS_ALL_CSV))

    conll_df = conll_df.reset_index(drop=True)
    conll_df_labeled = make_save_conll(conll_df, df_all_mentions, OUT_PATH)

    with open(os.path.join(out_path, MENTIONS_ENTITIES_JSON), "w") as file:
        json.dump(entity_mentions, file)

    with open(os.path.join(out_path, MENTIONS_EVENTS_JSON), "w") as file:
        json.dump(event_mentions, file)

    LOGGER.info(
        "Mentions that need manual review to define the head and its attributes have been saved to: " +
        MANUAL_REVIEW_FILE + " - Total: " + str(len(need_manual_review_mention_head)))

    with open(os.path.join(TMP_PATH, MANUAL_REVIEW_FILE), "w", encoding='utf-8') as file:
        json.dump(need_manual_review_mention_head, file)


    LOGGER.info("Splitting the dataset into train/val/test subsets...")

    with open("train_val_test_split.json", "r") as file:
        train_val_test_dict = json.load(file)

    for split, subtopic_ids in train_val_test_dict.items():
        conll_df_split = pd.DataFrame()
        for subtopic_id in subtopic_ids:
            conll_df_split = pd.concat([conll_df_split,
                                        conll_df_labeled[conll_df_labeled[TOPIC_SUBTOPIC_DOC].str.contains(f'0/{subtopic_id}')]])
        event_mentions_split = [m for m in event_mentions if any([subtopic_id in m[SUBTOPIC_ID] for subtopic_id in subtopic_ids])]
        entity_mentions_split = [m for m in entity_mentions if any([subtopic_id in m[SUBTOPIC_ID] for subtopic_id in subtopic_ids])]

        output_folder_split = os.path.join(OUT_PATH, split)
        if not os.path.exists(output_folder_split):
            os.mkdir(output_folder_split)

        with open(os.path.join(output_folder_split, MENTIONS_EVENTS_JSON), 'w', encoding='utf-8') as file:
            json.dump(event_mentions_split, file)

        with open(os.path.join(output_folder_split, MENTIONS_ENTITIES_JSON), 'w', encoding='utf-8') as file:
            json.dump(entity_mentions_split, file)

        make_save_conll(conll_df_split, event_mentions_split+entity_mentions_split, output_folder_split, assign_reference_labels=False)

    LOGGER.info(f'Done! \nNumber of unique mentions: {len(df_all_mentions)} '
                f'\nNumber of unique chains: {len(set(df_all_mentions[COREF_CHAIN].values))} ')

    LOGGER.info(f'Parsing of NP4E done!')


if __name__ == '__main__':
    conv_files()
