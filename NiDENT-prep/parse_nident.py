import xml.etree.ElementTree as ET
import os
import json
import string
import copy
import re
import pandas as pd
import numpy as np
import shortuuid
from nltk import Tree
import spacy
import sys
from tqdm import tqdm
from setup import *
from utils import *
from logger import LOGGER

NIDENT_PARSING_FOLDER = os.path.join(os.getcwd())
OUT_PATH = os.path.join(NIDENT_PARSING_FOLDER, OUTPUT_FOLDER_NAME)
nlp = spacy.load('en_core_web_sm')
source_path_nident = os.path.join(NIDENT_PARSING_FOLDER, NIDENT_FOLDER_NAME, "english-corpus")
source_path_np4e = os.path.join(NIDENT_PARSING_FOLDER, "..", NP4E, NP4E_FOLDER_NAME)

subtopic_fullname_dict = {
    "bukavu": "Bukavu_bombing",
    "peru": "Peru_hostages",
    "tajikistan": "Tajikistan_hostages",
    "israel": "Israel_suicide_bomb",
    "china": "China-Taiwan_hijack"
}
identity_dict = {"3": IDENTITY, "2": NEAR_IDENTITY}
replace_id = 0


def get_word_leaves(node) -> List[Tuple[str]]:
    global replace_id
    words = []
    if len(node):
        for child in node:
            words.extend(get_word_leaves(child))
    else:
        if "wdid" in node.attrib:
            w_id = node.get("wdid")
        else:
            w_id = f'P{replace_id}'
            node.attrib["wdid"] = w_id
            replace_id += 1
        words.append((node.get("wd"), w_id))
    return words


def get_entity_values(node) -> dict:
    mentions = {}
    words = []
    for child in node:
        if child.tag == "sn":
            child_mentions = get_entity_values(child)
            mentions.update(child_mentions)
            words.extend([w for v in child_mentions.values() for w in v["words"] ])
        else:
            words.append((child.get("wd"), child.get("wdid")))
    mentions[node.get("markerid")] = {
        "words": words,
        "coref_type": node.get("identdegree", "3"),
        "entity": node.get("entity")
    }
    return mentions


def conv_files():
    conll_df = pd.DataFrame()
    coref_pre_df = pd.DataFrame()
    need_manual_review_mention_head = {}
    coref_pre_dict = {}
    topic_name = "0_bomb_explosion_kidnap"
    topic_id = "0"
    entity_mentions = []

    if not os.path.exists(source_path_np4e):
        raise FileExistsError(f'NP4E dataset, which is required for the subtopic structure, doesn\'t exist! '
                              f'Follow \"setup.py\" to get the files of NP4E first. ')
    # get the subtopic structure
    suptopic_structure_dict = {}
    for subt_id, suptopic_folder in enumerate(os.listdir(source_path_np4e)):
        for subtopic_file in os.listdir(os.path.join(source_path_np4e, suptopic_folder)):
            if not subtopic_file.endswith(".mmax"):
                continue

            doc_id = re.sub("\D+", "", subtopic_file)
            suptopic_structure_dict[doc_id] = f'{subt_id}_{suptopic_folder}'

    for doc_file_name in tqdm(os.listdir(source_path_nident)):
        tree = ET.parse(os.path.join(source_path_nident, doc_file_name))
        root = tree.getroot()
        mention_dict = {}
        doc_id = re.sub("\D+", "", doc_file_name.split("_")[-1])
        subtopic_id = suptopic_structure_dict[doc_id]
        subtopic_name_composite_full = f'{subtopic_fullname_dict[subtopic_id.split("_")[-1]]}'
        topic_subtopic_doc = f'{topic_id}/{subtopic_id}/{doc_id}'

        for sent_id, sentence_elem in enumerate(root):
            token_id = 0
            for subelem in sentence_elem:
                # read tokens of the documents
                words = get_word_leaves(subelem)
                for word, w_id in words:
                    conll_df = pd.concat([conll_df,pd.DataFrame({
                        TOPIC_SUBTOPIC_DOC: topic_subtopic_doc,
                        DOC_ID: doc_id,
                        SENT_ID: sent_id,
                        TOKEN_ID: token_id,
                        TOKEN: word,
                        REFERENCE: "-"
                    }, index=[f'{doc_id}/{w_id}'])])
                    token_id += 1

            # collect mentions and entities
            for subelem in sentence_elem:
                if subelem.tag == "sn":
                    mention_dict.update(get_entity_values(subelem))

        doc_df = conll_df[conll_df[DOC_ID] == doc_id]

        for markable_id, mention_orig in mention_dict.items():
            coref_type = mention_orig["coref_type"]

            if coref_type == "1":
                # exclude non-identity
                continue

            word_start = mention_orig["words"][0][1]
            word_end = mention_orig["words"][-1][1]
            coref_id = mention_orig["entity"]
            markable_df = conll_df.loc[f'{doc_id}/{word_start}': f'{doc_id}/{word_end}']
            if not len(markable_df):
                continue

            # mention attributes
            token_ids = [int(t) for t in list(markable_df[TOKEN_ID].values)]
            tokens = {}
            token_str = ""
            tokens_text = list(markable_df[TOKEN].values)
            for token in tokens_text:
                token_str, word_fixed, no_whitespace = append_text(token_str, token)

            mention_id = shortuuid.uuid(f'{coref_id}{token_str}{markable_id}')

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

            token_mention_start_id = list(doc_df.index).index(f'{doc_id}/{word_start}')
            context_min_id = 0 if token_mention_start_id - CONTEXT_RANGE < 0 else token_mention_start_id - CONTEXT_RANGE
            context_max_id = min(token_mention_start_id + CONTEXT_RANGE, len(doc_df))

            mention_context_str = list(doc_df.iloc[context_min_id:context_max_id][TOKEN].values)

            # add to mentions if the variables are correct ( do not add for manual review needed )
            if f'{doc_id}/{mention_id}' not in need_manual_review_mention_head:
                mention_ner = mention_head.ent_type_ if mention_head.ent_type_ != "" else "O"

                if subtopic_id not in coref_pre_dict:
                    coref_pre_dict[subtopic_id] = {}

                coref_pre_dict[subtopic_id][f'{doc_id}/{coref_id}/{mention_id}'] = {
                    COREF_CHAIN: None,
                    MENTION_NER: mention_ner,
                    MENTION_HEAD_POS: mention_head.pos_,
                    MENTION_HEAD_LEMMA: mention_head.lemma_,
                    MENTION_HEAD: mention_head.text,
                    MENTION_HEAD_ID: int(mention_head_id),
                    DOC_ID: doc_id,
                    DOC: doc_id,
                    IS_CONTINIOUS: token_ids == list(
                        range(token_ids[0], token_ids[-1] + 1)),
                    IS_SINGLETON: len(tokens) == 1,
                    MENTION_ID: mention_id,
                    MENTION_TYPE: None,
                    MENTION_FULL_TYPE: None,
                    SCORE: -1.0,
                    SENT_ID: int(sent_id),
                    MENTION_CONTEXT: mention_context_str,
                    TOKENS_NUMBER: [int(t) for t in token_ids],
                    TOKENS_STR: token_str,
                    TOKENS_TEXT: tokens_text,
                    TOPIC_ID: topic_subtopic_doc.split("/")[0],
                    TOPIC: topic_name,
                    SUBTOPIC_ID: topic_subtopic_doc.split("/")[1],
                    SUBTOPIC: subtopic_name_composite_full,
                    COREF_TYPE: identity_dict[coref_type],
                    DESCRIPTION: None,
                    CONLL_DOC_KEY: topic_subtopic_doc
                }
                coref_pre_df = pd.concat([coref_pre_df, pd.DataFrame({
                    SUBTOPIC_ID: subtopic_id,
                    COREF_CHAIN: coref_id,
                    DOC_ID: doc_id,
                    MENTION_ID: mention_id,
                    TOKENS_STR: token_str,
                    MENTION_HEAD: mention_head.text,
                    MENTION_HEAD_POS: mention_head.pos_,
                    MENTION_NER: mention_ner
                }, index=[mention_id])], axis=0)

    for subtopic_id, mentions in coref_pre_dict.items():
        LOGGER.info(f'Reconstructing CDCR chains for {subtopic_id} subtopic...')
        coref_pre_df_subtopic = coref_pre_df[coref_pre_df[SUBTOPIC_ID] == subtopic_id]
        grouped_dfs = coref_pre_df_subtopic.groupby([COREF_CHAIN, DOC_ID])
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
        for col in tqdm(cand_chains_df.columns):
            if col in checked_sets:
                continue

            to_check = {col}
            # first: candidates that are similar to the seed chain
            to_check = to_check.union(set(cand_chains_df[cand_chains_df[col] >= 2].index))
            # second: candidates that are similar to the similar chains to the seed chain
            to_check = to_check.union(set(cand_chains_df.loc[list(to_check)][cand_chains_df.loc[list(to_check)][col] > 1].index))

            sim_mentions_df = pd.DataFrame()
            for cand_key in to_check:
                if cand_key in checked_sets:
                    continue

                doc_id, coref_chain = cand_key.split("/")
                sim_mentions_df = pd.concat([sim_mentions_df,
                                             coref_pre_df[(coref_pre_df[DOC_ID] == doc_id)
                                                          & (coref_pre_df[COREF_CHAIN] == coref_chain)]], axis=0)
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
                mention = coref_pre_dict[subtopic_id][f'{mention_row[DOC_ID]}/{mention_row[COREF_CHAIN]}/{mention_index}']
                # overwrite attributes related to the chain properties
                mention[MENTION_TYPE] = mention_type[:3]
                mention[MENTION_FULL_TYPE] = mention_type
                mention[DESCRIPTION] = description
                mention[COREF_CHAIN] = chain_id
                if chain_id not in chain_dict:
                    chain_dict[chain_id] = []
                chain_dict[chain_id].append(mention)

                # nident only has entities
                entity_mentions.append(mention)
            checked_sets = checked_sets.union(to_check)

    event_mentions = []
    df_all_mentions = pd.DataFrame()
    for mention in tqdm(entity_mentions + event_mentions):
        df_all_mentions = pd.concat([df_all_mentions, pd.DataFrame({
            attr: str(value) if type(value) == list else value for attr, value in mention.items()
        }, index=[mention[MENTION_ID]])], axis=0)
    df_all_mentions.to_csv(os.path.join(OUT_PATH, MENTIONS_ALL_CSV))

    conll_df = conll_df.reset_index(drop=True)
    make_save_conll(conll_df, df_all_mentions, OUT_PATH)

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
    LOGGER.info(f'Parsing of NiDENT done!')


if __name__ == '__main__':

    LOGGER.info(f"Processing NiDENT...")
    conv_files()


