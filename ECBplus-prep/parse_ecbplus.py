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
from utils import *
from nltk import Tree
from tqdm import tqdm
import warnings
import shortuuid
from setup import *
from logger import LOGGER

warnings.filterwarnings('ignore')

ECB_PARSING_FOLDER = os.path.join(os.getcwd())
ECBPLUS_FILE = "ecbplus.xml"
ECB_FILE = "ecb.xml"
IS_TEXT, TEXT = "is_text", TEXT

source_path = os.path.join(ECB_PARSING_FOLDER, ECBPLUS_FOLDER_NAME)
result_path = os.path.join(ECB_PARSING_FOLDER, OUTPUT_FOLDER_NAME, "test_parsing")
out_path = os.path.join(ECB_PARSING_FOLDER, OUTPUT_FOLDER_NAME)
path_sample = os.path.join(os.getcwd(), "..", SAMPLE_DOC_JSON)

nlp = spacy.load("en_core_web_sm")

validated_sentences_df = pd.read_csv(os.path.join(ECB_PARSING_FOLDER, ECBPLUS_FOLDER_NAME,
                                                  "ECBplus_coreference_sentences.csv")).set_index(
    ["Topic", "File", "Sentence Number"])

with open(os.path.join(ECB_PARSING_FOLDER, "train_val_test_split.json"), "r") as file:
    train_dev_test_split_dict = json.load(file)

with open(os.path.join(ECB_PARSING_FOLDER, "subtopic_names.json"), "r") as file:
    subtopic_names_dict = json.load(file)


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


def convert_files(topic_number_to_convert=3, check_with_list=True):
    coref_dics = {}
    selected_topics = os.listdir(source_path)[:topic_number_to_convert]
    conll_df = pd.DataFrame()
    entity_mentions = []
    event_mentions = []
    topic_names = []
    need_manual_review_mention_head = {}
    counter_annotated_mentions = 0
    counter_parsed_mentions = 0

    for topic_id in selected_topics:
        if topic_id == "__MACOSX":
            continue

        # if a file with confirmed sentences
        if os.path.isfile(os.path.join(source_path, topic_id)):
            continue

        LOGGER.info(f'Converting topic {topic_id}')
        diff_folders = {ECB_FILE: [], ECBPLUS_FILE: []}

        # assign the different folders according to the topics in the variable "diff_folders"
        for topic_file in os.listdir(os.path.join(source_path, topic_id)):
            if ECBPLUS_FILE in topic_file:
                diff_folders[ECBPLUS_FILE].append(topic_file)
            else:
                diff_folders[ECB_FILE].append(topic_file)

        for annot_folders in list(diff_folders.values()):
            t_number = annot_folders[0].split(".")[0].split("_")[0]
            t_name = re.search(r'[a-z]+', annot_folders[0].split(".")[0])[0]
            subtopic_id = t_number + t_name
            topic_names.append(subtopic_id)
            coref_dict = {}

            # for every themed-file in "commentated files"
            for topic_file in tqdm(annot_folders):
                doc_id = re.search(r'[\d+]+', topic_file.split(".")[0].split("_")[1])[0]
                topic_subtopic_doc = f'{topic_id}/{subtopic_id}/{doc_id}'

                # import the XML-Datei topic_file
                tree = ET.parse(os.path.join(source_path, topic_id, topic_file))
                root = tree.getroot()

                token_dict, mentions, mentions_map = {}, {}, {}

                t_id = -1
                old_sent = -1
                for elem in root:
                    if elem.tag == "token":
                        try:
                            # # increase t_id value by 1 if the sentence value in the xml element ''equals the value of old_sent
                            if old_sent == int(elem.attrib[SENTENCE]):
                                t_id += 1
                                # else set old_sent to the value of sentence and t_id to 0
                            else:
                                old_sent = int(elem.attrib[SENTENCE])
                                t_id = 0

                            # fill the token-dictionary with fitting attributes
                            token_dict[elem.attrib[T_ID]] = {TEXT: elem.text, SENT: elem.attrib[SENTENCE], NUM: t_id,
                                                             ID: elem.attrib[NUM]}

                            conll_df = conll_df.append(pd.DataFrame({
                                TOPIC_SUBTOPIC_DOC: topic_subtopic_doc,
                                DOC_ID: subtopic_id,
                                SENT_ID: int(token_dict[elem.attrib[T_ID]][SENT]),
                                # TOKEN_ID: int(t_id),
                                TOKEN_ID: int(token_dict[elem.attrib[T_ID]][ID]),
                                TOKEN: token_dict[elem.attrib[T_ID]][TEXT],
                                REFERENCE: "-"
                            }, index=[elem.attrib[T_ID]]))

                        except KeyError as e:
                            LOGGER.warning(f'Value with key {e} not found and will be skipped from parsing.')

                    if elem.tag == "Markables":
                        for i, subelem in enumerate(elem):
                            mention_tokens_ids_global = [token.attrib[T_ID] for token in subelem]
                            mention_tokens_ids_global.sort(key=int)  # sort tokens by their id
                            doc_df = conll_df[(conll_df[TOPIC_SUBTOPIC_DOC] == topic_subtopic_doc)]

                            # if "tokens" has values -> fill the "mention" dict with the value of the corresponding m_id
                            if len(mention_tokens_ids_global):
                                mention_id = shortuuid.uuid()
                                if len(set(mention_tokens_ids_global)) < len(mention_tokens_ids_global):
                                    mention_tokens_ids_global = sorted(list(set(mention_tokens_ids_global)))

                                sent_id = int(token_dict[mention_tokens_ids_global[0]][SENT])
                                counter_annotated_mentions += 1

                                # generate sentence doc with spacy
                                sentence_str = ""
                                sent_tokens = list(doc_df[doc_df[SENT_ID] == sent_id][TOKEN])
                                for t in sent_tokens:
                                    sentence_str, _, _ = append_text(sentence_str, t)
                                doc = nlp(sentence_str)

                                # tokenize the mention text
                                tokens_text, token_ids = [], []
                                tokens_str = ""
                                for t_id in mention_tokens_ids_global:
                                    tokens_text.append(token_dict[t_id][TEXT])
                                    token_ids.append(int(token_dict[t_id][NUM]))
                                    tokens_str, _, _ = append_text(tokens_str, token_dict[t_id][TEXT])

                                tolerance = 0
                                token_found = {t: None for t in token_ids}
                                prev_id = 0
                                for t_id, t in zip(token_ids, tokens_text):
                                    to_break = False
                                    for tolerance in range(10):
                                        for token in doc[max(0, t_id - tolerance): t_id + tolerance + 1]:
                                            if token.i < prev_id-1:
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
                                            elif token.text.endswith(t):
                                                token_found[t_id] = token.i
                                                prev_id = token.i
                                                to_break = True
                                                break
                                        if to_break:
                                            break

                                # whole mention string processed, look for the head
                                mention_head_id = token_ids[0]
                                mention_head = None
                                if mention_id not in need_manual_review_mention_head:
                                    found_mentions_tokens_ids = list([t for t in token_found.values() if t is not None])
                                    found_mentions_tokens = []
                                    if len(found_mentions_tokens_ids):
                                        found_mentions_tokens = doc[min(found_mentions_tokens_ids): max(
                                            found_mentions_tokens_ids) + 1]
                                        if len(found_mentions_tokens) == 1:
                                            mention_head = found_mentions_tokens[0]
                                            # remap the mention head back to the np4e original tokenization to get the ID for the output
                                            for t_orig, t_mapped in token_found.items():
                                                if t_mapped == mention_head.i:
                                                    mention_head_id = t_orig
                                                    break

                                    if mention_head is None:
                                        # found_mentions_tokens_ids = set([t.i for t in found_mentions_tokens])
                                        for i, t in enumerate(found_mentions_tokens):
                                            if t.head.i == t.i:
                                                # if a token is a root, it is a candidate for the head
                                                pass

                                            elif t.head.i >= min(found_mentions_tokens_ids) and t.head.i <= max(
                                                    found_mentions_tokens_ids):
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
                                if mention_head is None:  # also "if is None"
                                    if mention_id not in need_manual_review_mention_head:
                                        need_manual_review_mention_head[mention_id] = \
                                            {
                                                "mention_text": list(zip(token_ids, tokens_text)),
                                                "sentence_tokens": list(enumerate(sent_tokens)),
                                                "spacy_sentence_tokens": [(i, t.text) for i, t in enumerate(doc)],
                                                "tolerance": tolerance
                                            }
                                        LOGGER.warning(
                                            f"Mention with ID {doc_id}/{mention_id} ({tokens_str}) needs manual review. Could not "
                                            f"determine the mention head automatically. {str(tolerance)}")

                                doc_df.loc[:, "token_id_global"] = list(range(len(doc_df)))

                                token_mention_start_id = doc_df[(doc_df[SENT_ID] == sent_id) & (doc_df[TOKEN_ID] == token_ids[0])]["token_id_global"][0]
                                if token_mention_start_id - CONTEXT_RANGE < 0:
                                    context_min_id = 0
                                    tokens_number_context = list(
                                        doc_df[(doc_df[SENT_ID] == sent_id) & (doc_df[TOKEN_ID].isin(token_ids))][
                                            "token_id_global"])
                                else:
                                    context_min_id = token_mention_start_id - CONTEXT_RANGE
                                    global_token_ids = list(
                                        doc_df[(doc_df[SENT_ID] == sent_id) & (doc_df[TOKEN_ID].isin(token_ids))][
                                            "token_id_global"])
                                    tokens_number_context = [int(t - context_min_id) for t in global_token_ids]

                                context_max_id = min(token_mention_start_id + CONTEXT_RANGE, len(doc_df))
                                mention_context_str = list(doc_df.iloc[context_min_id:context_max_id][TOKEN].values)

                                if mention_id not in need_manual_review_mention_head:
                                    mention_ner = mention_head.ent_type_ if mention_head.ent_type_ != "" else "O"
                                    mentions[subelem.attrib[M_ID]] = {MENTION_TYPE: subelem.tag,
                                                                      MENTION_FULL_TYPE: subelem.tag,
                                                                      TOKENS_STR: tokens_str.strip(),
                                                                      MENTION_ID: mention_id,
                                                                      MENTION_NER: mention_ner,
                                                                      MENTION_HEAD_POS: mention_head.pos_,
                                                                      MENTION_HEAD_LEMMA: mention_head.lemma_,
                                                                      MENTION_HEAD: mention_head.text,
                                                                      MENTION_HEAD_ID: mention_head_id,
                                                                      TOKENS_NUMBER: token_ids,
                                                                      TOKENS_TEXT: tokens_text,
                                                                      DOC: topic_file.split(".")[0],
                                                                      SENT_ID: sent_id,
                                                                      MENTION_CONTEXT: mention_context_str,
                                                                      TOKENS_NUMBER_CONTEXT: tokens_number_context,
                                                                      TOPIC_SUBTOPIC_DOC: topic_subtopic_doc,
                                                                      TOPIC: topic_subtopic_doc}
                                    counter_parsed_mentions += 1
                            else:
                                # form coreference chain
                                # m_id points to the target
                                if "ent_type" in subelem.attrib:
                                    mention_type_annot = subelem.attrib["ent_type"]
                                elif "class" in subelem.attrib:
                                    mention_type_annot = subelem.attrib["class"]
                                elif "type" in subelem.attrib:
                                    mention_type_annot = subelem.attrib["type"]
                                else:
                                    mention_type_annot = subelem.tag

                                if "instance_id" in subelem.attrib:
                                    id_ = subelem.attrib["instance_id"]
                                else:
                                    descr = subelem.attrib["TAG_DESCRIPTOR"]
                                    id_ = ""

                                    for coref_id, coref_vals in coref_dict.items():
                                        if coref_vals[DESCRIPTION] == descr and coref_vals[COREF_TYPE] == mention_type_annot \
                                                and coref_vals["subtopic"] == subtopic_id and mention_type_annot:
                                            id_ = coref_id
                                            break

                                    if not len(id_):
                                        LOGGER.warning(
                                            f"Document {doc_id}: {subelem.tag} {subelem.attrib} doesn\'t have attribute instance_id. It will be created")
                                        id_ = mention_type_annot[:3]  + shortuuid.uuid()[:17]

                                    if not len(id_):
                                        continue

                                    subelem.attrib["instance_id"] = id_

                                if not len(id_):
                                    continue

                                if id_ not in coref_dict:
                                    coref_dict[id_] = {DESCRIPTION: subelem.attrib["TAG_DESCRIPTOR"],
                                                       COREF_TYPE: mention_type_annot,
                                                       "subtopic": subtopic_id}

                    if elem.tag == "Relations":
                        # for every false create a false-value in "mentions_map"
                        mentions_map = {m: False for m in list(mentions)}
                        for i, subelem in enumerate(elem):
                            tmp_instance_id = "None"
                            for j, subsubelm in enumerate(subelem):
                                if subsubelm.tag == "target":
                                    for prevelem in root:
                                        if prevelem.tag != "Markables":
                                            continue

                                        for k, prevsubelem in enumerate(prevelem):
                                            if prevsubelem.get("instance_id") is None:
                                                continue

                                            if subsubelm.attrib["m_id"] == prevsubelem.attrib["m_id"]:
                                                tmp_instance_id = prevsubelem.attrib["instance_id"]
                                                break

                            if tmp_instance_id != "None":
                                try:
                                    if "r_id" not in coref_dict[tmp_instance_id]:
                                        coref_dict[tmp_instance_id].update({
                                            "r_id": subelem.attrib["r_id"],
                                            # "coref_type": subelem.tag,
                                            "mentions": {mentions[m.attrib["m_id"]][MENTION_ID]: mentions[m.attrib["m_id"]]
                                                         for m in subelem if
                                                         m.tag == "source"}
                                        })
                                    else:
                                        for m in subelem:
                                            if m.tag == "source":
                                                mention_id_local = mentions[m.attrib["m_id"]][MENTION_ID]
                                                if mention_id_local in coref_dict[tmp_instance_id]["mentions"]:
                                                    continue

                                                coref_dict[tmp_instance_id]["mentions"][mention_id_local] = mentions[
                                                    m.attrib["m_id"]]
                                except KeyError as e:
                                    LOGGER.warning(
                                        f'Document {doc_id}: Mention with ID {str(e)} is not among the Markables and will be skipped.')
                            for m in subelem:
                                mentions_map[m.attrib[M_ID]] = True

                        for i, (m_id, used) in enumerate(mentions_map.items()):
                            if used:
                                continue

                            m = mentions[m_id]
                            chain_id_created = "Singleton_" + m[MENTION_TYPE][:3] + shortuuid.uuid()[:7]
                            if chain_id_created not in coref_dict:
                                coref_dict[chain_id_created] = {
                                    "r_id": str(10000 + i),
                                    COREF_TYPE: m[MENTION_TYPE],
                                    MENTIONS: {m_id: m},
                                    DESCRIPTION: m[TOKENS_STR],
                                    "subtopic": subtopic_id
                                }
                            else:
                                coref_dict[chain_id_created].update(
                                    {
                                        "r_id": str(10000 + i),
                                        COREF_TYPE: m[MENTION_TYPE],
                                        MENTIONS: {m_id: m},
                                        "subtopic": subtopic_id,
                                        DESCRIPTION:  m[TOKENS_STR],
                                    })

            coref_dics[topic_id] = coref_dict

            entity_mentions_local = []
            event_mentions_local = []
            mentions_local = []

            for chain_id, chain_vals in coref_dict.items():

                if MENTIONS not in chain_vals:
                    continue

                not_unique_heads = []

                for m in chain_vals[MENTIONS].values():
                    sent_id = int(m[SENT_ID])

                    # create variable "mention_id"
                    token_numbers = [int(t) for t in m[TOKENS_NUMBER]]
                    mention_id = f'{m[DOC]}_{str(chain_id)}_{str(m[SENT_ID])}_{str(m[TOKENS_NUMBER][0])}_{shortuuid.uuid()[:4]}'

                    not_unique_heads.append(m[MENTION_HEAD_LEMMA])

                    # create the dict. "mention" with all corresponding values
                    mention = {COREF_CHAIN: chain_id,
                               MENTION_NER: m[MENTION_NER],
                               MENTION_HEAD_POS: m[MENTION_HEAD_POS],
                               MENTION_HEAD_LEMMA: m[MENTION_HEAD_LEMMA],
                               MENTION_HEAD: m[MENTION_HEAD],
                               MENTION_HEAD_ID: m[MENTION_HEAD_ID],
                               DOC_ID: m[TOPIC_SUBTOPIC_DOC].split("/")[-1],
                               DOC: m[DOC],
                               IS_CONTINIOUS: bool(
                                   token_numbers == list(range(token_numbers[0], token_numbers[-1] + 1))),
                               IS_SINGLETON: bool(len(chain_vals[MENTIONS]) == 1),
                               MENTION_ID: mention_id,
                               MENTION_TYPE: m[MENTION_TYPE][:3],
                               MENTION_FULL_TYPE: m[MENTION_TYPE],
                               SCORE: -1.0,
                               SENT_ID: sent_id,
                               MENTION_CONTEXT: m[MENTION_CONTEXT],
                               TOKENS_NUMBER_CONTEXT: m[TOKENS_NUMBER_CONTEXT],
                               TOKENS_NUMBER: token_numbers,
                               TOKENS_STR: m[TOKENS_STR],
                               TOKENS_TEXT: m[TOKENS_TEXT],
                               TOPIC_ID: t_number,
                               TOPIC: t_number,
                               SUBTOPIC_ID: subtopic_id,
                               SUBTOPIC: subtopic_names_dict[subtopic_id],
                               COREF_TYPE: IDENTITY,
                               DESCRIPTION: chain_vals[DESCRIPTION],
                               CONLL_DOC_KEY: m[TOPIC_SUBTOPIC_DOC],
                               }

                    # if the first two entries of chain_id are "ACT" or "NEG", add the "mention" to the array "event_mentions_local"
                    if chain_id[:3] in ["ACT", "NEG"]:
                        event_mentions_local.append(mention)
                    # else add the "mention" to the array "event_mentions_local" and add the following values to the DF "summary_df"
                    else:
                        entity_mentions_local.append(mention)

                    if not mention[IS_SINGLETON]:
                        mentions_local.append(mention)
            entity_mentions.extend(entity_mentions_local)
            event_mentions.extend(event_mentions_local)

    if len(need_manual_review_mention_head):
        LOGGER.warning(f'Mentions ignored: {len(need_manual_review_mention_head)}. The ignored mentions are available here for a manual review: '
                    f'{os.path.join(out_path,MANUAL_REVIEW_FILE)}')
        with open(os.path.join(TMP_PATH, MANUAL_REVIEW_FILE), "w", encoding='utf-8') as file:
            json.dump(need_manual_review_mention_head, file)

    # take only validated sentences
    entity_mention_validated = []
    for mention in entity_mentions:
        subtopic_suf = re.sub(r'\d+', '', mention[SUBTOPIC_ID])
        if (int(mention[TOPIC_ID]), f"{mention[DOC_ID]}{subtopic_suf}", mention[SENT_ID]) not in validated_sentences_df.index:
            continue
        entity_mention_validated.append(mention)

    # take only validated sentences
    event_mention_validated = []
    for mention in event_mentions:
        subtopic_suf = re.sub(r'\d+', '', mention[SUBTOPIC_ID])
        if (int(mention[TOPIC_ID]), f"{mention[DOC_ID]}{subtopic_suf}", mention[SENT_ID]) not in validated_sentences_df.index:
            continue
        event_mention_validated.append(mention)

    for save_options in [
        # [entity_mentions, event_mentions, os.path.join(ECB_PARSING_FOLDER, OUTPUT_FOLDER_NAME+"-unvalidated")],
                         [entity_mention_validated, event_mention_validated, out_path]]:
        entity_m, event_m, save_folder = save_options
        LOGGER.info(f'Saving ECB+ into {save_folder}...')

        with open(os.path.join(save_folder, MENTIONS_ENTITIES_JSON), "w", encoding='utf-8') as file:
            json.dump(entity_m, file)

        with open(os.path.join(save_folder, MENTIONS_EVENTS_JSON), "w", encoding='utf-8') as file:
            json.dump(event_m, file)

        # create a csv. file out of the mentions summary_df
        df_all_mentions = pd.DataFrame()
        for mention in entity_m + event_m:
            df_all_mentions = pd.concat([df_all_mentions, pd.DataFrame({
                attr: str(value) if type(value) == list else value for attr, value in mention.items()
            }, index=[mention[MENTION_ID]])], axis=0)

        df_all_mentions.to_csv(os.path.join(save_folder, MENTIONS_ALL_CSV))

        conll_df_labels = make_save_conll(conll_df, df_all_mentions, save_folder)

        LOGGER.info(f'Done! \nNumber of unique mentions: {len(df_all_mentions)} '
                    f'\nNumber of unique chains: {len(set(df_all_mentions[COREF_CHAIN].values))} ')

        LOGGER.info(f'Splitting ECB+ into train/dev/test subsets...')
        for subset, topic_ids in train_dev_test_split_dict.items():
            LOGGER.info(f'Creating data for {subset} subset...')
            split_folder = os.path.join(save_folder, subset)
            if subset not in os.listdir(save_folder):
                os.mkdir(split_folder)

            selected_entity_mentions = []
            for mention in entity_m:
                if int(mention[TOPIC_ID]) in topic_ids:
                    selected_entity_mentions.append(mention)

            selected_event_mentions = []
            for mention in event_m:
                if int(mention[TOPIC_ID]) in topic_ids:
                    selected_event_mentions.append(mention)

            with open(os.path.join(split_folder, MENTIONS_ENTITIES_JSON), "w", encoding='utf-8') as file:
                json.dump(selected_entity_mentions, file)

            with open(os.path.join(split_folder, MENTIONS_EVENTS_JSON), "w", encoding='utf-8') as file:
                json.dump(selected_event_mentions, file)

            conll_df_split = pd.DataFrame()
            for t_id in topic_ids:
                conll_df_split = pd.concat([conll_df_split,
                                            conll_df_labels[conll_df_labels[TOPIC_SUBTOPIC_DOC].str.contains(f'{t_id}/')]], axis=0)
            make_save_conll(conll_df_split, selected_event_mentions+selected_entity_mentions, split_folder, False)
    LOGGER.info(f'Parsing ECB+ is done!')
    LOGGER.info(f'The annotated mentions ({counter_annotated_mentions}) and parsed mentions ({counter_parsed_mentions}).')


# main function for the input which topics of the ecb corpus are to be converted
if __name__ == '__main__':
    topic_num = 45
    convert_files(topic_num)
    LOGGER.info("\nConversion of {0} topics from xml to newsplease format and to annotations in a json file is "
                "done. \n\nFiles are saved to {1}. \n.".format(str(topic_num), result_path))
