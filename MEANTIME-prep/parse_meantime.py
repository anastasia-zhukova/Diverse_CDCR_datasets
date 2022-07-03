import xml.etree.ElementTree as ET
import os
import json
import string
import copy
import re
import pandas as pd
import numpy as np
from datetime import datetime
from nltk.corpus import wordnet
import sys
from nltk import Tree
import spacy
from tqdm import tqdm
from setup import *
from insert_whitespace import append_text
from config import DATA_PATH, TMP_PATH
from logger import LOGGER

path_sample = os.path.join(DATA_PATH, "_sample_doc.json")  # ->root/data/original/_sample_doc.json
MEANTIME_PARSING_FOLDER = os.path.join(os.getcwd())
OUT_PATH = os.path.join(MEANTIME_PARSING_FOLDER, OUTPUT_FOLDER_NAME)
CONTEXT_RANGE = 250

nlps = [spacy.load('en_core_web_sm'),
        spacy.load('es_core_news_sm'),
        spacy.load('nl_core_news_sm'),
        spacy.load('it_core_news_sm')]

# opens and loads the newsplease-format out of the json file: _sample_doc.json
with open(path_sample, "r") as file:
    newsplease_format = json.load(file)

import os
source_paths = [os.path.join(MEANTIME_PARSING_FOLDER, MEANTIME_FOLDER_NAME_ENGLISH),
                os.path.join(MEANTIME_PARSING_FOLDER, MEANTIME_FOLDER_NAME_SPANISH),
                os.path.join(MEANTIME_PARSING_FOLDER, MEANTIME_FOLDER_NAME_DUTCH),
                os.path.join(MEANTIME_PARSING_FOLDER, MEANTIME_FOLDER_NAME_ITALIAN)]
result_paths = [os.path.join(OUT_PATH, 'test_parsing_en'),
                os.path.join(OUT_PATH, 'test_parsing_es'),
                os.path.join(OUT_PATH, 'test_parsing_nl'),
                os.path.join(OUT_PATH, 'test_parsing_it')]
out_paths = [   os.path.join(OUT_PATH, 'en'),
                os.path.join(OUT_PATH, 'es'),
                os.path.join(OUT_PATH, 'nl'),
                os.path.join(OUT_PATH, 'it')]

meantime_types = {"PRO": "PRODUCT",
                  "FIN": "FINANCE",
                  "LOC": "LOCATION",
                  "ORG": "ORGANIZATION",
                  "OTH": "OTHER",
                  "PER": "PERSON",
                  "GRA": "GRAMMATICAL",
                  "SPE": "SPEECH_COGNITIVE",
                  "MIX": "MIXTURE"}

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


def conv_files(paths, result_path, out_path, language, nlp):
    doc_files = {}
    entity_mentions = []
    event_mentions = []
    summary_df = pd.DataFrame(columns=[DOC_ID, COREF_CHAIN, DESCRIPTION, MENTION_TYPE, MENTION_FULL_TYPE, TOKENS_STR])
    summary_conversion_df = pd.DataFrame()
    conll_df = pd.DataFrame(columns=[TOPIC_SUBTOPIC, DOC_ID, SENT_ID, TOKEN_ID, TOKEN, REFERENCE])
    final_output_str = ""
    need_manual_review_mention_head = {}

    for path in paths:
        dirs = os.listdir(path)
        cnt = 0

        for topic_dir in dirs:
            LOGGER.info(f"Parsing of {topic_dir} ({language}) [{path[-20:]}]. Please wait...")

            # information_df = pd.DataFrame()

            cnt = cnt + 1
            topic_files = os.listdir(os.path.join(path, topic_dir))
            if "_cross-doc_annotation" in path:
                topic_name = str(cnt) + "MEANTIME" #+ "cross"
                annotation_prefix = "cross_"
                # doc_type = "intra_cross"
            else:
                topic_name = str(cnt) + "MEANTIME"
                annotation_prefix = "intra_"
                # doc_type = "intra"

            coref_dict = {}
            entity_mentions_local = []
            event_mentions_local = []

            for topic_file in tqdm(topic_files):
                tree = ET.parse(os.path.join(path, topic_dir, topic_file))
                root = tree.getroot()
                title, text, date, url, time, time2, time3 = "", "", "", "", "", "", ""

                info_t_name = topic_file.split(".")[0].split("_")[0]
                t_subt = topic_dir + "/" + info_t_name

                token_dict, mentions, mentions_map = {}, {}, {}

                t_id = -1
                old_sent = 0
                sent_dict = {}

                for elem in root:
                    try:
                        if old_sent == int(elem.attrib["sentence"]):
                            t_id += 1
                        else:
                            old_sent = int(elem.attrib["sentence"])
                            t_id = 0
                        token_dict[elem.attrib["t_id"]] = {"text": elem.text, "sent": elem.attrib["sentence"], "id": t_id}

                        if int(elem.attrib["sentence"]) == 0:
                            title, word, no_whitespace = append_text(title, elem.text)
                        elif int(elem.attrib["sentence"]) == 1:
                            title, word, no_whitespace = append_text(date, elem.text)
                        else:
                            title, word, no_whitespace = append_text(text, elem.text)

                        prev_word = ""
                        if t_id >= 0:
                            prev_word = root[t_id - 1].text
                        if annotation_prefix == "cross_":   # do not overwrite conll previously generated
                            if elem.tag == "token" and len(conll_df.loc[(conll_df[TOPIC_SUBTOPIC] == t_subt) & (conll_df[DOC_ID] == topic_name) & (conll_df[SENT_ID] == int(elem.attrib["sentence"])) & (conll_df[TOKEN_ID] == t_id ) ]) < 1:
                                conll_df.loc[len(conll_df)] = {
                                    TOPIC_SUBTOPIC: t_subt,
                                    DOC_ID: topic_name,
                                    SENT_ID: int(elem.attrib["sentence"]),
                                    TOKEN_ID: t_id,
                                    TOKEN: elem.text,
                                    # "after": "\"" + append_text(prev_word, elem.text, "space") + "\"",
                                    REFERENCE: "-"
                                }
                                text, word, no_whitespace = append_text(text, elem.text)

                    except KeyError:
                        pass

                    if elem.tag == "Markables":
                        for i, subelem in enumerate(elem):
                            tokens = [token.attrib[T_ID] for token in subelem]
                            tokens.sort(key=int)   # sort tokens by their id
                            sent_tokens = [int(token_dict[t]["id"]) for t in tokens]

                            # skip if the token is contained more than once within the same mention
                            # (i.e. ignore entries with error in ecb+ tokenization)
                            if len(tokens) != len(list(set(tokens))):
                                continue

                            mention_text = ""
                            for t in tokens:
                                mention_text, _, _ = append_text(mention_text, token_dict[t][TEXT])
                            # if "tokens" has values -> fill the "mention" dict with the value of the corresponding m_id
                            if len(tokens):
                                sent_id = int(token_dict[tokens[0]][SENT])

                                # generate sentence doc with spacy
                                sentence_str = ""
                                for t in root:
                                    if t.tag == TOKEN and t.attrib[SENTENCE] == str(sent_id):
                                        sentence_str, _, _ = append_text(sentence_str, t.text)
                                doc = nlp(sentence_str)

                                # tokenize the mention text
                                mention_tokenized = []
                                for t_id in tokens:
                                    mention_tokenized.append(token_dict[t_id])

                                split_mention_text = re.split(" ", mention_text)

                                # counting character up to the first character of the mention within the sentence
                                if len(split_mention_text) > 1:
                                    first_char_of_mention = sentence_str.find(split_mention_text[0]+" "+split_mention_text[1])  # more accurate finding (reduce error if first word is occurring multiple times (i.e. "the")
                                else:
                                    first_char_of_mention = sentence_str.find(split_mention_text[0])
                                # last character directly behind mention
                                last_char_of_mention = sentence_str.find(split_mention_text[-1], len(sentence_str[
                                                                                                     :first_char_of_mention]) + len(
                                    mention_text) - len(split_mention_text[-1])) + len(
                                    split_mention_text[-1])
                                if last_char_of_mention == 0:  # last char can't be first char of string
                                    # handle special case if the last punctuation is part of mention in ecb
                                    last_char_of_mention = len(sentence_str)

                                #print(str(first_char_of_mention))
                                #print(str(last_char_of_mention))

                                counter = 0
                                while True:
                                    if counter > 50:  # an error must have occurred, so break and add to manual review
                                        #print("Counter too high")
                                        need_manual_review_mention_head[str(t_subt) + "_" + str(mention_text)] = {
                                            "mention_text": mention_text,
                                            "sentence_str": sentence_str,
                                            "mention_head": "unknown",
                                            "mention_tokens_amount": len(tokens),
                                            "tolerance": tolerance
                                        }
                                        LOGGER.info(f"Mention with ID {str(t_subt)}_{str(mention_text)} needs manual review. Could not determine the mention head automatically. {str(tolerance)}")
                                        break

                                    if sentence_str[-1] not in ".!?" or mention_text[-1] == ".":
                                        # if the sentence does not end with a ".", we have to add one
                                        # for the algorithm to understand the sentence.
                                        # (this "." isn't represented in the output later)
                                        sentence_str = sentence_str + "."
                                    char_after_first_token = sentence_str[
                                        first_char_of_mention + len(split_mention_text[0])]

                                    if len(split_mention_text) < len(re.split(" ", sentence_str[
                                                                                   first_char_of_mention:last_char_of_mention])) + 1 and \
                                            (last_char_of_mention >= len(sentence_str) or
                                             sentence_str[last_char_of_mention] in string.punctuation or
                                             sentence_str[last_char_of_mention] == " ") and \
                                            str(sentence_str[first_char_of_mention - 1]) in str(
                                        string.punctuation + " ") and \
                                            char_after_first_token in str(string.punctuation + " "):
                                        # The end of the sentence was reached or the next character is a punctuation

                                        processed_chars = 0
                                        added_spaces = 0
                                        mention_doc_ids = []

                                        # get the tokens within the spacy doc
                                        for t in doc:
                                            processed_chars = processed_chars + len(t.text)
                                            spaces = sentence_str[:processed_chars].count(" ") - added_spaces
                                            added_spaces = added_spaces + spaces
                                            processed_chars = processed_chars + spaces

                                            if last_char_of_mention >= processed_chars >= first_char_of_mention:
                                                # mention token detected
                                                mention_doc_ids.append(t.i)
                                            elif processed_chars > last_char_of_mention:
                                                # whole mention has been processed
                                                break

                                        # allow for dynamic differences in tokenization
                                        # (longer mention texts may lead to more differences)
                                        tolerance = len(tokens) / 2
                                        if tolerance > 2:
                                            tolerance = 2
                                        # tolerance for website mentions
                                        if ".com" in mention_text or ".org" in mention_text:
                                            tolerance = tolerance + 2
                                        # tolerance when the mention has external tokens inbetween mention tokens
                                        tolerance = tolerance \
                                                    + int(tokens[-1]) \
                                                    - int(tokens[0]) \
                                                    - len(tokens) \
                                                    + 1
                                        # increase tolerance for every punctuation included in mention text
                                        tolerance = tolerance + sum(
                                            [1 for c in mention_text if c in string.punctuation])

                                        if abs(len(re.split(" ", sentence_str[
                                                                 first_char_of_mention:last_char_of_mention])) - len(
                                            tokens)) <= tolerance and sentence_str[
                                            first_char_of_mention - 1] in string.punctuation + " " and sentence_str[
                                            last_char_of_mention] in string.punctuation + " ":
                                            # Whole mention found in sentence (and tolerance is OK)
                                            #print("difference OK")
                                            break
                                        else:
                                            counter = counter + 1
                                            #print("Difference to big")
                                            # The next char is not a punctuation, so it therefore it is just a part of a bigger word
                                            first_char_of_mention = sentence_str.find(
                                                re.split(" ", mention_text)[0],
                                                first_char_of_mention + 2)
                                            last_char_of_mention = sentence_str.find(
                                                re.split(" ", mention_text)[-1],
                                                first_char_of_mention + len(
                                                    re.split(" ", mention_text)[0])) + len(
                                                re.split(" ", mention_text)[-1])
                                            #print(first_char_of_mention)
                                            #print(last_char_of_mention)
                                            #print("-")

                                    else:
                                        counter = counter + 1
                                        # The next char is not a punctuation, so it therefore we just see a part of a bigger word
                                        # i.g. do not accept "her" if the next letter is "s" ("herself")
                                        first_char_of_mention = sentence_str.find(re.split(" ", mention_text)[0],
                                                                                  first_char_of_mention + 2)
                                        if len(re.split(" ", mention_text)) == 1:
                                            last_char_of_mention = first_char_of_mention + len(mention_text)
                                        else:
                                            last_char_of_mention = sentence_str.find(re.split(" ", mention_text)[-1],
                                                                                     first_char_of_mention + len(
                                                                                         re.split(" ", mention_text)[
                                                                                             0])) + len(
                                                re.split(" ", mention_text)[-1])
                                        #print("is part of bigger")
                                        #print(first_char_of_mention)
                                        #print(last_char_of_mention)
                                        #print("-")

                                # whole mention string processed, look for the head
                                if str(t_subt) + "_" + str(mention_text) not in need_manual_review_mention_head:
                                    for i in mention_doc_ids:
                                        ancestors_in_mention = 0
                                        for a in doc[i].ancestors:
                                            if a.i in mention_doc_ids:
                                                ancestors_in_mention = ancestors_in_mention + 1
                                                break  # one is enough to make the token inviable as a head
                                        if ancestors_in_mention == 0:
                                            # head within the mention
                                            mention_head = doc[i]
                                else:
                                    mention_head = doc[0]  # as placeholder for manual checking

                                mention_head_lemma = mention_head.lemma_
                                mention_head_pos = mention_head.pos_

                                mention_ner = mention_head.ent_type_
                                if mention_ner == "":
                                    mention_ner = "O"

                                # remap the mention head back to the ecb+ original tokenization to get the ID for the output
                                mention_head_id = None
                                mention_head_text = mention_head.text
                                for t in tokens:
                                    if str(token_dict[t][TEXT]).startswith(mention_head_text):
                                        mention_head_id = token_dict[t][ID]
                                if not mention_head_id and len(tokens) == 1:
                                    mention_head_id = token_dict[tokens[0]][ID]
                                elif not mention_head_id:
                                    for t in tokens:
                                        if mention_head_text.startswith(str(token_dict[t][TEXT])):
                                            mention_head_id = token_dict[str(t)][ID]
                                if not mention_head_id:
                                    for t in tokens:
                                        if str(token_dict[t][TEXT]).endswith(mention_head_text):
                                            mention_head_id = token_dict[str(t)][ID]

                                # add to manual review if the resulting token is not inside the mention
                                # (error must have happened)
                                if mention_head_id not in sent_tokens:  # also "if is None"
                                    if str(t_subt) + "_" + str(mention_text) not in need_manual_review_mention_head:
                                        need_manual_review_mention_head[str(t_subt) + "_" + str(mention_text)] = \
                                            {
                                                "mention_text": mention_text,
                                                "sentence_str": sentence_str,
                                                "mention_head": str(mention_head),
                                                "mention_tokens_amount": len(tokens),
                                                "tolerance": tolerance
                                            }
                                        with open(os.path.join(out_path, MANUAL_REVIEW_FILE.replace(".json", "_"+language+".json")), "w",
                                                  encoding='utf-8') as file:
                                            json.dump(need_manual_review_mention_head, file)
                                        LOGGER.info(f"Mention with ID {str(t_subt)}_{str(mention_text)} needs manual review. Could not determine the mention head automatically. {str(tolerance)}")

                                # get the context
                                tokens_int = [int(x) for x in tokens]
                                context_min_id, context_max_id = [0 if int(min(tokens_int)) - CONTEXT_RANGE < 0 else
                                                                  int(min(tokens_int)) - CONTEXT_RANGE,
                                                                  len(token_dict) - 1
                                                                  if int(max(tokens_int)) + CONTEXT_RANGE > len(
                                                                      token_dict)
                                                                  else int(max(tokens_int)) + CONTEXT_RANGE]

                                mention_context_str = []
                                for t in root:
                                    if t.tag == "token" and int(t.attrib["t_id"]) >= context_min_id and int(t.attrib["t_id"]) <= context_max_id:
                                        mention_context_str.append(t.text)

                                #[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]
                                #print("mention_text " + str(mention_text))
                                #print("sentence_str " + str(sentence_str))
                                #print("mention_head_text " + str(mention_head_text))
                                #print("tokens " + str(tokens))
                                #print("sent_tokens " + str(sent_tokens))
                                #print("mention_head_id " + str(mention_head_id))
                                #print("------")
                                #if mention_head_id not in [13, 20, 15, 14, 15, 3, 48]:
                                #    sys.exit()

                                mentions[subelem.attrib["m_id"]] = {"type": subelem.tag,
                                                                    "text": " ".join(
                                                                        [token_dict[t]["text"] for t in tokens]),
                                                                    "sent_doc": doc,
                                                                    MENTION_NER: mention_ner,
                                                                    MENTION_HEAD_POS: mention_head_pos,
                                                                    MENTION_HEAD_LEMMA: mention_head_lemma,
                                                                    MENTION_HEAD: mention_head.text,
                                                                    MENTION_HEAD_ID: mention_head.i,
                                                                    TOKENS_NUMBER: sent_tokens,
                                                                    #"token_doc_numbers": token_ids_in_doc,
                                                                    DOC_ID: topic_file.split(".")[0],
                                                                    SENT_ID: int(sent_id),
                                                                    "mention_context": mention_context_str,
                                                                    TOPIC_SUBTOPIC: t_subt}

                            else:
                                try:
                                    if subelem.attrib["instance_id"] not in coref_dict:
                                        coref_dict[subelem.attrib["instance_id"]] = {
                                            "descr": subelem.attrib["TAG_DESCRIPTOR"]}
                                    # m_id points to the target
                                except KeyError:
                                    pass
                                    # unknown_tag = subelem.tag
                                    # coref_dict[unknown_tag + subelem.attrib["m_id"]] = {"descr": subelem.attrib["TAG_DESCRIPTOR"]}

                    if elem.tag == "Relations":
                        mentions_map = {m: False for m in list(mentions)}
                        for i, subelem in enumerate(elem):
                            for j, subsubelm in enumerate(subelem):
                                if subsubelm.tag == "target":
                                    for prevelem in root:
                                        if prevelem.tag == "Markables":
                                            for k, prevsubelem in enumerate(prevelem):
                                                if prevsubelem.get("instance_id") is not None:
                                                    if subsubelm.attrib["m_id"] == prevsubelem.attrib["m_id"]:
                                                        tmp_instance_id = prevsubelem.attrib["instance_id"]
                                                else:
                                                    tmp_instance_id = "None"

                            try:
                                if "r_id" not in coref_dict[tmp_instance_id]:
                                    coref_dict[tmp_instance_id].update({
                                        "r_id": subelem.attrib["r_id"],
                                        "coref_type": subelem.tag,
                                        "mentions": [mentions[m.attrib["m_id"]] for m in subelem if
                                                     m.tag == "source"]
                                    })
                                else:
                                    coref_dict[tmp_instance_id]["mentions"].extend(
                                        [mentions[m.attrib["m_id"]] for m in subelem if
                                         m.tag == "source"])
                            except KeyError:
                                pass
                            for m in subelem:
                                mentions_map[m.attrib["m_id"]] = True

                newsplease_custom = copy.copy(newsplease_format)

                newsplease_custom["title"] = None   # title
                newsplease_custom["date_publish"] = None

                # if len(text):
                #     text = text if text[-1] != "," else text[:-1] + "."
                newsplease_custom["filename"] = topic_file
                newsplease_custom["text"] = text
                newsplease_custom["source_domain"] = topic_file.split(".")[0]
                newsplease_custom["language"] = result_path[-14:].split("_")[2]
                newsplease_custom["title"] = " ".join(topic_file.split(".")[0].split("_")[1:])
                if newsplease_custom["title"][-1] not in string.punctuation:
                    newsplease_custom["title"] += "."

                doc_files[topic_file.split(".")[0]] = newsplease_custom
                if topic_name not in os.listdir(result_path):
                    os.mkdir(os.path.join(result_path, topic_name))

                with open(os.path.join(result_path, topic_name, newsplease_custom["source_domain"] + ".json"),
                          "w") as file:
                    json.dump(newsplease_custom, file)
                    #LOGGER.info('Saved {topic_name}/{newsplease_custom["source_domain"]}')
            # coref_dics[topic_dir] = coref_dict

            for chain_index, (chain_id, chain_vals) in enumerate(coref_dict.items()):
                if chain_vals.get("mentions") is not None and chain_id != "":
                    for m in chain_vals["mentions"]:

                        sent_id = m["sent_id"]

                        token_numbers = [int(t) for t in m[TOKENS_NUMBER]]
                        mention_id = m["doc_id"] + "_" + str(chain_id) + "_" + str(m["sent_id"]) + "_" + str(
                            m[TOKENS_NUMBER][0])

                        if not any(n[MENTION_ID] == mention_id for n in entity_mentions) and not any(n[MENTION_ID] == mention_id for n in event_mentions):
                            mention = { COREF_CHAIN: annotation_prefix+chain_id,
                                        MENTION_NER: m["mention_ner"],
                                        MENTION_HEAD_POS: m["mention_head_pos"],
                                        MENTION_HEAD_LEMMA: m["mention_head_lemma"],
                                        MENTION_HEAD: m["mention_head"],
                                        MENTION_HEAD_ID: m["mention_head_id"],
                                        DOC_ID_FULL: m["doc_id"],
                                        IS_CONTINIOUS: True if token_numbers == list(
                                            range(token_numbers[0], token_numbers[-1] + 1))
                                        else False,
                                        IS_SINGLETON: len(chain_vals["mentions"]) == 1,
                                        MENTION_ID: mention_id,
                                        MENTION_TYPE: chain_id[:3],
                                        MENTION_FULL_TYPE: meantime_types[chain_id[:3]],
                                        SCORE: -1.0,
                                        SENT_ID: sent_id,
                                        TOKENS_NUMBER: token_numbers,
                                        TOKENS_STR: m["text"],
                                        TOPIC_SUBTOPIC: m[TOPIC_SUBTOPIC],
                                        COREF_TYPE: chain_vals["coref_type"],
                                        DESCRIPTION: chain_vals["descr"]
                                       }
                            if "EVENT" in m["type"]:
                                event_mentions_local.append(mention)
                            else:
                                entity_mentions_local.append(mention)
                            summary_df.loc[len(summary_df)] = {
                                DOC_ID: m["doc_id"],
                                COREF_CHAIN: annotation_prefix + chain_id,
                                DESCRIPTION: chain_vals["descr"],
                                MENTION_TYPE: chain_id[:3],
                                MENTION_FULL_TYPE: m["type"],
                                TOKENS_STR: m["text"]
                            }
                        else:
                            for i in range(len(entity_mentions)):
                                if entity_mentions[i][MENTION_ID] == mention_id:
                                    entity_mentions[i][COREF_CHAIN] = "cross_intra_" + chain_id
                            for i in range(len(event_mentions)):
                                if event_mentions[i][MENTION_ID] == mention_id:
                                    event_mentions[i][COREF_CHAIN] = "cross_intra_" + chain_id

            annot_path = os.path.join(result_path, topic_name, "annotation",
                                      "original")  # ->root/data/MEANTIME-prep/test_parsing/topicName/annotation/original
            if topic_name not in os.listdir(os.path.join(result_path)):
                os.mkdir(os.path.join(result_path, topic_name))

            if "annotation" not in os.listdir(os.path.join(result_path, topic_name)):
                os.mkdir(os.path.join(result_path, topic_name, "annotation"))
                os.mkdir(annot_path)

            with open(os.path.join(annot_path, "entity_mentions_" + topic_name + ".json"), "w") as file:
                json.dump(entity_mentions_local, file)

            with open(os.path.join(annot_path, "event_mentions_" + topic_name + ".json"), "w") as file:
                json.dump(event_mentions_local, file)

            entity_mentions.extend(entity_mentions_local)
            event_mentions.extend(event_mentions_local)

            summary_conversion_df = pd.concat([summary_conversion_df, pd.DataFrame({
                "files": len(topic_files),
                "tokens": len(conll_df),
                "chains": len(coref_dict),
                "event_mentions": len(event_mentions_local),
                "entity_mentions": len(entity_mentions_local),
                "singletons": sum([v["is_singleton"] for v in event_mentions_local]) + sum(
                    [v["is_singleton"] for v in entity_mentions_local])
            }, index=[topic_name])])

    conll_df = conll_df.reset_index(drop=True)

    # create a conll string from the conll_df
    LOGGER.info("Generating conll string...")
    for i, row in tqdm(conll_df.iterrows(), total=conll_df.shape[0]):
        if row[REFERENCE] is None:
            reference_str = "-"
        else:
            reference_str = row[REFERENCE]

        for mention in [m for m in entity_mentions + event_mentions]:
            if mention[TOPIC_SUBTOPIC] == row[TOPIC_SUBTOPIC] and mention[SENT_ID] == row[SENT_ID] and row[
                    TOKEN_ID] in mention[TOKENS_NUMBER]:
                token_numbers = [int(t) for t in mention[TOKENS_NUMBER]]
                chain = mention[COREF_CHAIN]
                # one and only token
                if len(token_numbers) == 1 and token_numbers[0] == row[TOKEN_ID]:
                    reference_str = reference_str + '| (' + str(chain) + ')'
                # one of multiple tokes
                elif len(token_numbers) > 1 and token_numbers[0] == row[TOKEN_ID]:
                    reference_str = reference_str + '| (' + str(chain)
                elif len(token_numbers) > 1 and token_numbers[len(token_numbers) - 1] == row[TOKEN_ID]:
                    reference_str = reference_str + '| ' + str(chain) + ')'

        #if row[DOC_ID] == topic_name:  # do not overwrite conll rows of previous topic iterations
        conll_df.at[i, REFERENCE] = reference_str

    for i, row in conll_df.iterrows():  # remove the leading characters if necessary (left from initialization)
        if row[REFERENCE].startswith("-| "):
            conll_df.at[i, REFERENCE] = row[REFERENCE][3:]

    conll_topic_df = conll_df[conll_df[TOPIC_SUBTOPIC].str.contains(f'{topic_name}/')].drop(columns=[DOC_ID])

    outputdoc_str = ""
    for (topic_local), topic_df in conll_topic_df.groupby(by=[TOPIC_SUBTOPIC]):
        outputdoc_str += f'#begin document ({topic_local}); part 000\n'

        for (sent_id_local), sent_df in topic_df.groupby(by=[SENT_ID], sort=[SENT_ID]):
            np.savetxt(os.path.join(MEANTIME_PARSING_FOLDER, "tmp.txt"), sent_df.values, fmt='%s', delimiter="\t",
                       encoding="utf-8")
            with open(os.path.join(MEANTIME_PARSING_FOLDER, "tmp.txt"), "r", encoding="utf8") as file:
                saved_lines = file.read()
            outputdoc_str += saved_lines + "\n"

        outputdoc_str += "#end document\n"
    final_output_str += outputdoc_str

    # Check if the brackets ( ) are correct
    try:
        brackets_1 = 0
        brackets_2 = 0
        for i, row in conll_df.iterrows():  # only count brackets in reference column (exclude token text)
            brackets_1 += str(row[REFERENCE]).count("(")
            brackets_2 += str(row[REFERENCE]).count(")")
        LOGGER.info("Amount of mentions in this topic: " + str(len(event_mentions_local + entity_mentions_local)))
        LOGGER.info("Total mentions parsed (all topics): " + str(len(event_mentions + entity_mentions)))
        LOGGER.info("brackets '(' , ')' : " + str(brackets_1) + " , " + str(brackets_2))
        assert brackets_1 == brackets_2
    except AssertionError:
        LOGGER.warning(f'Number of opening and closing brackets in conll does not match! topic: {str(topic_name)}')
        conll_df.to_csv(os.path.join(out_path, CONLL_CSV.replace(".csv", "_"+language+".csv")))
        with open(os.path.join(annot_path, f'{topic_name}.conll'), "w", encoding='utf-8') as file:
            file.write(outputdoc_str)
        sys.exit()

    conll_df.to_csv(os.path.join(out_path, CONLL_CSV.replace(".csv", "_"+language+".csv")))
    with open(os.path.join(annot_path, f'{topic_name}.conll'), "w", encoding='utf-8') as file:
        file.write(outputdoc_str)

    LOGGER.info(
        "Mentions that need manual review to define the head and its attributes have been saved to: " +
        MANUAL_REVIEW_FILE.replace(".json", "_"+language+".json - Total: " + str(len(need_manual_review_mention_head))))
    with open(os.path.join(out_path, MANUAL_REVIEW_FILE.replace(".json", "_"+language+".json")), "w", encoding='utf-8') as file:
        json.dump(need_manual_review_mention_head, file)

    with open(os.path.join(out_path, "conll_as_json_" + language + ".json"), "w", encoding='utf-8') as file:
        json.dump(conll_df.to_dict('records'), file)

    with open(os.path.join(out_path, 'meantime_' + language + '.conll'), "w", encoding='utf-8') as file:
        file.write(final_output_str)

    with open(os.path.join(out_path, MENTIONS_ENTITIES_JSON.replace(".json", "_"+language+".json")), "w", encoding='utf-8') as file:
        json.dump(entity_mentions, file)

    with open(os.path.join(out_path, MENTIONS_EVENTS_JSON.replace(".json", "_"+language+".json")), "w", encoding='utf-8') as file:
        json.dump(event_mentions, file)

    summary_df.to_csv(os.path.join(out_path, MENTIONS_ALL_CSV.replace(".csv", "_"+language+".csv")))
    #summary_conversion_df.to_csv(os.path.join(result_path, now.strftime("%Y-%m-%d_%H-%M") + "_" + "dataset_summary.csv"))

    #merge_intra_cross_results(conll_df, entity_mentions, event_mentions, summary_df)

    LOGGER.info(f'Parsing of MEANTIME annotation with language {language} done!')

def merge_intra_cross_results(conll_df, entity_mentions, event_mentions, summary_df):
    #for m in entity_mentions:

    return None

if __name__ == '__main__':

    for i, source_path in enumerate(source_paths):
        #if i < 2:
        #    continue
        LOGGER.info(f"Processing MEANTIME language {source_path[-34:].split('_')[2]}.")
        intra = os.path.join(source_path, 'intra-doc_annotation')
        intra_cross = os.path.join(source_path, 'intra_cross-doc_annotation')
        conv_files([intra_cross, intra], result_paths[i], out_paths[i], source_path[-34:].split("_")[2], nlps[i])
        merge_intra_cross_results()

    # print('Please enter the number of the set, you want to convert:\n'
    #       '   1 MEANTIME intra document annotation\n'
    #       '   2 MEANTIME cross-document annotation\n'
    #       '   3 both')
    #
    #
    # def choose_input():
    #     setnumber = input()
    #     if setnumber == "1":
    #         c_format = "\"MEANTIME intra document annotation\""
    #         print(conv_files(intra))
    #         return c_format
    #     elif setnumber == "2":
    #         c_format = "\"MEANTIME cross-document annotation\""
    #         print(conv_files(intra_cross))
    #         return c_format
    #     elif setnumber == "3":
    #         c_format = "\"MEANTIME intra and intra cross-document annotations\""
    #         print(conv_files(intra))
    #         print(conv_files(intra_cross))
    #         return c_format
    #     else:
    #         print("Please choose one of the 3 numbers!")
    #         return choose_input()


    # co_format = choose_input()

    # print("\nConversion of {0} from xml to newsplease format and to annotations in a json file is "
    #       "done. \n\nFiles are saved to {1}. \nCopy the topics on which you want to execute Newsalyze to "
    #       "{2}.".format(co_format, result_path, DATA_PATH))
