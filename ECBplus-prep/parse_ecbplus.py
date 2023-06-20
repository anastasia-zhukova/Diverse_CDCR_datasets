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

with open(os.path.join(ECB_PARSING_FOLDER, "train_dev_test_split.json"), "r") as file:
    train_dev_test_split_dict = json.load(file)

with open(os.path.join(ECB_PARSING_FOLDER, "subtopic_names.json"), "r") as file:
    subtopic_names_dict = json.load(file)

with open(path_sample, "r") as file:
    newsplease_format = json.load(file)


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


def convert_files(topic_number_to_convert=3, check_with_list=True):
    doc_files = {}
    coref_dics = {}

    selected_topics = os.listdir(source_path)[:topic_number_to_convert]

    conll_df = pd.DataFrame()
    final_output_str = ""
    entity_mentions = []
    event_mentions = []
    topic_names = []
    need_manual_review_mention_head = {}

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
            doc_sent_map = {}

            # for every themed-file in "commentated files"
            for topic_file in tqdm(annot_folders):
                doc_name_ecb = topic_file.split(".")[0].split("_")[-1]
                mention_counter_got, mentions_counter_found = 0, 0
                doc_id = re.search(r'[\d+]+', topic_file.split(".")[0].split("_")[1])[0]
                topic_subtopic_doc = f'{topic_id}/{subtopic_id}/{doc_id}'

                # import the XML-Datei topic_file
                tree = ET.parse(os.path.join(source_path, topic_id, topic_file))
                root = tree.getroot()

                title, text, url, time, time2, time3 = "", "", "", "", "", ""

                token_dict, mentions, mentions_map = {}, {}, {}

                t_id = -1
                old_sent = -1
                sent_dict = {}

                for elem in root:
                    if elem.tag == "token":
                        try:
                            # increase t_id value by 1 if the sentence value in the xml element ''equals the value of old_sent
                            if old_sent == int(elem.attrib[SENTENCE]):
                                t_id += 1
                                # else set old_sent to the value of sentence and t_id to 0
                            else:
                                old_sent = int(elem.attrib[SENTENCE])
                                t_id = 0

                            # fill the token-dictionary with fitting attributes
                            token_dict[elem.attrib[T_ID]] = {TEXT: elem.text, SENT: elem.attrib[SENTENCE], ID: t_id,
                                                             NUM: elem.attrib[NUM]}

                            prev_word = ""
                            if t_id >= 0:
                                prev_word = root[t_id - 1].text

                            if elem.tag == TOKEN:
                                _, word, space = append_text(prev_word, elem.text)
                                conll_df = conll_df.append(pd.DataFrame({
                                    TOPIC_SUBTOPIC_DOC: topic_subtopic_doc,
                                    DOC_ID: subtopic_id,
                                    SENT_ID: int(token_dict[elem.attrib[T_ID]][SENT]),
                                    TOKEN_ID: int(t_id),
                                    TOKEN: word,
                                    REFERENCE: "-"
                                }, index=[elem.attrib[T_ID]]))

                            if ECB_FILE in topic_file:
                                # set the title-attribute of all words of the first sentence together und the text-attribute of the rest
                                if int(elem.attrib[SENTENCE]) == 0:
                                    title, _, _ = append_text(title, elem.text)
                                else:
                                    text, _, _ = append_text(text, elem.text)

                            if ECBPLUS_FILE in topic_file:
                                # add a string with the complete sentence to the sentence-dictionary for every different sentence
                                new_text, _, _ = append_text(sent_dict.get(int(elem.attrib[SENTENCE]), ""), elem.text)
                                sent_dict[int(elem.attrib[SENTENCE])] = new_text

                        except KeyError as e:
                            LOGGER.warning(f'Value with key {e} not found and will be skipped from parsing.')

                    if elem.tag == "Markables":
                        for i, subelem in enumerate(elem):
                            tokens = [token.attrib[T_ID] for token in subelem]
                            tokens.sort(key=int)  # sort tokens by their id
                            sent_tokens = [int(token_dict[t][NUM]) for t in tokens]

                            # skip if the token is contained more than once within the same mention
                            # (i.e. ignore entries with error in ecb+ tokenization)
                            if len(tokens) != len(list(set(tokens))):
                                continue

                            mention_text = ""
                            for t in tokens:
                                mention_text, _, _ = append_text(mention_text, token_dict[t][TEXT])

                            # if "tokens" has values -> fill the "mention" dict with the value of the corresponding m_id
                            if len(tokens):
                                mention_counter_got += 1
                                sent_id = int(token_dict[tokens[0]][SENT])

                                # take only validated sentences
                                if (int(t_number), doc_name_ecb, sent_id) not in validated_sentences_df.index:
                                    continue

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
                                # first_char_of_mention = sentence_str.find(split_mention_text[0])
                                if len(split_mention_text) > 1:
                                    first_char_of_mention = sentence_str.find(
                                        split_mention_text[0] + " " + split_mention_text[
                                            1])  # more accurate finding (reduce error if first word is occurring multiple times (i.e. "the")
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

                                counter = 0
                                mention_doc_ids = []
                                tolerance = 0
                                mention_id = str(topic_subtopic_doc) + "_" + str(mention_text)

                                while True:
                                    if counter > 50:  # an error must have occurred, so break and add to manual review

                                        need_manual_review_mention_head[mention_id] = {
                                            "mention_text": mention_text,
                                            "sentence_str": sentence_str,
                                            "mention_head": str(mention_head),
                                            "mention_tokens_amount": len(tokens)
                                        }
                                        LOGGER.info(f"Mention with ID {mention_id} needs manual review. "
                                                    f"Could not determine the mention head automatically.")
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
                                                    + int(subelem[-1].attrib[T_ID]) \
                                                    - int(subelem[0].attrib[T_ID]) \
                                                    - len(subelem) \
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
                                            break
                                        else:
                                            counter = counter + 1
                                            # The next char is not a punctuation, so it therefore it is just a part of a bigger word
                                            first_char_of_mention = sentence_str.find(
                                                re.split(" ", mention_text)[0],
                                                first_char_of_mention + 2)
                                            last_char_of_mention = sentence_str.find(
                                                re.split(" ", mention_text)[-1],
                                                first_char_of_mention + len(
                                                    re.split(" ", mention_text)[0])) + len(
                                                re.split(" ", mention_text)[-1])

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

                                # whole mention string processed, look for the head
                                # mention_head = None
                                if mention_id not in need_manual_review_mention_head:
                                    mention_head = doc[0]
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
                                    if mention_id not in need_manual_review_mention_head:
                                        need_manual_review_mention_head[mention_id] = \
                                            {
                                                "mention_text": mention_text,
                                                "sentence_str": sentence_str,
                                                "mention_head": str(mention_head),
                                                "mention_tokens_amount": len(tokens),
                                                "tolerance": tolerance
                                            }
                                        with open(os.path.join(TMP_PATH, MANUAL_REVIEW_FILE), "w",
                                                  encoding='utf-8') as file:
                                            json.dump(need_manual_review_mention_head, file)
                                        LOGGER.warning(
                                            f"Document {doc_id}: Mention with ID {mention_id} needs manual review. "
                                            f"Could not determine the mention head automatically. {str(tolerance)}")

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
                                    if t.tag == TOKEN and int(t.attrib[T_ID]) >= context_min_id and int(
                                            t.attrib[T_ID]) <= context_max_id:
                                        mention_context_str.append(t.text)
                                mentions[subelem.attrib[M_ID]] = {MENTION_TYPE: subelem.tag,
                                                                  MENTION_FULL_TYPE: subelem.tag,
                                                                  TOKENS_STR: mention_text.strip(),
                                                                  MENTION_ID: mention_id,
                                                                  MENTION_NER: mention_ner,
                                                                  MENTION_HEAD_POS: mention_head_pos,
                                                                  MENTION_HEAD_LEMMA: mention_head_lemma,
                                                                  MENTION_HEAD: mention_head_text,
                                                                  MENTION_HEAD_ID: mention_head_id,
                                                                  TOKENS_NUMBER: [int(token_dict[t][ID]) for t in
                                                                                  tokens],
                                                                  TOKENS_TEXT: [token_dict[t][TEXT] for t in tokens],
                                                                  DOC: topic_file.split(".")[0],
                                                                  SENT_ID: sent_id,
                                                                  MENTION_CONTEXT: mention_context_str,
                                                                  TOPIC_SUBTOPIC_DOC: topic_subtopic_doc,
                                                                  TOPIC: topic_subtopic_doc}
                                mentions_counter_found += 1
                            else:
                                # form coreference chain
                                # m_id points to the target
                                if "ent_type" in subelem.attrib:
                                    # mention_type_annot = meantime_types.get(subelem.attrib["ent_type"], "")
                                    mention_type_annot = subelem.attrib["ent_type"]
                                elif "class" in subelem.attrib:
                                    mention_type_annot = subelem.attrib["class"]
                                elif "type" in subelem.attrib:
                                    mention_type_annot = subelem.attrib["type"]
                                else:
                                    mention_type_annot = ""

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
                                            f"Document {doc_id}: {subelem.attrib} doesn\'t have attribute instance_id. It will be created")
                                        if "ent_type" in subelem.attrib:
                                            id_ = subelem.attrib["ent_type"] + shortuuid.uuid()[:17]
                                        elif "class" in subelem.attrib:
                                            id_ = subelem.attrib["class"][:3] + shortuuid.uuid()[:17]
                                        elif "type" in subelem.attrib:
                                            id_ = subelem.attrib["type"][:3] + shortuuid.uuid()[:17]
                                        else:
                                            id_ = ""

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
                            try:
                                if "r_id" not in coref_dict[subelem.attrib["note"]]:
                                    # update the coref_dict-element and add "r_id", "coref_type" and MENTIONS
                                    # just elements of the subelements with tag "sourse" are added to mentions
                                    coref_dict[subelem.attrib["note"]].update({
                                        "r_id": subelem.attrib["r_id"],
                                        COREF_TYPE: subelem.tag,
                                        MENTIONS: [mentions[m.attrib[M_ID]] for m in subelem if m.tag == "source"]
                                    })
                                else:
                                    coref_dict[subelem.attrib["note"]][MENTIONS].extend(
                                        [mentions[m.attrib[M_ID]] for m in subelem if m.tag == "source"])
                            except KeyError as e:
                                pass
                                # LOGGER.warning(
                                # f'Document {doc_id}: Mention with ID {str(e)} is not amoung the Markables and will be skipped.')

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
                                    MENTIONS: [m],
                                    DESCRIPTION: m[TOKENS_STR],
                                    "subtopic": subtopic_id
                                }
                            else:
                                coref_dict[chain_id_created].update(
                                    {
                                        "r_id": str(10000 + i),
                                        COREF_TYPE: m[MENTION_TYPE],
                                        MENTIONS: [m],
                                        "subtopic": subtopic_id,
                                        DESCRIPTION:  m[TOKENS_STR],
                                    })
                    a = 1

                newsplease_custom = copy.copy(newsplease_format)

                if ECBPLUS_FILE in topic_file:
                    sent_df = pd.DataFrame(columns=[IS_TEXT, TEXT])
                    for sent_key, text in sent_dict.items():
                        # expand the dataframe
                        # IS_TEXT is 0 (if the (number of number-signs in Text)/(Textlength) >= 0.1 or Counter(sent_key) = 0. else 1
                        # TEXT to text
                        # index to sent_key
                        sent_df = sent_df.append(pd.DataFrame({
                            IS_TEXT: 0 if len(re.sub(r'[\D]+', "", text)) / len(text) >= 0.1 or sent_key == 0 else 1,
                            TEXT: text
                        }, index=[sent_key]))
                    doc_sent_map[topic_file.split(".")[0]] = sent_df

                    small_df = sent_df[sent_df[IS_TEXT] == 0]
                    # create/ add values to DataFrame news_please_custom :
                    # "date_published" as " " + all other TEXT- values of the Dataframe w.o. the first. "" if not there
                    # "title" with first value of TEXT, of wich IS_TEXT = 1 (<= 10% Zahlen)
                    # set Variable TEXT to " "+ rest of values (w.o. 1st one) that consist <= 10% of the numbers

                    # URL
                    url_space_text = ""
                    if len(small_df) > 0:
                        url_space_list = str(list(small_df[TEXT].values)[0]).split(" ")
                        if len(url_space_list) > 1:
                            for c in url_space_list:
                                url_space_text, _, space = append_text(url_space_text, c)

                    newsplease_custom["url"] = url_space_text
                    newsplease_custom["date_publish"] = " ".join(list(small_df[TEXT].values)[1:]) if len(
                        small_df) > 1 else ""
                    newsplease_custom["title"] = list(sent_df[sent_df[IS_TEXT] == 1][TEXT].values)[0]
                    text = " ".join(list(sent_df[sent_df[IS_TEXT] == 1][TEXT].values)[1:])
                else:
                    newsplease_custom["title"] = title
                    newsplease_custom["date_publish"] = None

                if len(text):
                    text = text if text[-1] != "," else text[:-1] + "."
                # create/ add values toDataFrame news_please_custom:
                # TEXT = text
                # "source_domain" = name of file
                newsplease_custom[TEXT] = text
                newsplease_custom[SOURCE_DOMAIN] = topic_file.split(".")[0]
                if newsplease_custom[TITLE][-1] not in string.punctuation:
                    newsplease_custom["title"] += "."

                doc_files[topic_file.split(".")[0]] = newsplease_custom
                if subtopic_id not in os.listdir(result_path):
                    os.mkdir(os.path.join(result_path, subtopic_id))

                with open(os.path.join(result_path, subtopic_id, newsplease_custom[SOURCE_DOMAIN] + ".json"),
                          "w") as file:
                    json.dump(newsplease_custom, file)

                # LOGGER.info(f'GOT:   {mention_counter_got}')
                # LOGGER.info(f'FOUND: {mentions_counter_found}\n')

            coref_dics[topic_id] = coref_dict

            entity_mentions_local = []
            event_mentions_local = []
            mentions_local = []

            for chain_id, chain_vals in coref_dict.items():
                not_unique_heads = []

                if MENTIONS not in chain_vals:
                    continue

                for m in chain_vals[MENTIONS]:

                    if ECBPLUS_FILE.split(".")[0] not in m[DOC]:
                        sent_id = int(m[SENT_ID])
                    else:
                        df = doc_sent_map[m[DOC]]
                        # sent_id = np.sum([df.iloc[:list(df.index).index(int(m[SENT_ID])) + 1][IS_TEXT].values])
                        sent_id = int(m[SENT_ID])
                    # converts TOKENS_NUMBER of m to an array TOKENS_NUMBER with int values

                    # create variable "mention_id" out of doc_id+_+chain_id+_+sent_id+_+first value of TOKENS_NUMBER of "m"
                    token_numbers = [int(t) for t in m[TOKENS_NUMBER]]
                    mention_id = m[DOC] + "_" + str(chain_id) + "_" + str(m[SENT_ID]) + "_" + str(
                        m[TOKENS_NUMBER][0]) + "_" + shortuuid.uuid()[:4]

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

            # create annot_path and file-structure (if not already) for the output of the annotations
            annot_path = os.path.join(result_path, subtopic_id, "annotation", "original")
            if subtopic_id not in os.listdir(os.path.join(result_path)):
                os.mkdir(os.path.join(result_path, subtopic_id))

            if "annotation" not in os.listdir(os.path.join(result_path, subtopic_id)):
                os.mkdir(os.path.join(result_path, subtopic_id, "annotation"))
                os.mkdir(annot_path)

            # create the entity-mentions and event-mentions - .json files out of the arrays
            with open(os.path.join(annot_path, f'{ENTITY}_{MENTIONS}_{subtopic_id}.json'), "w") as file:
                json.dump(entity_mentions_local, file)

            entity_mentions.extend(entity_mentions_local)

            with open(os.path.join(annot_path, f'{EVENT}_{MENTIONS}_{subtopic_id}.json'), "w") as file:
                json.dump(event_mentions_local, file)
            event_mentions.extend(event_mentions_local)

            conll_topic_df = conll_df[conll_df[TOPIC_SUBTOPIC_DOC].str.contains(f'{subtopic_id}/')].drop(columns=[DOC_ID])
            make_save_conll(conll_topic_df, event_mentions_local+entity_mentions_local, annot_path)

    if len(need_manual_review_mention_head):
        LOGGER.warning(f'Mentions ignored: {len(need_manual_review_mention_head)}. The ignored mentions are available here for a manual review: '
                    f'{os.path.join(out_path,MANUAL_REVIEW_FILE)}')
        with open(os.path.join(out_path, MANUAL_REVIEW_FILE), "w", encoding='utf-8') as file:
            json.dump(need_manual_review_mention_head, file)

    with open(os.path.join(out_path, ECB_PLUS.split("-")[0] + '.conll'), "w", encoding='utf-8') as file:
        file.write(final_output_str)

    with open(os.path.join(out_path, MENTIONS_ENTITIES_JSON), "w", encoding='utf-8') as file:
        json.dump(entity_mentions, file)

    with open(os.path.join(out_path, MENTIONS_EVENTS_JSON), "w", encoding='utf-8') as file:
        json.dump(event_mentions, file)

    # create a csv. file out of the mentions summary_df
    df_all_mentions = pd.DataFrame()
    for mention in entity_mentions + event_mentions:
        df_all_mentions = pd.concat([df_all_mentions, pd.DataFrame({
            attr: str(value) if type(value) == list else value for attr, value in mention.items()
        }, index=[mention[MENTION_ID]])], axis=0)

    df_all_mentions.to_csv(os.path.join(out_path, MENTIONS_ALL_CSV))

    make_save_conll(conll_df, df_all_mentions, out_path)

    LOGGER.info(f'Done! \nNumber of unique mentions: {len(df_all_mentions)} '
                f'\nNumber of unique chains: {len(set(df_all_mentions[COREF_CHAIN].values))} ')

    LOGGER.info(f'Splitting ECB+ into train/dev/test subsets...')
    for subset, topic_ids in train_dev_test_split_dict.items():
        LOGGER.info(f'Creating data for {subset} subset...')
        split_folder = os.path.join(out_path, subset)
        if subset not in os.listdir(out_path):
            os.mkdir(split_folder)

        selected_entity_mentions = []
        for mention in entity_mentions:
            if int(mention[TOPIC_ID]) in topic_ids:
                selected_entity_mentions.append(mention)

        selected_event_mentions = []
        for mention in event_mentions:
            if int(mention[TOPIC_ID]) in topic_ids:
                selected_event_mentions.append(mention)

        with open(os.path.join(split_folder, MENTIONS_ENTITIES_JSON), "w", encoding='utf-8') as file:
            json.dump(selected_entity_mentions, file)

        with open(os.path.join(split_folder, MENTIONS_EVENTS_JSON), "w", encoding='utf-8') as file:
            json.dump(selected_event_mentions, file)

        conll_df_split = pd.DataFrame()
        for t_id in topic_ids:
            conll_df_split = pd.concat([conll_df_split,
                                        conll_df[conll_df[TOPIC_SUBTOPIC_DOC].str.contains(f'{t_id}/')]], axis=0)
        make_save_conll(conll_df_split, selected_event_mentions+selected_entity_mentions, split_folder, False)
    LOGGER.info(f'Parsing ECB+ is done!')


# main function for the input which topics of the ecb corpus are to be converted
if __name__ == '__main__':
    topic_num = 45
    convert_files(topic_num)
    LOGGER.info("\nConversion of {0} topics from xml to newsplease format and to annotations in a json file is "
                "done. \n\nFiles are saved to {1}. \n.".format(str(topic_num), result_path))
