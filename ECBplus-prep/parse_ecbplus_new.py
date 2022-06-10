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
from insert_whitespace import append_text
from nltk import Tree
from tqdm import tqdm
import warnings
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

nlp = spacy.load(
    r'C:\Users\snake\Documents\GitHub\Diverse_CDCR_datasets\venv\Lib\site-packages\en_core_web_sm\en_core_web_sm-3.2.0')

validated_sentences_df = pd.read_csv(os.path.join(ECB_PARSING_FOLDER, ECBPLUS_FOLDER_NAME,
                                                  "ECBplus_coreference_sentences.csv")).set_index(
    ["Topic", "File", "Sentence Number"])

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

    for topic_folder in selected_topics:
        if topic_folder == "__MACOSX":
            continue

        # if a file with confirmed sentences
        if os.path.isfile(os.path.join(source_path, topic_folder)):
            continue

        LOGGER.info(f'Converting topic {topic_folder}')
        diff_folders = {ECB_FILE: [], ECBPLUS_FILE: []}

        # assign the different folders according to the topics in the variable "diff_folders"
        for topic_file in os.listdir(os.path.join(source_path, topic_folder)):
            if ECBPLUS_FILE in topic_file:
                diff_folders[ECBPLUS_FILE].append(topic_file)
            else:
                diff_folders[ECB_FILE].append(topic_file)

        for annot_folders in list(diff_folders.values()):
            t_number = annot_folders[0].split(".")[0].split("_")[0]
            t_name = re.search(r'[a-z]+', annot_folders[0].split(".")[0])[0]
            topic_name = t_number + t_name
            topic_names.append(topic_name)
            coref_dict = {}
            doc_sent_map = {}

            # for every themed-file in "commentated files"
            for topic_file in tqdm(annot_folders):
                doc_name_ecb = topic_file.split(".")[0].split("_")[-1]
                mention_counter_got, mentions_counter_found = 0, 0
                info_t_name = re.search(r'[\d+]+', topic_file.split(".")[0].split("_")[1])[0]
                t_subt = f'{topic_folder}/{topic_name}/{info_t_name}'

                # import the XML-Datei topic_file
                tree = ET.parse(os.path.join(source_path, topic_folder, topic_file))
                root = tree.getroot()

                title, text, url, time, time2, time3 = "", "", "", "", "", ""

                token_dict, mentions, mentions_map = {}, {}, {}

                t_id = -1
                old_sent = -1
                sent_dict = {}

                for elem in root:
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
                                TOPIC_SUBTOPIC: t_subt,
                                DOC_ID: topic_name,
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
                        pass

                    if elem.tag == "Markables":

                        for i, subelem in enumerate(elem):
                            tokens = [token.attrib[T_ID] for token in subelem]
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
                                while True:
                                    if counter > 50:  # an error must have occurred, so break and add to manual review
                                        need_manual_review_mention_head[str(t_subt) + "_" + str(mention_text)] = {
                                            "mention_text": mention_text,
                                            "sentence_str": sentence_str,
                                            "mention_head": str(mention_head),
                                            "mention_tokens_amount": len(tokens)
                                        }
                                        LOGGER.info("Mention with ID " + str(t_subt) + "_" + str(
                                            mention_text) + " needs manual review. Could not determine the mention head automatically.")
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
                                            mention_head_id = token_dict[t][ID]

                                # add to manual review if the resulting token is not inside the mention
                                # (error must have happened)
                                if mention_head_id not in sent_tokens:  # also "if is None"
                                    if str(t_subt) + "_" + str(mention_text) not in need_manual_review_mention_head:
                                        need_manual_review_mention_head[str(t_subt) + "_" + str(mention_text)] = \
                                            {
                                                "mention_text": mention_text,
                                                "sentence_str": sentence_str,
                                                "mention_head": str(mention_head),
                                                "mention_tokens_amount": len(tokens)
                                            }
                                        with open(os.path.join(out_path, MANUAL_REVIEW_FILE), "w",
                                                  encoding='utf-8') as file:
                                            json.dump(need_manual_review_mention_head, file)
                                        LOGGER.info("Mention with ID " + str(t_subt) + "_" + str(
                                            mention_text) + " needs manual review. Could not determine the mention head id automatically.")

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
                                                                  MENTION_NER: mention_ner,
                                                                  MENTION_HEAD_POS: mention_head_pos,
                                                                  MENTION_HEAD_LEMMA: mention_head_lemma,
                                                                  MENTION_HEAD: mention_head_text,
                                                                  MENTION_HEAD_ID: mention_head_id,
                                                                  TOKENS_NUMBER: [int(token_dict[t][ID]) for t in
                                                                                  tokens],
                                                                  TOKENS_TEXT: [token_dict[t][TEXT] for t in tokens],
                                                                  DOC_ID: topic_file.split(".")[0],
                                                                  SENT_ID: sent_id,
                                                                  MENTION_CONTEXT: mention_context_str,
                                                                  TOPIC: t_subt}
                                mentions_counter_found += 1
                            else:
                                try:
                                    # if there are no t_ids (token is empty) and the "instance_id" is not in coref-dict:
                                    if subelem.attrib["instance_id"] not in coref_dict:
                                        # save in coref_dict to the instance_id
                                        coref_dict[subelem.attrib["instance_id"]] = {
                                            DESCRIPTION: subelem.attrib["TAG_DESCRIPTOR"]}
                                    # m_id points to the target
                                except KeyError as e:
                                    pass

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
                            except KeyError:
                                pass

                            for m in subelem:
                                mentions_map[m.attrib[M_ID]] = True

                        for i, (m_id, used) in enumerate(mentions_map.items()):
                            if used:
                                continue

                            m = mentions[m_id]
                            if "Singleton_" + m[MENTION_TYPE][:4] + "_" + str(m_id) + "_" + m[DOC_ID] not in coref_dict:
                                coref_dict["Singleton_" + m[MENTION_TYPE][:4] + "_" + str(m_id) + "_" + m[DOC_ID]] = {
                                    "r_id": str(10000 + i),
                                    COREF_TYPE: "Singleton",
                                    MENTIONS: [m],
                                    DESCRIPTION: ""
                                }
                            else:
                                coref_dict["Singleton_" + m["type"][:4] + "_" + str(m_id) + "_" + m[DOC_ID]].update(
                                    {
                                        "r_id": str(10000 + i),
                                        COREF_TYPE: "Singleton",
                                        MENTIONS: [m],
                                        DESCRIPTION: ""
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
                if topic_name not in os.listdir(result_path):
                    os.mkdir(os.path.join(result_path, topic_name))

                with open(os.path.join(result_path, topic_name, newsplease_custom[SOURCE_DOMAIN] + ".json"),
                          "w") as file:
                    json.dump(newsplease_custom, file)

                # LOGGER.info(f'GOT:   {mention_counter_got}')
                # LOGGER.info(f'FOUND: {mentions_counter_found}\n')

            coref_dics[topic_folder] = coref_dict

            entity_mentions_local = []
            event_mentions_local = []
            mentions_local = []

            for chain_id, chain_vals in coref_dict.items():
                not_unique_heads = []

                if MENTIONS not in chain_vals:
                    continue

                for m in chain_vals[MENTIONS]:

                    if ECBPLUS_FILE.split(".")[0] not in m[DOC_ID]:
                        sent_id = int(m[SENT_ID])
                    else:
                        df = doc_sent_map[m[DOC_ID]]
                        # sent_id = np.sum([df.iloc[:list(df.index).index(int(m[SENT_ID])) + 1][IS_TEXT].values])
                        sent_id = int(m[SENT_ID])
                    # converts TOKENS_NUMBER of m to an array TOKENS_NUMBER with int values

                    # create variable "mention_id" out of doc_id+_+chain_id+_+sent_id+_+first value of TOKENS_NUMBER of "m"
                    token_numbers = [int(t) for t in m[TOKENS_NUMBER]]
                    mention_id = m[DOC_ID] + "_" + str(chain_id) + "_" + str(m[SENT_ID]) + "_" + str(
                        m[TOKENS_NUMBER][0])

                    not_unique_heads.append(m[MENTION_HEAD_LEMMA])

                    # create the dict. "mention" with all corresponding values
                    mention = {COREF_CHAIN: chain_id,
                               MENTION_NER: m[MENTION_NER],
                               MENTION_HEAD_POS: m[MENTION_HEAD_POS],
                               MENTION_HEAD_LEMMA: m[MENTION_HEAD_LEMMA],
                               MENTION_HEAD: m[MENTION_HEAD],
                               MENTION_HEAD_ID: m[MENTION_HEAD_ID],
                               DOC_ID: m[DOC_ID],
                               # DOC_ID_FULL: m[DOC_ID],
                               IS_CONTINIOUS: bool(
                                   token_numbers == list(range(token_numbers[0], token_numbers[-1] + 1))),
                               IS_SINGLETON: bool(len(chain_vals[MENTIONS]) == 1),
                               MENTION_ID: mention_id,
                               MENTION_TYPE: m[MENTION_TYPE][:3],
                               MENTION_FULL_TYPE: m[MENTION_TYPE],
                               SCORE: -1.0,
                               SENT_ID: sent_id,
                               MENTION_CONTEXT: m[MENTION_CONTEXT],
                               # now the token numbers based on spacy tokenization, not ecb+ tokenization   ("token_doc_numbers") -> changed back to original
                               TOKENS_NUMBER: token_numbers,
                               TOKENS_STR: m[TOKENS_STR],
                               TOKENS_TEXT: m[TOKENS_TEXT],
                               TOPIC_ID: int(t_number),
                               TOPIC: topic_name,
                               # COREF_TYPE: chain_vals[COREF_TYPE],
                               COREF_TYPE: STRICT,
                               DESCRIPTION: chain_vals[DESCRIPTION],
                               "t_subt": m[TOPIC],
                               }

                    # if the first two entries of chain_id are "ACT" or "NEG", add the "mention" to the array "event_mentions_local"
                    if chain_id[:3] in ["ACT", "NEG"]:
                        event_mentions_local.append(mention)
                    # else add the "mention" to the array "event_mentions_local" and add the following values to the DF "summary_df"
                    else:
                        entity_mentions_local.append(mention)

                    if not mention[IS_SINGLETON]:
                        mentions_local.append(mention)

            conll_df = conll_df.reset_index(drop=True)

            # create a conll string from the conll_df
            for i, row in conll_df.iterrows():
                reference_str = "-"
                for mention in [m for m in event_mentions_local + entity_mentions_local if
                                m["t_subt"] == row[TOPIC_SUBTOPIC] and m[SENT_ID] == row[SENT_ID] and row[TOKEN_ID] in
                                m[TOKENS_NUMBER]]:

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

                if row[DOC_ID] == topic_name:  # do not overwrite conll rows of previous topic iterations
                    conll_df.at[i, REFERENCE] = reference_str

            for i, row in conll_df.iterrows():  # remove the leading characters if necessary (left from initialization)
                if row[REFERENCE].startswith("-| "):
                    conll_df.at[i, REFERENCE] = row[REFERENCE][3:]

            # create annot_path and file-structure (if not already) for the output of the annotations
            annot_path = os.path.join(result_path, topic_name, "annotation", "original")
            # ->root/data/ECBplus-prep/test_parsing/topicName/annotation/original
            if topic_name not in os.listdir(os.path.join(result_path)):
                os.mkdir(os.path.join(result_path, topic_name))

            if "annotation" not in os.listdir(os.path.join(result_path, topic_name)):
                os.mkdir(os.path.join(result_path, topic_name, "annotation"))
                os.mkdir(annot_path)
            # create the entity-mentions and event-mentions - .json files out of the arrays
            with open(os.path.join(annot_path, f'{ENTITY}_{MENTIONS}_{topic_name}.json'), "w") as file:
                json.dump(entity_mentions_local, file)
            entity_mentions.extend(entity_mentions_local)

            with open(os.path.join(annot_path, f'{EVENT}_{MENTIONS}_{topic_name}.json'), "w") as file:
                json.dump(event_mentions_local, file)
            event_mentions.extend(event_mentions_local)

            with open(os.path.join(annot_path, MANUAL_REVIEW_FILE), "w") as file:
                json.dump(event_mentions_local, file)

            conll_topic_df = conll_df[conll_df[TOPIC_SUBTOPIC].str.contains(f'{topic_name}/')].drop(columns=[DOC_ID])

            outputdoc_str = ""
            for (topic_local), topic_df in conll_topic_df.groupby(by=[TOPIC_SUBTOPIC]):
                outputdoc_str += f'#begin document ({topic_local}); part 000\n'

                for (sent_id_local), sent_df in topic_df.groupby(by=[SENT_ID], sort=[SENT_ID]):
                    np.savetxt(os.path.join(ECB_PARSING_FOLDER, "tmp.txt"), sent_df.values, fmt='%s', delimiter="\t",
                               encoding="utf-8")
                    with open(os.path.join(ECB_PARSING_FOLDER, "tmp.txt"), "r", encoding="utf8") as file:
                        saved_lines = file.read()
                    outputdoc_str += saved_lines + "\n"

                outputdoc_str += "#end document\n"
            final_output_str += outputdoc_str

            # Check if the brackets ( ) are correct
            LOGGER.info("Checking equal brackets in conll for " + str(
                topic_name) + " (if unequal, the result may be incorrect):")
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
                LOGGER.warning(
                    f'Number of opening and closing brackets in conll does not match! topic: ' + str(topic_name))
                conll_df.to_csv(os.path.join(out_path, CONLL_CSV))
                with open(os.path.join(annot_path, f'{topic_name}.conll'), "w", encoding='utf-8') as file:
                    file.write(outputdoc_str)
                sys.exit()

            conll_df.to_csv(os.path.join(out_path, CONLL_CSV))

            with open(os.path.join(annot_path, f'{topic_name}.conll'), "w", encoding='utf-8') as file:
                file.write(outputdoc_str)

            # Average number of unique head lemmas within a cluster
            # (excluding singletons for fair comparison)
            for m in mentions_local:
                head_lemmas = [m[MENTION_HEAD_LEMMA]]
                uniques = 1
                for m2 in mentions_local:
                    if m2[MENTION_HEAD_LEMMA] not in head_lemmas and m2[TOKENS_STR] == m[TOKENS_STR] and m2[TOPIC_ID] == \
                            m[TOPIC_ID]:
                        head_lemmas.append(m[MENTION_HEAD_LEMMA])
                        uniques = uniques + 1
                m["mention_head_lemma_uniques"] = uniques

    conll_df.to_csv(os.path.join(out_path, CONLL_CSV))

    LOGGER.info("Mentions that need manual review to define the head and its attributes have been saved to: " + MANUAL_REVIEW_FILE)

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

    df_all_mentions.to_csv(os.path.join(os.getcwd(), OUTPUT_FOLDER_NAME, MENTIONS_ALL_CSV))
    LOGGER.info(f'Done! \nNumber of unique mentions: {len(df_all_mentions)} '
                f'\nNumber of unique chains: {len(set(df_all_mentions[COREF_CHAIN].values))} ')


# main function for the input which topics of the ecb corpus are to be converted
if __name__ == '__main__':
    topic_num = 45
    convert_files(topic_num)
    LOGGER.info("\nConversion of {0} topics from xml to newsplease format and to annotations in a json file is "
                "done. \n\nFiles are saved to {1}. \n.".format(str(topic_num), result_path))
