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
from tqdm import tqdm
from setup import *
from insert_whitespace import append_text
from config import DATA_PATH, TMP_PATH
from logger import LOGGER
from pathlib import Path

path_sample = os.path.join(DATA_PATH, "_sample_doc.json")  # ->root/data/original/_sample_doc.json
NIDENT_PARSING_FOLDER = os.path.join(os.getcwd())
OUT_PATH = os.path.join(NIDENT_PARSING_FOLDER, OUTPUT_FOLDER_NAME)
CONTEXT_RANGE = 250

nlp = spacy.load('en_core_web_sm')

# opens and loads the newsplease-format out of the json file: _sample_doc.json
with open(path_sample, "r") as file:
    newsplease_format = json.load(file)


source_path = os.path.join(NIDENT_PARSING_FOLDER, NIDENT_FOLDER_NAME)
result_path = os.path.join(OUT_PATH, 'test_parsing')
out_path = os.path.join(OUT_PATH)

count_punct = lambda l: sum([1 for x in l if x in string.punctuation])

def to_nltk_tree(node):
    """
        Converts a sentence to a visually helpful tree-structure output.
        Can be used to double-check if a determined head is correct.
    """
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


def conv_files(path):
    """
        Converts the given dataset for a specified language into the desired format.
        :param paths: The paths desired to process (intra- & cross_intra annotations)
        :param result_path: the test_parsing folder for that language
        :param out_path: the output folder for that language
        :param language: the language to process
        :param nlp: the spacy model that fits the desired language
    """
    doc_files = {}
    coref_dics = {}
    relation = {}

    entity_mentions = []
    event_mentions = []
    ee_mentions = []

    relations = []
    recorded_ent = []

    dirs = os.listdir(path)
    cnt = 0
    mark_counter = 0

    doc_files = {}
    entity_mentions = []
    event_mentions = []
    summary_df = pd.DataFrame(
        columns=[DOC_ID, COREF_CHAIN, DESCRIPTION, MENTION_TYPE, MENTION_FULL_TYPE, MENTION_ID, TOKENS_STR])
    conll_df = pd.DataFrame(columns=[TOPIC_SUBTOPIC, DOC_ID, SENT_ID, TOKEN_ID, TOKEN, REFERENCE])
    final_output_str = ""
    need_manual_review_mention_head = {}

    # map xml files to their topics (np4e folder structure)
    topic_dict = {}
    topic_ids = { "bukavu": 1, "china": 2, "israel": 3, "peru": 4, "tajikistan": 5 }
    for topic_dir_np4e in os.listdir(os.path.join(Path(os.getcwd()).parent.absolute(), NP4E, NP4E_FOLDER_NAME, "mmax2")):
        cnt = cnt + 1
        event_mentions_local = []
        entity_mentions_local = []

        doc_type = "nident"
        coref_dict = {}
        LOGGER.info(f"Parsing of NiDENT topic {topic_dir_np4e}...")
        topic_files = os.listdir(os.path.join(path, "english-corpus"))
        topic_name = str(topic_dir_np4e)
        topic_id = topic_ids[topic_name]

        coref_dict = {}

        for file in os.listdir(os.path.join(Path(os.getcwd()).parent.absolute(), NP4E, NP4E_FOLDER_NAME, "mmax2", topic_dir_np4e)):
            file_spec = str(file).split(".")[0]
            topic_file = "20111107.near2_"+file_spec+".xml"
            # topic_dict[file_spec] = { "topic": str(topic_dir), "topic_id": topic_ids[str(topic_dir)] }
            if "Basedata" in topic_file or "common" in topic_file or "markables" in topic_file or not os.path.isfile(os.path.join(path,"english-corpus", topic_file)):
                continue    # skip directories

            tree = ET.parse(os.path.join(path,"english-corpus", topic_file))
            root = tree.getroot()
            title, text, url, time, time2, time3 = "", "", "", "", "", ""

            token_dict, mentions, mentions_map = {}, {}, {}

            t_id = -1
            sent_id = -1

            old_sent = 0
            word_count = 0
            sent_dict = {}
            sentences = tree.findall('.//sentence')
            for sentence in sentences:
                sent_word_ar = []
                token_id = -1
                sent_id = int(sentence.get("id")[1:])

                for s_elem in sentence.iter():
                    if s_elem.tag == 'word':
                        sent_word_ar.append(s_elem)

                for word in sent_word_ar:
                    prev_word = ""
                    if token_id >= 0:
                        prev_word = sent_word_ar[token_id]
                    token_id += 1
                    word_count += 1

                    info_t_name = str(topic_file.split("_")[1]).split(".")[0]
                    t_subt = topic_name + "/" + info_t_name

                    # information.txt-dataframe construction
                    conll_df = pd.concat([conll_df,pd.DataFrame({
                        TOPIC_SUBTOPIC: t_subt,
                        SENT_ID: sent_id,
                        TOKEN_ID: token_id,
                        TOKEN: word.get("wd"),
                        REFERENCE: "-"
                    }, index=[0])])

                    if 'S' + str(old_sent) == str(sentence.attrib["id"]):
                        t_id += 1
                    else:
                        old_sent = str(sentence.attrib["id"]).split("S")[1]
                        t_id = 0
                    token_dict[str(word_count)] = {"text": word.attrib["wd"],
                                                   "sent": str(sentence.attrib["id"]).split("S")[1], "id": t_id}
                    if str(sentence.attrib["id"]) == "S1":
                        title, word_fixed, no_whitespace = append_text(title, word.attrib["wd"])
                    else:
                        text, word_fixed, no_whitespace = append_text(text, word.attrib["wd"])

                # Markables
                markables = sentence.findall('.//sn')

                for markable in markables:

                    mark_counter += 1
                    marker_id = markable.get("markerid")
                    entity_id = markable.get("entity")
                    coref_id = markable.get("corefid")
                    if not coref_id:
                        coref_id = entity_id
                    else:
                        coref_id = coref_id+"_"+entity_id

                    marker_comment = markable.get("markercomment")  # most of the times None

                    #for t in markable:
                    #    print(ET.tostring(t))
                    #print("----")

                    token_numbers = []
                    token_sent_numbers = []
                    token_str = ""
                    token_text = []
                    markable_words = markable.findall('.//word')
                    for word in markable_words:
                        if word.get("wdid") is not None:
                            token_sent_numbers.append(int(word.get("wdid")[1:]))

                        m_word_cnt = 0
                        # get counted word-id(first called "number") in sentence for every word in the markable
                        # if punctuation (wich does not have a wdid) then count last word +1 or if first set number = 0
                        for m_word in sent_word_ar:
                            if m_word.get("wdid") == word.get("wdid"):
                                number = m_word_cnt
                                if word.get("wdid") is None and token_numbers:
                                    number = token_numbers[len(token_numbers) - 1] + 1
                                elif word.get("wdid") is None and not token_numbers:
                                    number = 0
                            m_word_cnt += 1

                        # using the "wdid"- attribute from xml for token_ids does not provide an ascending sequence of numbers
                        # number_ar = re.findall(r'\d+', str(word.get('wdid')))
                        # if number_ar is None or not number_ar:
                        #     number = 0
                        # else:
                        #     number = number_ar[0]

                        token_numbers.append(number)
                        token_str, word_fixed, no_whitespace = append_text(token_str, str(word.attrib['wd']))
                        token_text.append(str(word.attrib['wd']))
                    #print(token_str)
                    #print(token_numbers)
                    #print(token_sent_numbers)

                    doc_id = str(topic_file.split(".xml")[0])
                    entity = str(markable.attrib["entity"])
                    if markable.get("identdegree") == "1":
                        mention_full_type = "weak near-identity"
                    elif markable.get("identdegree") == "2":
                        mention_full_type = "strong near-identity"
                    elif markable.get("identdegree") == "3":
                        mention_full_type = "total identity"
                    else:
                        mention_full_type = "-"

                    # determine the sentences as a string
                    tokens = sentence.findall('.//word')

                    sentence_str = ""
                    for t in tokens:
                        sentence_str, _, _ = append_text(sentence_str, t.get("wd"))

                    sent_tokens = [int(t.get("wdid")[1:]) for t in tokens if t.get("wdid") is not None]

                    # pass the string into spacy
                    #print(sentence_str)
                    #print(sent_tokens)
                    doc = nlp(sentence_str)

                    # skip if the token is contained more than once within the same mention
                    # (i.e. ignore entries with error in meantime tokenization)
                    if len(tokens) != len(list(set(tokens))):
                        continue

                    mention_text = token_str
                    #print("mention_text: " + mention_text)
                    # if "tokens" has values -> fill the "mention" dict with the value of the corresponding m_id
                    if len(tokens):

                        # tokenize the mention text
                        mention_tokenized = []
                        for t in tokens:
                            if t.get("wdid") is not None:
                                mention_tokenized.append(int(t.get("wdid")[1:]))
                            else:
                                mention_tokenized.append(None)

                        split_mention_text = re.split(" ", mention_text)

                        # counting character up to the first character of the mention within the sentence
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
                            # handle special case if the last punctuation is part of mention
                            last_char_of_mention = len(sentence_str)

                        #print("first: " + str(first_char_of_mention) )
                        #print("last: " + str(last_char_of_mention) )

                        counter = 0
                        while True:
                            if counter > 50:  # an error must have occurred, so break and add to manual review
                                need_manual_review_mention_head[str(t_subt) + "_" + str(mention_text)[:10]] = {
                                    "mention_text": mention_text,
                                    "sentence_str": sentence_str,
                                    "mention_head": "unknown",
                                    "mention_tokens_amount": len(tokens),
                                    "tolerance": tolerance
                                }
                                LOGGER.info(
                                    f"Mention with ID {str(t_subt)}_{str(mention_text)} needs manual review. Could not determine the mention head automatically \n(Exceeded max iterations). {str(tolerance)}")
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
                                tolerance = 0#len(tokens) / 2
                                if tolerance > 2:
                                    tolerance = 2
                                # tolerance for website mentions
                                if ".com" in mention_text or ".org" in mention_text:
                                    tolerance = tolerance + 2
                                # tolerance when the mention has external tokens inbetween mention tokens
                                tolerance = tolerance \
                                            + int(count_punct(token_str)) \
                                            + 1
                                           # + int(tokens[-1]) \
                                            #- int(tokens[0]) \
                                            #- len(tokens) \

                                # increase tolerance for every punctuation included in mention text
                                #tolerance = tolerance + sum(
                                #    [1 for c in mention_text if c in string.punctuation])

                                #print("first: " + str(first_char_of_mention) + " " + sentence_str[first_char_of_mention])
                                #print("last: " + str(last_char_of_mention) + " " + sentence_str[last_char_of_mention])


                                if abs(len(re.split(" ", sentence_str[
                                                         first_char_of_mention:last_char_of_mention])) - len(
                                    markable_words)) <= tolerance and sentence_str[
                                    first_char_of_mention - 1] in string.punctuation + " " and sentence_str[
                                    last_char_of_mention] in string.punctuation + " ":
                                    # Whole mention found in sentence (and tolerance is OK)
                                    #print("tolerance OK")
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
                                #print("first: " + str(first_char_of_mention) + " " + sentence_str[first_char_of_mention])
                                #print("last: " + str(last_char_of_mention) + " " + sentence_str[last_char_of_mention])
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
                                if ancestors_in_mention == 0 and doc[i].text not in string.punctuation:     # puncts should not be heads
                                    # head within the mention
                                    mention_head = doc[i]
                        else:
                            mention_head = doc[0]  # as placeholder for manual checking

                        mention_head_lemma = mention_head.lemma_
                        mention_head_pos = mention_head.pos_

                        mention_ner = mention_head.ent_type_
                        if mention_ner == "":
                            mention_ner = "O"

                        # remap the mention head back to the meantime original tokenization to get the ID for the output
                        mention_head_id = None
                        mention_head_text = mention_head.text
                        #print("head text: " + mention_head_text)
                        for t in tokens:
                            if str(t.get("wd")).startswith(mention_head_text) and t.get("wdid") is not None:
                                mention_head_id = int(t.get("wdid")[1:])

                        if not mention_head_id:
                            for t in tokens:
                                if mention_head_text.startswith(str(t.get("wd"))) and t.get("wdid") is not None:
                                    mention_head_id = int(t.get("wdid")[1:])
                        if not mention_head_id:
                            for t in tokens:
                                if str(t.get("wd")).endswith(mention_head_text) and t.get("wdid") is not None:
                                    mention_head_id = int(t.get("wdid")[1:])

                        #print("head id: " + str(mention_head_id))

                        # add to manual review if the resulting token is not inside the mention
                        # (error must have happened)
                        if mention_head_id not in sent_tokens:  # also "if is None"
                            if str(t_subt) + "_" + str(mention_text) not in need_manual_review_mention_head:
                                need_manual_review_mention_head[str(t_subt) + "_" + str(mention_text)[:10]] = \
                                    {
                                        "mention_text": mention_text,
                                        "sentence_str": sentence_str,
                                        "mention_head": str(mention_head),
                                        "mention_tokens_amount": len(tokens),
                                        "tolerance": tolerance
                                    }
                                with open(os.path.join(out_path, MANUAL_REVIEW_FILE),
                                          "w",
                                          encoding='utf-8') as file:
                                    json.dump(need_manual_review_mention_head, file)
                                [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]
                                LOGGER.info(
                                    f"Mention with ID {str(t_subt)}_{str(mention_text)} needs manual review. Could not determine the mention head automatically. {str(tolerance)}")

                        # get the context
                        context_min_id, context_max_id = [0 if int(min(token_sent_numbers)) - CONTEXT_RANGE < 0 else
                                                          int(min(token_sent_numbers)) - CONTEXT_RANGE,
                                                          len(token_dict) - 1
                                                          if int(max(token_sent_numbers)) + CONTEXT_RANGE > len(
                                                              token_dict)
                                                          else int(max(token_sent_numbers)) + CONTEXT_RANGE]

                        mention_context_str = []
                        break_indicator = False
                        # append to the mention context string list
                        for sent in sentences:
                            sent_words = []
                            for s in sent.iter():
                                if s.tag == 'word':
                                    sent_words.append(s)
                            for word in sent_words:
                                if word.get("wdid") is None:
                                    if len(mention_context_str) > 0:
                                        mention_context_str.append(word.get("wd"))
                                elif int(word.get("wdid")[1:]) > context_max_id:    # break when all needed words processed
                                    break_indicator = True
                                    break
                                elif int(word.get("wdid")[1:]) >= context_min_id and int(word.get("wdid")[1:]) <= context_max_id:
                                    mention_context_str.append(word.get("wd"))
                            if break_indicator is True:
                                break

                        # add to mentions if the variables are correct ( do not add for manual review needed )
                        if str(t_subt) + "_" + str(mention_text) not in need_manual_review_mention_head:
                            mention = {COREF_CHAIN: coref_id,
                                       MENTION_NER: mention_ner,
                                       MENTION_HEAD_POS: mention_head_pos,
                                       MENTION_HEAD_LEMMA: mention_head_lemma,
                                       MENTION_HEAD: mention_head_text,
                                       MENTION_HEAD_ID: mention_head_id,
                                       DOC_ID: doc_id,
                                       DOC_ID_FULL: doc_id,
                                       IS_CONTINIOUS: token_numbers == list(range(token_numbers[0], token_numbers[-1] + 1)),
                                       IS_SINGLETON: len(tokens) == 1,
                                       MENTION_ID: marker_id,
                                       MENTION_TYPE: None,
                                       MENTION_FULL_TYPE: None,
                                       SCORE: -1.0,
                                       SENT_ID: sent_id,
                                       MENTION_CONTEXT: mention_context_str,
                                       TOKENS_NUMBER: token_numbers,
                                       TOKENS_STR: token_str,
                                       TOKENS_TEXT: token_text,
                                       TOPIC_ID: cnt,
                                       TOPIC: t_subt.split("/")[0],
                                       SUBTOPIC: t_subt.split("/")[1],
                                       TOPIC_SUBTOPIC: t_subt,
                                       COREF_TYPE: None,
                                       DESCRIPTION: marker_comment,
                                       CONLL_DOC_KEY: t_subt
                                       }
                            #if "EVENT" in m["type"]:
                            event_mentions_local.append(mention)
                            #else:
                            entity_mentions_local.append(mention)

                            summary_df.loc[len(summary_df)] = {
                                DOC_ID: doc_id,
                                COREF_CHAIN: entity_id,
                                DESCRIPTION: marker_comment,
                                MENTION_TYPE: None,
                                MENTION_FULL_TYPE: None,
                                MENTION_ID: marker_id,
                                TOKENS_STR: token_str
                            }

            # sets singleton-values and creates relations
            '''
            for mention_a in ee_mentions:
                change = False
                m_cnt = 0
                if mention_a.get("coref_chain") not in recorded_ent:
                    m_cnt += 1
                    relation = {"mention_" + str(m_cnt): mention_a.get("mention_id"),
                                }
                    # print("neuer Eintrag"+mention_a.get("coref_chain"))
                    change = True
                for mention_b in ee_mentions:
                    if mention_a.get("mention_id") != mention_b.get("mention_id"):
                        if mention_a.get("coref_chain") == mention_b.get("coref_chain") and mention_a.get(
                                "coref_chain") not in recorded_ent:
                            mention_a["is_singleton"] = False

                            m_cnt += 1
                            relation["mention_" + str(m_cnt)] = mention_b.get("mention_id")
                            change = True

                        elif mention_a.get("coref_chain") == mention_b.get("coref_chain"):
                            mention_a["is_singleton"] = False

                if change:
                    relation["relation_type"] = mention_a.get("mention_full_type")
                    relation["concept_id"] = mention_a.get("mention_type")  # -> identdegree
                    relations.append(relation)
                    recorded_ent.append(mention_a.get("coref_chain"))
            '''
            newsplease_custom = copy.copy(newsplease_format)

            newsplease_custom["title"] = title
            newsplease_custom["date_publish"] = None

            newsplease_custom["text"] = text
            newsplease_custom["source_domain"] = topic_file.split(".xml")[0]

            if newsplease_custom["title"][-1] not in string.punctuation:
                newsplease_custom["title"] += "."

            doc_files[topic_file.split(".")[0]] = newsplease_custom
            if topic_name not in os.listdir(result_path):
                os.mkdir(os.path.join(result_path, topic_name))

            with open(os.path.join(result_path, topic_name, newsplease_custom["source_domain"] + ".json"),
                      "w") as file:
                json.dump(newsplease_custom, file)

        #coref_dics[topic_dirs] = coref_dict

        annot_path = os.path.join(result_path, topic_name, "annotation",
                                  "original")  # ->root/data/NP4E+NiDENT-prep/test_parsing/topicName/annotation/original
        if topic_name not in os.listdir(os.path.join(result_path)):
            os.mkdir(os.path.join(result_path, topic_name))

        if "annotation" not in os.listdir(os.path.join(result_path, topic_name)):
            os.mkdir(os.path.join(result_path, topic_name, "annotation"))
            os.mkdir(annot_path)

        event_mentions.extend(event_mentions_local)
        entity_mentions.extend(entity_mentions_local)

        conll_topic_df = conll_df[conll_df[TOPIC_SUBTOPIC].str.contains(t_subt.split("/")[0])].reset_index(
            drop=True)

        # create a conll string from the conll_df
        LOGGER.info("Generating conll string for this topic...")
        for i, row in tqdm(conll_topic_df.iterrows(), total=conll_topic_df.shape[0]):
            if row[REFERENCE] is None:
                reference_str = "-"
            else:
                reference_str = row[REFERENCE]

            for mention in [m for m in entity_mentions]: # + event_mentions
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

            # if row[DOC_ID] == topic_name:  # do not overwrite conll rows of previous topic iterations
            conll_topic_df.at[i, REFERENCE] = reference_str

        # remove the leading characters if necessary (left from initialization)
        for i, row in conll_topic_df.iterrows():
            if row[REFERENCE].startswith("-| "):
                conll_topic_df.at[i, REFERENCE] = row[REFERENCE][3:]

        conll_topic_df = conll_topic_df.drop(columns=[DOC_ID])

        outputdoc_str = ""
        for (topic_local), topic_df in conll_topic_df.groupby(by=[TOPIC_SUBTOPIC]):
            outputdoc_str += f'#begin document ({topic_local}); part 000\n'

            for (sent_id_local), sent_df in topic_df.groupby(by=[SENT_ID], sort=[SENT_ID]):
                np.savetxt(os.path.join(NIDENT_PARSING_FOLDER, "tmp.txt"), sent_df.values, fmt='%s',
                           delimiter="\t",
                           encoding="utf-8")
                with open(os.path.join(NIDENT_PARSING_FOLDER, "tmp.txt"), "r", encoding="utf8") as file:
                    saved_lines = file.read()
                outputdoc_str += saved_lines + "\n"

            outputdoc_str += "#end document\n"

        # Check if the brackets ( ) are correct
        try:
            brackets_1 = 0
            brackets_2 = 0
            for i, row in conll_topic_df.iterrows():  # only count brackets in reference column (exclude token text)
                brackets_1 += str(row[REFERENCE]).count("(")
                brackets_2 += str(row[REFERENCE]).count(")")
            LOGGER.info(
                f"Amount of mentions in this topic: {str(len(entity_mentions_local))}")
            LOGGER.info(f"Total mentions parsed (all topics): {str(len(entity_mentions))}")   # + entity_mentions
            LOGGER.info(f"brackets '(' , ')' : {str(brackets_1)}, {str(brackets_2)}")
            assert brackets_1 == brackets_2
        except AssertionError:
            LOGGER.warning(
                f'Number of opening and closing brackets in conll does not match! topic: {str(topic_name)}')
            conll_topic_df.to_csv(os.path.join(annot_path, CONLL_CSV))
            with open(os.path.join(annot_path, f'{topic_name}.conll'), "w", encoding='utf-8') as file:
                file.write(outputdoc_str)
            # sys.exit()

        with open(os.path.join(annot_path, f'{topic_name}.conll'), "w", encoding='utf-8') as file:
            file.write(outputdoc_str)

        with open(os.path.join(annot_path, "entity_mentions_" + topic_name + ".json"), "w") as file:
            json.dump(entity_mentions_local, file)

        with open(os.path.join(annot_path, "event_mentions_" + topic_name + ".json"), "w") as file:
            json.dump(event_mentions_local, file)

        with open(os.path.join(annot_path, "mentions_" + topic_name + ".json"), "w") as file:
            json.dump(ee_mentions, file)

        #with open(os.path.join(annot_path, "relations.json"), "w") as file:
        #    json.dump(relations, file)

        #np.savetxt(os.path.join(annot_path, "information.txt"), conll_df.values, fmt='%s', delimiter="\t",
        #           header="topic/subtopic_name\tsent_id\ttoken_id\ttoken\tafter\tcoref")

    conll_df = conll_df.reset_index(drop=True)

    # create a conll string from the conll_df
    LOGGER.info("Generating conll string...")
    for i, row in tqdm(conll_df.iterrows(), total=conll_df.shape[0]):
        if row[REFERENCE] is None:
            reference_str = "-"
        else:
            reference_str = row[REFERENCE]

        for mention in [m for m in entity_mentions]:
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

        # if row[DOC_ID] == topic_name:  # do not overwrite conll rows of previous topic iterations
        conll_df.at[i, REFERENCE] = reference_str

    for i, row in conll_df.iterrows():  # remove the leading characters if necessary (left from initialization)
        if row[REFERENCE].startswith("-| "):
            conll_df.at[i, REFERENCE] = row[REFERENCE][3:]

    conll_df = conll_df.drop(columns=[DOC_ID])

    outputdoc_str = ""
    for (topic_local), topic_df in conll_df.groupby(by=[TOPIC_SUBTOPIC]):
        outputdoc_str += f'#begin document ({topic_local}); part 000\n'

        for (sent_id_local), sent_df in topic_df.groupby(by=[SENT_ID], sort=[SENT_ID]):
            np.savetxt(os.path.join(NIDENT_PARSING_FOLDER, "tmp.txt"), sent_df.values, fmt='%s', delimiter="\t",
                       encoding="utf-8")
            with open(os.path.join(NIDENT_PARSING_FOLDER, "tmp.txt"), "r", encoding="utf8") as file:
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
        LOGGER.info(f"Total mentions parsed (all topics): {str(len(entity_mentions))}")
        LOGGER.info(f"brackets '(' , ')' : {str(brackets_1)}, {str(brackets_2)}")
        assert brackets_1 == brackets_2
    except AssertionError:
        LOGGER.warning(f'Number of opening and closing brackets in conll does not match! topic: {str(topic_name)}')
        conll_df.to_csv(os.path.join(out_path, CONLL_CSV))
        with open(os.path.join(out_path, 'meantime.conll'), "w", encoding='utf-8') as file:
            file.write(final_output_str)
        # sys.exit()

    conll_df.to_csv(os.path.join(out_path, CONLL_CSV))

    LOGGER.info(
        "Mentions that need manual review to define the head and its attributes have been saved to: " +
        MANUAL_REVIEW_FILE + " - Total: " + str(len(need_manual_review_mention_head)))
    with open(os.path.join(out_path, MANUAL_REVIEW_FILE), "w", encoding='utf-8') as file:
        json.dump(need_manual_review_mention_head, file)

    with open(os.path.join(out_path, "conll_as_json.json"), "w", encoding='utf-8') as file:
        json.dump(conll_df.to_dict('records'), file)

    with open(os.path.join(out_path, 'nident.conll'), "w", encoding='utf-8') as file:
        file.write(final_output_str)

    with open(os.path.join(out_path, "entity_mentions.json"), "w") as file:
        json.dump(entity_mentions, file)

    with open(os.path.join(out_path, "event_mentions.json"), "w") as file:
        json.dump(event_mentions, file)

    summary_df.drop(columns=[MENTION_ID], inplace=True)
    summary_df.to_csv(os.path.join(out_path, MENTIONS_ALL_CSV))

    LOGGER.info(f'Parsing of NiDENT done!')


if __name__ == '__main__':

    LOGGER.info(f"Processing MEANTIME language {source_path[-34:].split('_')[2]}.")
    intra = os.path.join(source_path, 'intra-doc_annotation')
    intra_cross = os.path.join(source_path, 'intra_cross-doc_annotation')
    conv_files(source_path)


