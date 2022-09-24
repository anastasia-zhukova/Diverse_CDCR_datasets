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
from insert_whitespace import append_text
from nltk import Tree
from tqdm import tqdm
import warnings
from setup import *
from logger import LOGGER

path_sample = os.path.join(DATA_PATH, "_sample_doc.json")  # ->root/data/original/_sample_doc.json
GVC_PARSING_FOLDER = os.path.join(os.getcwd())
OUT_PATH = os.path.join(GVC_PARSING_FOLDER, OUTPUT_FOLDER_NAME)
source_path = os.path.join(GVC_PARSING_FOLDER, GVC_FOLDER_NAME)
result_path = os.path.join(OUT_PATH, 'test_parsing')

CONTEXT_COMB_SPAN = 5

nlp = spacy.load('en_core_web_sm')

# opens and loads the newsplease-format out of the json file: _sample_doc.json
with open(path_sample, "r") as file:
    newsplease_format = json.load(file)

def conv_files(path):
    """
        Converts the given dataset for a specified language into the desired format.
        :param paths: The paths desired to process (intra- & cross_intra annotations)
        :param result_path: the test_parsing folder for that language
        :param out_path: the output folder for that language
        :param language: the language to process
        :param nlp: the spacy model that fits the desired language
    """

    conll_df = pd.DataFrame(columns=[TOPIC_SUBTOPIC, DOC_ID, SENT_ID, TOKEN_ID, TOKEN, REFERENCE])

    LOGGER.info("Reading gold.conll...")
    mention_identifiers = []

    with open(os.path.join(path, 'gold.conll'), encoding="utf-8") as f:
        conll_str = f.read()
        conll_lines = conll_str.split("\n")
        for i, conll_line in tqdm(enumerate(conll_lines), total=len(conll_lines)):
            if i + 1 == len(conll_lines):
                break
            if "#begin document" in conll_line or "#end document" in conll_line:
                continue
            if conll_line.split("\t")[0].split(".")[1] == "DCT":
                continue

            if ")" in conll_line.split("\t")[3]:
                mention_identifiers.append(conll_line.split("\t")[3].replace('(', '').replace(')', '') + "_" + str(
                    conll_line.split("\t")[0].split(".")[1][1:]))  # format: corefID_sentID

            conll_df = pd.concat([conll_df, pd.DataFrame({
                TOPIC_SUBTOPIC: conll_line.split("\t")[0].split(".")[0],
                SENT_ID: int(conll_line.split("\t")[0].split(".")[1][1:]),
                TOKEN_ID: int(conll_line.split("\t")[0].split(".")[2]),
                TOKEN: conll_line.split("\t")[1],
                REFERENCE: conll_line.split("\t")[3]
            }, index=[0])])

    print(conll_df.head(10))

    for mention_identifier in mention_identifiers:
        coref_id = mention_identifier.split("_")[0]
        sent_id = mention_identifier.split("_")[1]

        sent_conll = conll_df[conll_df[SENT_ID] == sent_id]
        sentence_str = ""
        mention_tokenized = []
        split_mention_text = []
        mention_text = ""
        for i, row in sent_conll.iterrows():
            sentence_str, word_fixed, no_whitespace = append_text(sentence_str, row[TOKEN])
            mention_tokenized.append(row[TOKEN_ID])
            if coref_id in row[REFERENCE]:
                mention_text, word_fixed, no_whitespace = append_text(mention_text, row[TOKEN])
                split_mention_text.append(row[TOKEN])

        doc = nlp(sentence_str)

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

        print(mention_text)
        print(sentence_str)
        print(first_char_of_mention)
        print(last_char_of_mention)

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
                tolerance = 0
                if tolerance > 2:
                    tolerance = 2
                # tolerance for website mentions
                if ".com" in mention_text or ".org" in mention_text:
                    tolerance = tolerance + 2
                # tolerance when the mention has external tokens inbetween mention tokens
                tolerance = tolerance \
                            + int(count_punct(token_str)) \
                            + 1

                if abs(len(re.split(" ", sentence_str[
                                         first_char_of_mention:last_char_of_mention])) - len(
                    markable_words)) <= tolerance and sentence_str[
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
                # i.g. do not accept "her" as a full word if the next letter is "s" ("herself")
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
                if ancestors_in_mention == 0 and doc[i].text not in string.punctuation:  # puncts should not be heads
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
                                          int(max(token_sent_numbers)) + CONTEXT_RANGE]

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
                elif int(word.get("wdid")[1:]) > context_max_id:  # break when all needed words processed
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
                       MENTION_TYPE: "MIS",
                       MENTION_FULL_TYPE: "MISC",
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
                       COREF_TYPE: coref_type,
                       DESCRIPTION: marker_comment,
                       CONLL_DOC_KEY: t_subt
                       }

            # nident only has entities
            entity_mentions_local.append(mention)

            summary_df.loc[len(summary_df)] = {
                DOC_ID: doc_id,
                COREF_CHAIN: entity_id,
                DESCRIPTION: marker_comment,
                MENTION_TYPE: "MIS",
                MENTION_FULL_TYPE: "MISC",
                MENTION_ID: marker_id,
                TOKENS_STR: token_str
            }





if __name__ == '__main__':
    LOGGER.info(f"Processing GVC {source_path[-34:].split('_')[2]}.")
    conv_files(source_path)
