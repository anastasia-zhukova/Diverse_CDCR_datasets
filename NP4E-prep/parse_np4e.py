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
from logger import LOGGER

path_sample = os.path.join(DATA_PATH, "_sample_doc.json")  # ->root/data/original/_sample_doc.json
NP4E_PARSING_FOLDER = os.path.join(os.getcwd())
OUT_PATH = os.path.join(NP4E_PARSING_FOLDER, OUTPUT_FOLDER_NAME)
CONTEXT_RANGE = 250

nlp = spacy.load('en_core_web_sm')

# opens and loads the newsplease-format out of the json file: _sample_doc.json
with open(path_sample, "r") as file:
    newsplease_format = json.load(file)


source_path = os.path.join(NP4E_PARSING_FOLDER, NP4E_FOLDER_NAME)
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

    for topic_dir in os.listdir(os.path.join(path, "mmax2")):
        cnt = cnt + 1
        entity_mentions_local = []
        event_mentions_local = []

        LOGGER.info(f"Parsing of NP4E topic {topic_dir}...")
        topic_name = str(topic_dir)

        for file in tqdm(os.listdir(os.path.join(path, "mmax2", topic_dir))):

            if file == "Basedata":
                title, text, url, time, time2, time3 = "", "", "", "", "", ""

                for word_file in os.listdir(os.path.join(path, "mmax2", topic_dir, file)):

                    if word_file.split(".")[1] == "xml":
                        tree = ET.parse(os.path.join(path, "mmax2", topic_dir, "Basedata", word_file))
                        root = tree.getroot()
                        title, text, url, time, time2, time3 = "", "", "", "", "", ""

                        token_dict, mentions, mentions_map = {}, {}, {}

                        t_id = -1
                        old_sent = 0
                        word_count = 0
                        sent_cnt = 0
                        for elem in root:
                            # correct sentence-endings
                            word_count += 1
                            if old_sent == int(sent_cnt):
                                t_id += 1
                            else:
                                old_sent = int(sent_cnt)
                                t_id = 0
                            token_dict[word_count] = {"text": elem.text, "sent": sent_cnt,
                                                      "id": t_id}
                            if int(sent_cnt) == 0:
                                title, word_fixed, no_whitespace = append_text(title, elem.text)
                            else:
                                text, word_fixed, no_whitespace = append_text(text, elem.text)

                            if elem.text in "\".!?)]}'":
                                sent_cnt += 1
                    # TODO: Markables with relations for NP4E (should be a similar result as in NiDENT)

                    newsplease_custom = copy.copy(newsplease_format)

                    newsplease_custom["title"] = title
                    newsplease_custom["date_publish"] = None

                    newsplease_custom["text"] = text
                    newsplease_custom["source_domain"] = word_file.split(".xml")[0]
                    # print(topic_file.split(".xml")[0])
                    if newsplease_custom["title"][-1] not in string.punctuation:
                        newsplease_custom["title"] += "."

                    doc_files[word_file.split(".")[0]] = newsplease_custom
                    if topic_name not in os.listdir(result_path):
                        os.mkdir(os.path.join(result_path, topic_name))

                    with open(os.path.join(result_path, topic_name, newsplease_custom["source_domain"] + ".json"),
                              "w") as file:
                        json.dump(newsplease_custom, file)

                    annot_path = os.path.join(result_path, topic_name, "annotation",
                                              "original")  # ->root/data/NP4E+NiDENT-prep/test_parsing/topicName/annotation/original
                    if topic_name not in os.listdir(os.path.join(result_path)):
                        os.mkdir(os.path.join(result_path, topic_name))

                    if "annotation" not in os.listdir(os.path.join(result_path, topic_name)):
                        os.mkdir(os.path.join(result_path, topic_name, "annotation"))
                        os.mkdir(annot_path)

                    with open(os.path.join(annot_path, "entity_mentions_" + topic_name + ".json"), "w") as file:
                        json.dump(entity_mentions_local, file)

                    with open(os.path.join(annot_path, "event_mentions_" + topic_name + ".json"), "w") as file:
                        json.dump(event_mentions_local, file)

    LOGGER.info(f'Parsing of NP4E done!')


if __name__ == '__main__':

    LOGGER.info(f"Processing NP4E: {source_path[-34:].split('_')[2]}.")
    conv_files(source_path)


