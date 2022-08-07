import xml.etree.ElementTree as ET
import os
import json
import string
import copy
import re
import pandas as pd
import numpy as np
from datetime import datetime
from nltk import Tree
from insert_whitespace import append_text
from tqdm import tqdm
from config import DATA_PATH
import spacy


path_sample = os.path.join(DATA_PATH, "_sample.json")  # ->root/data/original/_sample.json

with open(path_sample, "r") as file:
    newsplease_format = json.load(file)

source_path = os.path.join(DATA_PATH, 'NP4E+NiDENT-prep')
result_path = os.path.join(source_path, 'test_parsing')
out_path = os.path.join(source_path, "output_data")
nident = os.path.join(source_path, 'NiDENT\\')
np4e = os.path.join(source_path, 'NP4E', 'mmax2\\')

nlp = spacy.load('en_core_web_sm')

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_

def conv_files(path):
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

    print("Parsing of NP4E. This process can take several minutes. Please wait ...")
    cnt = cnt + 1
    topic_name = str(cnt) + "NP4E"
    for topic_dirs in dirs:
        # print(topic_dirs)
        topics_p = os.listdir(np4e + topic_dirs)
        for i, topic in enumerate(topics_p):
            if topic == "Basedata":
                word_files = os.listdir(np4e + topic_dirs + '/' + topic)
                title, text, url, time, time2, time3 = "", "", "", "", "", ""

                for word_file in word_files:

                    if word_file.split(".")[1] == "xml":
                        tree = ET.parse(np4e + topic_dirs + '/' + topic + '/' + word_file)
                        root = tree.getroot()
                        title, text, url, time, time2, time3 = "", "", "", "", "", ""

                        token_dict, mentions, mentions_map = {}, {}, {}
                        coref_dict = {}

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
                            # print(elem.text)
                            token_dict[word_count] = {"text": elem.text, "sent": sent_cnt,
                                                      "id": t_id}
                            if int(sent_cnt) == 0:
                                title = append_text(title, elem.text)
                            else:
                                text = append_text(text, elem.text)

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

                    coref_dics[topic_dirs] = coref_dict

                    entity_mentions = []
                    event_mentions = []

                    annot_path = os.path.join(result_path, topic_name, "annotation",
                                              "original")  # ->root/data/NP4E+NiDENT-prep/test_parsing/topicName/annotation/original
                    if topic_name not in os.listdir(os.path.join(result_path)):
                        os.mkdir(os.path.join(result_path, topic_name))

                    if "annotation" not in os.listdir(os.path.join(result_path, topic_name)):
                        os.mkdir(os.path.join(result_path, topic_name, "annotation"))
                        os.mkdir(annot_path)

                    with open(os.path.join(annot_path, "entity_mentions_" + topic_name + ".json"), "w") as file:
                        json.dump(entity_mentions, file)

                    with open(os.path.join(annot_path, "event_mentions_" + topic_name + ".json"), "w") as file:
                        json.dump(event_mentions, file)


    if __name__ == '__main__':
    print('Please enter the number of the set, you want to convert:\n'
          '   1 NiDENT\n'
          '   2 NP4E\n'
          '   3 both')


    def choose_input():
        setnumber = input()
        if setnumber == "1":
            c_format = "NiDENT"
            conv_files(nident)
            return c_format
        elif setnumber == "2":
            c_format = "NP4E"
            conv_files(np4e)
            return c_format
        elif setnumber == "3":
            c_format = "NiDENT + NP4E"
            conv_files(nident)
            conv_files(np4e)
            return c_format
        else:
            print("Please chose one of the 3 numbers!")
            return choose_input()


    co_format = choose_input()

    print("Conversion of {0} from xml to newsplease format and to annotations in a json file is "
          "done. \n\nFiles are saved to {1}."
          "{2}.".format(co_format, result_path, DATA_PATH))
