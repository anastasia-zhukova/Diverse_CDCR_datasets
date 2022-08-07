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
    if "NiDENT" in path:
        # topic_files = os.listdir(path)
        for topic_dirs in dirs:
            cnt = cnt + 1
            topic_name = str(cnt) + "NiDENT"

            doc_type = "nident"
            coref_dict = {}
            print("Parsing of NiDENT. This process can take several minutes. Please wait ...")
            topic_files = os.listdir(path + topic_dirs)

            coref_dict = {}

            conll_df = pd.DataFrame()
            for topic_file in tqdm(topic_files):
                tree = ET.parse(path + topic_dirs + '/' + topic_file)
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
                    sent_id += 1

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

                        #information.txt-dataframe construction
                        conll_df = conll_df.append(pd.DataFrame({
                            "topic/subtopic_name": t_subt,
                            "sent_id": sent_id,
                            "token_id": token_id,
                            "token": word.get("wd"),
                            # "after": "\"" + append_text(prev_word, word.get("wd"), "space") + "\"",
                            "coref": "-"
                        }, index=[0]))

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
                            text, word_fixed, no_whitespace  = append_text(text, word.attrib["wd"])

                    #Markables
                    markables = sentence.findall('.//sn')

                    for markable in markables:

                        mark_counter += 1
                        token_numbers = []
                        token_int_numbers = []
                        token_str = ""
                        markable_words = markable.findall('.//word')
                        for word in markable_words:
                            m_word_cnt = 0
                            #get counted word-id(first called "number") in sentence for every word in the markable
                            #if punctuation (wich does not have a wdid) then count last word +1 or if first set number = 0
                            for m_word in sent_word_ar:
                                if m_word.get("wdid") == word.get("wdid"):
                                    number = m_word_cnt
                                    if word.get("wdid") is None and token_numbers:
                                        number = token_numbers[len(token_numbers)-1] + 1
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

                        for num in token_numbers:
                            if num is not None:
                                token_int_numbers.append(int(num))
                            else:
                                token_int_numbers.append(0)

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

                        #determine the sentences as a string
                        tokens = sentence.findall('.//word')
                        tokens_str = []
                        sentence_str = ""
                        for i, t in enumerate(tokens):
                            tokens_str.append(t.attrib["wd"])
                            if ("wdid" in t):
                                t.attrib["t_id"] = int(t.attrib["wdid"][1:])
                            elif i == 0:
                                t.attrib["t_id"] = 0
                            else:
                                t.attrib["t_id"] = tokens[i-1].attrib["t_id"] + 1
                            
                            if t.attrib["pos"] == "PUNCT" or i == 0:
                                sentence_str = sentence_str + t.attrib["wd"]
                            else:
                                sentence_str = sentence_str + " " + t.attrib["wd"]
                        
                        #pass the string into spacy
                        doc = nlp(sentence_str)
                        token_ids_in_doc = []

                        for to in doc:
                                if len(token_ids_in_doc) == len(tokens):
                                    break
                                for i, ts in enumerate(tokens_str):
                                    if len(token_ids_in_doc) == len(tokens):
                                        break
                                    if (to.text.startswith(ts) or to.text.endswith(ts)):

                                        if len(token_ids_in_doc) > 0:
                                            if i >= 1 and i < len(tokens):
                                                diff_tokens = int(tokens[i].attrib["t_id"])-int(tokens[i-1].attrib["t_id"])   #the difference between tokens in mention in datasets tokenization
                                            else:
                                                diff_tokens = 1

                                            if (to.i not in token_ids_in_doc and abs(to.i - token_ids_in_doc[-1]) <= diff_tokens+1 ):  #account for small differences in tokenization 
                                                token_ids_in_doc.append(to.i)
                                            if abs(to.i - token_ids_in_doc[-1]) > diff_tokens+1 and len(token_ids_in_doc) < len(tokens):
                                                print("RESET NECCESARY AT ID " + str(to.i))
                                                print(abs(to.i - token_ids_in_doc[-1]))
                                                print(len(token_ids_in_doc))
                                                print(len(tokens))
                                                token_ids_in_doc = [to.i]   #reset
                                        else:
                                            token_ids_in_doc.append(to.i)

                        if len(token_ids_in_doc) == 0:  #if no token has been found, set the condition more broadly
                            for to in doc:
                                if len(token_ids_in_doc) == len(tokens):
                                    break
                                for ts in tokens_str:
                                    if len(token_ids_in_doc) == len(tokens):
                                        break
                                    if (to.text.startswith(ts) or to.text.endswith(ts)):

                                        if len(token_ids_in_doc) > 0:
                                            if (to.i not in token_ids_in_doc and abs(to.i - token_ids_in_doc[-1]) <= 2):  #account for small differences in tokenization 
                                                token_ids_in_doc.append(to.i)
                                            if abs(to.i - token_ids_in_doc[-1]) > 2 and len(token_ids_in_doc) < len(tokens):
                                                print("RESET NECCESARY AT ID " + str(to.i))
                                                token_ids_in_doc = [to.i]   #reset
                                        else:
                                            token_ids_in_doc.append(to.i)
                        
                        mention_head = "NONE FOUND"

                        for i in token_ids_in_doc:
                            ancestors_in_mention = 0
                            for a in doc[i].ancestors:
                                if a.i in token_ids_in_doc:
                                    ancestors_in_mention = ancestors_in_mention + 1
                                    break   #one is enough to make the token unviable as a head
                            if ancestors_in_mention == 0:
                                #head within the mention
                                mention_head = doc[i]


                        print(sentence)
                        [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]
                        print(tokens_str)
                        print(doc)
                        print(token_ids_in_doc)
                        print(token_numbers)
                        print(mention_head)
                        print("___________________________-")

                        #determine the head (by looking at spacy ancestors of the tokens)

                        # print("Marker_id:" + str(markable.attrib["markerid"]) + " Chain_id" + str(chain_id))
                        mention = {"coref_chain": entity,
                                   "doc_id": doc_id,
                                   "is_continuous": True if token_int_numbers == list(
                                       range(token_int_numbers[0], token_int_numbers[-1] + 1))
                                   else False,
                                   "is_singleton": True,
                                   "mention_id": doc_id + "_" + entity + "_" + markable.get("markerid"),
                                   "mention_type": str(markable.get("identdegree")),
                                   "mention_full_type": mention_full_type,
                                   "score": -1.0,
                                   # "sent_id": re.findall(r'\d+', sentence.attrib["id"])[0],  /does not provide an ascending sequence of numbers
                                   "sent_id": sent_id,
                                   "tokens_number": token_numbers,
                                   "tokens_str": token_str,
                                   "topic_id": topic_name,
                                   "coref_type": "-",  # no Information in xml given
                                   "decription": str(markable.get("corefcmt")),
                                   }

                        if token_numbers[0] == token_numbers[len(token_numbers) - 1]:
                            # information_df.loc[(information_df['sent_id'] == sent_id) & (information_df['token_id'] == token_numbers[0]), 'coref'] = "TESTES!!!"
                            if str(conll_df.loc[(conll_df['sent_id'] == sent_id) & (
                                    conll_df['token_id'] == token_numbers[0]) & (conll_df[
                                                                                        'topic/subtopic_name'] == t_subt), 'coref'].values[
                                       0]) == "-":

                                conll_df.loc[(conll_df['sent_id'] == sent_id) & (
                                            conll_df['token_id'] == token_numbers[0]) & (conll_df[
                                                                                                'topic/subtopic_name'] == t_subt), 'coref'] = '(' + str(
                                    mark_counter) + ')'

                            else:
                                conll_df.loc[(conll_df['sent_id'] == sent_id) & (
                                            conll_df['token_id'] == token_numbers[0]) & (conll_df[
                                                                                                'topic/subtopic_name'] == t_subt), 'coref'] = str(
                                    conll_df.loc[(conll_df['sent_id'] == sent_id) & (
                                                conll_df['token_id'] == token_numbers[0]) & (conll_df[
                                                                                                    'topic/subtopic_name'] == t_subt), 'coref'].values[
                                        0]) + ', (' + str(mark_counter) + ')'

                        else:
                            # if there stands "-" at the location with the df with the right token and sentence and topic-file, hence first entry, then for the 1st token of the arrays:
                            if str(conll_df.loc[(conll_df['sent_id'] == sent_id) & (
                                    conll_df['token_id'] == token_numbers[0]) & (conll_df[
                                                                                        'topic/subtopic_name'] == t_subt), 'coref'].values[
                                       0]) == "-":
                                conll_df.loc[(conll_df['sent_id'] == sent_id) & (
                                            conll_df['token_id'] == token_numbers[0]) & (conll_df[
                                                                                                'topic/subtopic_name'] == t_subt), 'coref'] = '(' + str(
                                    mark_counter)

                            else:
                                conll_df.loc[(conll_df['sent_id'] == sent_id) & (
                                            conll_df['token_id'] == token_numbers[0]) & (conll_df[
                                                                                                'topic/subtopic_name'] == t_subt), 'coref'] = str(
                                    conll_df.loc[(conll_df['sent_id'] == sent_id) & (
                                                conll_df['token_id'] == token_numbers[0]) & (conll_df[
                                                                                                    'topic/subtopic_name'] == t_subt), 'coref'].values[
                                        0]) + ', (' + str(mark_counter)

                            # if there stands "-" at the location with the df with the right token and sentence and topic-file, hence first entry, then for the last token of the arrays:
                            if str(conll_df.loc[(conll_df['sent_id'] == sent_id) & (
                                    conll_df['token_id'] == token_numbers[len(token_numbers) - 1]) & (conll_df[
                                                                                                          'topic/subtopic_name'] == t_subt), 'coref'].values[
                                       0]) == "-":
                                conll_df.loc[(conll_df['sent_id'] == sent_id) & (
                                            conll_df['token_id'] == token_numbers[len(token_numbers) - 1]) & (
                                                               conll_df[
                                                                   'topic/subtopic_name'] == t_subt), 'coref'] = str(
                                    mark_counter) + ')'

                            else:
                                conll_df.loc[(conll_df['sent_id'] == sent_id) & (
                                            conll_df['token_id'] == token_numbers[len(token_numbers) - 1]) & (
                                                               conll_df[
                                                                   'topic/subtopic_name'] == t_subt), 'coref'] = str(
                                    conll_df.loc[(conll_df['sent_id'] == sent_id) & (
                                                conll_df['token_id'] == token_numbers[len(token_numbers) - 1]) & (
                                                                   conll_df[
                                                                       'topic/subtopic_name'] == t_subt), 'coref'].values[
                                        0]) + ', ' + str(mark_counter) + ')'

                        ee_mentions.append(mention)

                # sets singleton-values and creates relations

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
                            if mention_a.get("coref_chain") == mention_b.get("coref_chain") and mention_a.get("coref_chain") not in recorded_ent:
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

                coref_dics[topic_dirs] = coref_dict

                annot_path = os.path.join(result_path, topic_name, "annotation",
                                          "original")  # ->root/data/NP4E+NiDENT-prep/test_parsing/topicName/annotation/original
                if topic_name not in os.listdir(os.path.join(result_path)):
                    os.mkdir(os.path.join(result_path, topic_name))

                if "annotation" not in os.listdir(os.path.join(result_path, topic_name)):
                    os.mkdir(os.path.join(result_path, topic_name, "annotation"))
                    os.mkdir(annot_path)

                with open(os.path.join(out_path, "entity_mentions_" + topic_name + ".json"), "w") as file:
                    json.dump(entity_mentions, file)

                with open(os.path.join(out_path, "event_mentions_" + topic_name + ".json"), "w") as file:
                    json.dump(event_mentions, file)

                with open(os.path.join(out_path, "mentions_" + topic_name + ".json"), "w") as file:
                    json.dump(ee_mentions, file)

                with open(os.path.join(annot_path, "relations.json"), "w") as file:
                    json.dump(relations, file)

                np.savetxt(os.path.join(annot_path, "information.txt"), conll_df.values, fmt='%s', delimiter="\t",
                           header="topic/subtopic_name\tsent_id\ttoken_id\ttoken\tafter\tcoref")

    else:
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
