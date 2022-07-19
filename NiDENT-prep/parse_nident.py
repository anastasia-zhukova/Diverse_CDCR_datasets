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
out_path = os.path.join(OUT_PATH, 'en')

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
    summary_conversion_df = pd.DataFrame()
    conll_df = pd.DataFrame(columns=[TOPIC_SUBTOPIC, DOC_ID, SENT_ID, TOKEN_ID, TOKEN, REFERENCE])
    final_output_str = ""
    need_manual_review_mention_head = {}

    for topic_dirs in dirs:     # corpora
        cnt = cnt + 1
        topic_name = str(cnt) + "NiDENT"

        doc_type = "nident"
        coref_dict = {}
        print("Parsing of NiDENT. This process can take several minutes. Please wait ...")
        topic_files = os.listdir(os.path.join(path,"english-corpus"))

        coref_dict = {}

        conll_df = pd.DataFrame()
        for topic_file in tqdm(topic_files):
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
                        text, word_fixed, no_whitespace = append_text(text, word.attrib["wd"])

                # Markables
                markables = sentence.findall('.//sn')

                for markable in markables:

                    mark_counter += 1
                    marker_id = markable.get("markerid")
                    entity_id = markable.get("entity")
                    coref_id = markable.get("corefid")
                    token_numbers = []
                    token_int_numbers = []
                    token_str = ""
                    markable_words = markable.findall('.//word')
                    for word in markable_words:
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

                    # determine the sentences as a string
                    tokens = sentence.findall('.//word')

                    for t in tokens:
                        print(ET.tostring(t))

                    sent_tokens = [int(t.get("wdid")[1:]) for t in tokens if t.get("wdid") is not None]
                    tokens_str = []
                    sentence_str = ""
                    for i, t in enumerate(tokens):
                        tokens_str.append(t.attrib["wd"])
                        if ("wdid" in t):
                            t.attrib["t_id"] = int(t.attrib["wdid"][1:])
                        elif i == 0:
                            t.attrib["t_id"] = 0
                        else:
                            t.attrib["t_id"] = tokens[i - 1].attrib["t_id"] + 1

                        if t.attrib["pos"] == "PUNCT" or i == 0:
                            sentence_str = sentence_str + t.attrib["wd"]
                        else:
                            sentence_str = sentence_str + " " + t.attrib["wd"]

                    # pass the string into spacy
                    doc = nlp(sentence_str)
                    token_ids_in_doc = []

                    # skip if the token is contained more than once within the same mention
                    # (i.e. ignore entries with error in meantime tokenization)
                    if len(tokens) != len(list(set(tokens))):
                        continue

                    mention_text = ""
                    for t in tokens:
                        mention_text, _, _ = append_text(mention_text, t.get("wd"))
                    # if "tokens" has values -> fill the "mention" dict with the value of the corresponding m_id
                    if len(tokens):

                        # generate sentence doc with spacy
                        sentence_str = ""
                        for t in root:
                            if t.tag == TOKEN and t.attrib[SENTENCE] == str(sent_id):
                                sentence_str, _, _ = append_text(sentence_str, t.text)
                        doc = nlp(sentence_str)

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
                            # handle special case if the last punctuation is part of mention in ecb
                            last_char_of_mention = len(sentence_str)

                        counter = 0
                        while True:
                            if counter > 50:  # an error must have occurred, so break and add to manual review
                                need_manual_review_mention_head[str(t_subt) + "_" + str(mention_text)] = {
                                    "mention_text": mention_text,
                                    "sentence_str": sentence_str,
                                    "mention_head": "unknown",
                                    "mention_tokens_amount": len(tokens),
                                    "tolerance": tolerance
                                }
                                LOGGER.info(
                                    f"Mention with ID {str(t_subt)}_{str(mention_text)} needs manual review. Could not determine the mention head automatically. {str(tolerance)}")
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

                        # remap the mention head back to the meantime original tokenization to get the ID for the output
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
                                with open(os.path.join(out_path, MANUAL_REVIEW_FILE),
                                          "w",
                                          encoding='utf-8') as file:
                                    json.dump(need_manual_review_mention_head, file)
                                LOGGER.info(
                                    f"Mention with ID {str(t_subt)}_{str(mention_text)} needs manual review. Could not determine the mention head automatically. {str(tolerance)}")

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
                            if t.tag == "token" and int(t.attrib["t_id"]) >= context_min_id and int(
                                    t.attrib["t_id"]) <= context_max_id:
                                mention_context_str.append(t.text)

                        # tokens text
                        tokens_text = []
                        for t in tokens:
                            tokens_text = str(token_dict[t][TEXT])

                        # add to mentions if the variables are correct ( do not add for manual review needed )
                        if str(t_subt) + "_" + str(mention_text) not in need_manual_review_mention_head:
                            mentions[topic_file.split(".")[0]+str(sent_id)+str(marker_id)] = {
                                "type": "-",    #subelem.tag
                                "text": " ".join(
                                    [token_dict[t]["text"] for t in tokens]),
                                "sent_doc": doc,
                                MENTION_NER: mention_ner,
                                MENTION_HEAD_POS: mention_head_pos,
                                MENTION_HEAD_LEMMA: mention_head_lemma,
                                MENTION_HEAD: mention_head.text,
                                MENTION_HEAD_ID: mention_head.i,
                                TOKENS_NUMBER: sent_tokens,
                                TOKENS_TEXT: tokens_text,
                                DOC_ID: topic_file.split(".")[0],
                                SENT_ID: int(sent_id),
                                MENTION_CONTEXT: mention_context_str,
                                TOPIC_SUBTOPIC: t_subt}

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

    LOGGER.info(f'Parsing of NiDENT done!')


if __name__ == '__main__':

    LOGGER.info(f"Processing MEANTIME language {source_path[-34:].split('_')[2]}.")
    intra = os.path.join(source_path, 'intra-doc_annotation')
    intra_cross = os.path.join(source_path, 'intra_cross-doc_annotation')
    conv_files(source_path)


