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
                    info_t_name = str(word_file.split("_")[0])
                    t_subt = topic_name + "/" + info_t_name

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

                            conll_df = pd.concat([conll_df, pd.DataFrame({
                                TOPIC_SUBTOPIC: t_subt,
                                SENT_ID: sent_cnt,
                                TOKEN_ID: t_id,
                                TOKEN: elem.text,
                                REFERENCE: "-"
                            }, index=[0])])

                            if elem.text in "\".!?)]}'":
                                sent_cnt += 1

                        markables = ET.parse(os.path.join(path, "mmax2", topic_dir, "markables", word_file.split("_")[0]+"_coref_level.xml"))
                        markables_root = markables.getroot()

                        for markable in markables_root:
                            marker_id = markable.get("id")
                            span_str = markable.get("span")
                            coref_id = markable.get("coref_set")
                            if span_str == "p":
                                continue

                            # mention attributes
                            token_ids = []

                            if "." in span_str:
                                span_list = [int(span_str.split(".")[0].split("_")[1]),
                                             int(span_str.split(".")[2].split("_")[1])]
                                for word_id in range(span_list[0], span_list[1]+1):
                                    token_ids.append(word_id)
                            else:
                                span_list = [int(span_str.split(".")[0].split("_")[1])]
                                token_ids = [int(span_str.split("_")[1])]
                            token_numbers = []
                            tokens = {}
                            token_str = ""
                            tokens_text = []
                            for t_id in token_ids:
                                tokens[t_id] = {"text": token_dict[t_id]["text"], "sent": token_dict[t_id]["sent"], "id": token_dict[t_id]["id"]}
                                token_numbers.append(token_dict[t_id]["id"])
                                token_str, word_fixed, no_whitespace = append_text(token_str, str(token_dict[t_id]["text"]))
                                tokens_text.append(str(token_dict[t_id]["text"]))

                            doc_id = str(word_file.split(".xml")[0])

                            # determine the sentences as a string
                            sentence_str = ""
                            sent_tokens = []
                            sent_id = token_dict[span_list[0]]["sent"]
                            for t in token_dict:
                                if token_dict[t]["sent"] == sent_id:
                                    sentence_str, _, _ = append_text(sentence_str, token_dict[t]["text"])
                                    sent_tokens.append(token_dict[t]["id"])

                            # pass the string into spacy
                            doc = nlp(sentence_str)

                            mention_text = token_str
                            # if "tokens" has values -> fill the "mention" dict with the value of the corresponding m_id
                            if len(token_numbers):

                                # tokenize the mention text
                                mention_tokenized = []
                                for t_num in token_numbers:
                                    if t_num is not None:
                                        mention_tokenized.append(t_num)
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

                                counter = 0
                                while True:
                                    if counter > 50:  # an error must have occurred, so break and add to manual review
                                        need_manual_review_mention_head[str(topic_name) + "_" + str(mention_text)[:10]] = {
                                            "mention_text": mention_text,
                                            "sentence_str": sentence_str,
                                            "mention_head": "unknown",
                                            "mention_tokens_amount": len(token_numbers),
                                            "tolerance": tolerance
                                        }
                                        LOGGER.info(
                                            f"Mention with ID {str(topic_name)}_{str(mention_text)} needs manual review. Could not determine the mention head automatically \n(Exceeded max iterations). {str(tolerance)}")
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
                                            token_numbers)) <= tolerance and sentence_str[
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
                                if str(topic_name) + "_" + str(mention_text)[:10] not in need_manual_review_mention_head:
                                    for i in mention_doc_ids:
                                        ancestors_in_mention = 0
                                        for a in doc[i].ancestors:
                                            if a.i in mention_doc_ids:
                                                ancestors_in_mention = ancestors_in_mention + 1
                                                break  # one is enough to make the token inviable as a head
                                        if ancestors_in_mention == 0 and doc[
                                            i].text not in string.punctuation:  # puncts should not be heads
                                            # head within the mention
                                            mention_head = doc[i]
                                else:
                                    mention_head = doc[0]  # as placeholder for manual checking

                                mention_head_lemma = mention_head.lemma_
                                mention_head_pos = mention_head.pos_

                                mention_ner = mention_head.ent_type_
                                if mention_ner == "":
                                    mention_ner = "O"

                                # remap the mention head back to the np4e original tokenization to get the ID for the output
                                mention_head_id = None
                                mention_head_text = mention_head.text

                                for t in tokens:
                                    if str(tokens[t]["text"]).startswith(mention_head_text):
                                        mention_head_id = int(tokens[t]["id"])

                                if not mention_head_id:
                                    for t in tokens:
                                        if mention_head_text.startswith(str(tokens[t]["text"])):
                                            mention_head_id = int(tokens[t]["id"])
                                if not mention_head_id:
                                    for t in tokens:
                                        if str(tokens[t]["text"]).endswith(mention_head_text):
                                            mention_head_id = int(tokens[t]["id"])

                                # add to manual review if the resulting token is not inside the mention
                                # (error must have happened)
                                if mention_head_id not in sent_tokens:  # also "if is None"
                                    if str(topic_name) + "_" + str(mention_text) not in need_manual_review_mention_head:
                                        need_manual_review_mention_head[str(topic_name) + "_" + str(mention_text)[:10]] = \
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
                                        #[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]
                                        LOGGER.info(
                                            f"Mention with ID {str(topic_name)}_{str(mention_text)} needs manual review. Could not determine the mention head automatically. {str(tolerance)}")

                                # get the context
                                context_min_id, context_max_id = [
                                    0 if int(min(tokens)) - CONTEXT_RANGE < 0 else
                                    int(min(tokens)) - CONTEXT_RANGE,
                                    len(token_dict) - 1
                                    if int(max(tokens)) + CONTEXT_RANGE > len(
                                        token_dict)
                                    else int(max(tokens)) + CONTEXT_RANGE]

                                mention_context_str = []
                                break_indicator = False
                                # append to the mention context string list
                                for t in token_dict:
                                    if int(token_dict[t]["id"]) > context_max_id:  # break when all needed words processed
                                        break
                                    elif int(token_dict[t]["id"]) >= context_min_id and int(token_dict[t]["id"]) <= context_max_id:
                                        mention_context_str.append(token_dict[t]["text"])

                                # add to mentions if the variables are correct ( do not add for manual review needed )
                                if str(topic_name) + "_" + str(mention_text) not in need_manual_review_mention_head:
                                    mention = {COREF_CHAIN: coref_id,
                                               MENTION_NER: mention_ner,
                                               MENTION_HEAD_POS: mention_head_pos,
                                               MENTION_HEAD_LEMMA: mention_head_lemma,
                                               MENTION_HEAD: mention_head_text,
                                               MENTION_HEAD_ID: mention_head_id,
                                               DOC_ID: doc_id,
                                               DOC_ID_FULL: doc_id,
                                               IS_CONTINIOUS: token_numbers == list(
                                                   range(token_numbers[0], token_numbers[-1] + 1)),
                                               IS_SINGLETON: len(tokens) == 1,
                                               MENTION_ID: marker_id,
                                               MENTION_TYPE: "MIS",
                                               MENTION_FULL_TYPE: "MISC",
                                               SCORE: -1.0,
                                               SENT_ID: sent_id,
                                               MENTION_CONTEXT: mention_context_str,
                                               TOKENS_NUMBER: token_numbers,
                                               TOKENS_STR: token_str,
                                               TOKENS_TEXT: tokens_text,
                                               TOPIC_ID: cnt,
                                               TOPIC: t_subt.split("/")[0],
                                               SUBTOPIC: t_subt.split("/")[1],
                                               TOPIC_SUBTOPIC: t_subt,
                                               COREF_TYPE: "STRICT",
                                               DESCRIPTION: None,
                                               CONLL_DOC_KEY: t_subt
                                               }

                                    # np4e only has entities
                                    entity_mentions_local.append(mention)

                                    summary_df.loc[len(summary_df)] = {
                                        DOC_ID: doc_id,
                                        COREF_CHAIN: coref_id,
                                        DESCRIPTION: None,
                                        MENTION_TYPE: "MIS",
                                        MENTION_FULL_TYPE: "MISC",
                                        MENTION_ID: marker_id,
                                        TOKENS_STR: token_str
                                    }

                    newsplease_custom = copy.copy(newsplease_format)

                    newsplease_custom["title"] = title
                    newsplease_custom["date_publish"] = None

                    newsplease_custom["text"] = text
                    newsplease_custom["source_domain"] = word_file.split(".xml")[0]
                    if newsplease_custom["title"][-1] not in string.punctuation:
                        newsplease_custom["title"] += "."

                    doc_files[word_file.split(".")[0]] = newsplease_custom
                    if topic_name not in os.listdir(result_path):
                        os.mkdir(os.path.join(result_path, topic_name))

                    with open(os.path.join(result_path, topic_name, newsplease_custom["source_domain"] + ".json"),
                              "w") as file:
                        json.dump(newsplease_custom, file)

                    annot_path = os.path.join(result_path, topic_name, "annotation",
                                              "original")
                    if topic_name not in os.listdir(os.path.join(result_path)):
                        os.mkdir(os.path.join(result_path, topic_name))

                    if "annotation" not in os.listdir(os.path.join(result_path, topic_name)):
                        os.mkdir(os.path.join(result_path, topic_name, "annotation"))
                        os.mkdir(annot_path)

        entity_mentions.extend(entity_mentions_local)

        conll_topic_df = conll_df[conll_df[TOPIC_SUBTOPIC].str.contains(t_subt.split("/")[0])].reset_index(
            drop=True)

        # create a conll string from the conll_df   (per topic)
        LOGGER.info("Generating conll string for this topic...")
        for i, row in tqdm(conll_topic_df.iterrows(), total=conll_topic_df.shape[0]):
            if row[REFERENCE] is None:
                reference_str = "-"
            else:
                reference_str = row[REFERENCE]

            for mention in [m for m in entity_mentions]:
                if mention[TOPIC_SUBTOPIC] == row[TOPIC_SUBTOPIC] and mention[SENT_ID] == row[SENT_ID] and \
                        row[
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
                np.savetxt(os.path.join(NP4E_PARSING_FOLDER, "tmp.txt"), sent_df.values, fmt='%s',
                           delimiter="\t",
                           encoding="utf-8")
                with open(os.path.join(NP4E_PARSING_FOLDER, "tmp.txt"), "r", encoding="utf8") as file:
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
            LOGGER.info(
                f"Total mentions parsed (all topics): {str(len(entity_mentions))}")  # + entity_mentions
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

        with open(os.path.join(annot_path, "entities_mentions_" + topic_name + ".json"), "w") as file:
            json.dump(entity_mentions_local, file)

    conll_df = conll_df.reset_index(drop=True)

    # create a conll string from the conll_df (all topics combined)
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
            np.savetxt(os.path.join(NP4E_PARSING_FOLDER, "tmp.txt"), sent_df.values, fmt='%s',
                       delimiter="\t",
                       encoding="utf-8")
            with open(os.path.join(NP4E_PARSING_FOLDER, "tmp.txt"), "r", encoding="utf8") as file:
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
        LOGGER.warning(
            f'Number of opening and closing brackets in conll does not match! topic: {str(topic_name)}')
        conll_df.to_csv(os.path.join(out_path, CONLL_CSV))
        with open(os.path.join(out_path, 'np4e.conll'), "w", encoding='utf-8') as file:
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

    with open(os.path.join(out_path, 'np4e.conll'), "w", encoding='utf-8') as file:
        file.write(final_output_str)

    with open(os.path.join(out_path, MENTIONS_ENTITIES_JSON), "w") as file:
        json.dump(entity_mentions, file)

    summary_df.drop(columns=[MENTION_ID], inplace=True)
    summary_df.to_csv(os.path.join(out_path, MENTIONS_ALL_CSV))

    LOGGER.info(f'Parsing of NP4E done!')


if __name__ == '__main__':

    LOGGER.info(f"Processing NP4E: {source_path[-34:].split('_')[2]}.")
    conv_files(source_path)


