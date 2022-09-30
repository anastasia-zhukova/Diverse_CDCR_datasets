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
sys.path.insert(0, '..')
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
    Converts the given dataset into the desired format.
    :param path: The path desired to process (intra- & cross_intra annotations)
    """
    conll_df = pd.DataFrame(
        columns=[TOPIC_SUBTOPIC, DOC_ID, SENT_ID, TOKEN_ID, "doc_token_id", TOKEN, "title_text_id", REFERENCE])
    summary_df = pd.DataFrame(
        columns=[DOC_ID, COREF_CHAIN, DESCRIPTION, MENTION_TYPE, MENTION_FULL_TYPE, MENTION_ID, TOKENS_STR])

    # --> process the datasets conll into a pd dataframe for processing
    LOGGER.info("Reading verbose.conll...")
    mention_identifiers = []
    prev_doc_id = "placeholder"
    doc_token_counter = int(0)
    doc_ids = []
    with open(os.path.join(path, 'verbose.conll'), encoding="utf-8") as f:
        conll_str = f.read()
        conll_lines = conll_str.split("\n")
        for i, conll_line in tqdm(enumerate(conll_lines), total=len(conll_lines)):
            if i+1 == len(conll_lines):
                break
            if "#begin document" in conll_line or "#end document" in conll_line:
                continue
            if conll_line.split("\t")[0].split(".")[1] == "DCT":
                continue

            if "t" in conll_line.split("\t")[0].split(".")[1]:
                sent_id = 0
            else:
                sent_id = int(conll_line.split("\t")[0].split(".")[1][1:])

            doc_id = conll_line.split("\t")[0].split(".")[0]
            if doc_id != prev_doc_id:
                doc_ids.append(doc_id)
                prev_doc_id = doc_id
                doc_token_counter = 0

            if "(" in conll_line.split("\t")[4]:
                # format: corefID_docID_sentID_token1ID
                mention_identifiers.append(conll_line.split("\t")[4].replace('(','').replace(')','')+"_"+conll_line.split("\t")[0].split(".")[0]+"_"+str(sent_id)+"_"+str(conll_line.split("\t")[0].split(".")[2]))

            conll_df = pd.concat([conll_df, pd.DataFrame({
                TOPIC_SUBTOPIC: conll_line.split("\t")[3].split(".")[0],
                DOC_ID: conll_line.split("\t")[0].split(".")[0],
                SENT_ID: sent_id,
                TOKEN_ID: int(conll_line.split("\t")[0].split(".")[2]),
                "doc_token_id": doc_token_counter,
                TOKEN: conll_line.split("\t")[1],
                "title_text_id": conll_line.split("\t")[2],
                REFERENCE: conll_line.split("\t")[4]
            }, index=[0])])

            doc_token_counter = int(doc_token_counter + 1)

    # count punctuations in a string (used to determine best tolerance for mention detection)
    count_punct = lambda l: sum([1 for x in l if x in string.punctuation])

    # --> reassemble the conll data into its original articles
    LOGGER.info("Reassembling the original articles from the conll data...")
    doc_files = {}
    for i, mention_identifier in tqdm(enumerate(mention_identifiers), total=len(mention_identifiers)):
        doc_id = mention_identifier.split("_")[1]
        doc_conll = conll_df[conll_df[DOC_ID] == doc_id]

        doc_str = ""
        for i, row in doc_conll[doc_conll["title_text_id"] == "BODY"].iterrows():
            doc_str, word_fixed, no_whitespace = append_text(doc_str, row[TOKEN])
        title_str = ""
        for i, row in doc_conll[doc_conll["title_text_id"] == "TITLE"].iterrows():
            title_str, word_fixed, no_whitespace = append_text(title_str, row[TOKEN])

        newsplease_custom = copy.copy(newsplease_format)

        newsplease_custom["filename"] = "verbose.conll"
        newsplease_custom["text"] = doc_str
        newsplease_custom["source_domain"] = doc_id
        newsplease_custom["language"] = "en"
        newsplease_custom["title"] = title_str
        if newsplease_custom["title"][-1] not in string.punctuation:
            newsplease_custom["title"] += "."

        doc_files[doc_id] = newsplease_custom

        with open(os.path.join(result_path, newsplease_custom["source_domain"] + ".json"),
                  "w") as file:
            json.dump(newsplease_custom, file)

    need_manual_review_mention_head = {}
    entity_mentions_local = []
    topic_id = 0

    # --> generate coref mention files

    # go through every mention identifier and determine its attributes to save into entity_mentions_local
    LOGGER.info("Generating the mentions file...")
    for i, mention_identifier in tqdm(enumerate(mention_identifiers), total=len(mention_identifiers)):
        coref_id = mention_identifier.split("_")[0]
        if mention_identifier.split("_")[1] != doc_id and i != 0:
            topic_id = topic_id + 1
        doc_id = mention_identifier.split("_")[1]
        sent_id = int(mention_identifier.split("_")[2])

        # get only the conll data that corresponds to the sentence we want to process
        sent_conll = conll_df[(conll_df[SENT_ID] == sent_id) & (conll_df[DOC_ID] == doc_id)]

        sentence_str = ""
        mention_tokenized = []
        split_mention_text = []
        mention_text = ""
        currently_adding = False
        whole_mention_added = False
        for i, row in sent_conll.iterrows():
            sentence_str, word_fixed, no_whitespace = append_text(sentence_str, row[TOKEN])
            if coref_id in row[REFERENCE] and not whole_mention_added:
                currently_adding = True
                mention_tokenized.append(row[TOKEN_ID])
                mention_text, word_fixed, no_whitespace = append_text(mention_text, row[TOKEN])
                split_mention_text.append(row[TOKEN])
            else:
                if currently_adding == True:
                    whole_mention_added = True

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

        token_str = mention_text
        t_subt = sent_conll[sent_conll[TOPIC_SUBTOPIC] != "-"].iloc[0][TOPIC_SUBTOPIC]
        conll_df.loc[(conll_df[DOC_ID] == doc_id), TOPIC_SUBTOPIC] = t_subt  # set subtopics in conll
        counter = 0
        while True:
            if counter > 50:  # an error must have occurred, so break and add to manual review
                need_manual_review_mention_head[str(t_subt) + "_" + str(mention_text)[:10]] = {
                    "mention_text": mention_text,
                    "sentence_str": sentence_str,
                    "mention_head": "unknown",
                    "mention_tokens_amount": len(mention_tokenized),
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
                    mention_tokenized)) <= tolerance and sentence_str[
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

        # remap the mention head back to the original tokenization to get the ID for the output
        mention_head_id = None
        mention_head_text = mention_head.text
        for i, t in enumerate(split_mention_text):
            if t.startswith(mention_head_text):
                mention_head_id = mention_tokenized[i]

        if not mention_head_id:
            for i, t in enumerate(split_mention_text):
                if mention_head_text.startswith(t):
                    mention_head_id = mention_tokenized[i]
        if not mention_head_id:
            for i, t in enumerate(split_mention_text):
                if t.endswith(mention_head_text):
                    mention_head_id = mention_tokenized[i]

        # add to manual review if the resulting token is not inside the mention
        # (error must have happened)
        if mention_head_id not in mention_tokenized:  # also "if is None"
            if str(t_subt) + "_" + str(mention_text) not in need_manual_review_mention_head:
                need_manual_review_mention_head[str(t_subt) + "_" + str(mention_text)[:10]] = \
                    {
                        "mention_text": mention_text,
                        "sentence_str": sentence_str,
                        "mention_head": str(mention_head),
                        "mention_tokens_amount": len(mention_tokenized),
                        "tolerance": tolerance
                    }
                with open(os.path.join(OUT_PATH, MANUAL_REVIEW_FILE),
                          "w",
                          encoding='utf-8') as file:
                    json.dump(need_manual_review_mention_head, file)
                LOGGER.info(
                    f"Mention with ID {str(t_subt)}_{str(mention_text)} needs manual review. Could not determine the mention head automatically. {str(tolerance)}")

        # get the context range
        context_min_id, context_max_id = [0 if int(min(mention_tokenized)) - CONTEXT_RANGE < 0 else
                                          int(min(mention_tokenized)) - CONTEXT_RANGE,
                                          int(max(mention_tokenized)) + CONTEXT_RANGE]

        # iterate over tokens and append if they are in context range
        mention_context_str = []
        for i, row in conll_df[conll_df[DOC_ID] == doc_id].iterrows():
            if row["doc_token_id"] > context_max_id:
                break
            elif context_min_id <= row["doc_token_id"] <= context_max_id:
                mention_context_str.append(row[TOKEN])

        # add attributes to mentions if the variables are correct ( do not add when manual review needed )
        if str(t_subt) + "_" + str(mention_text) not in need_manual_review_mention_head:
            mention = {COREF_CHAIN: coref_id,
                       MENTION_NER: mention_ner,
                       MENTION_HEAD_POS: mention_head_pos,
                       MENTION_HEAD_LEMMA: mention_head_lemma,
                       MENTION_HEAD: mention_head_text,
                       MENTION_HEAD_ID: mention_head_id,
                       DOC_ID: doc_id,
                       DOC_ID_FULL: doc_id,
                       IS_CONTINIOUS: mention_tokenized == list(range(mention_tokenized[0], mention_tokenized[-1] + 1)),
                       IS_SINGLETON: len(mention_tokenized) == 1,
                       MENTION_ID: mention_identifier,
                       MENTION_TYPE: "MIS",
                       MENTION_FULL_TYPE: "MISC",
                       SCORE: -1.0,
                       SENT_ID: sent_id,
                       MENTION_CONTEXT: mention_context_str,
                       TOKENS_NUMBER: mention_tokenized,
                       TOKENS_STR: token_str,
                       TOKENS_TEXT: split_mention_text,
                       TOPIC_ID: 0,    # topic_id if subtopic desired
                       TOPIC: "-",
                       SUBTOPIC: t_subt,
                       TOPIC_SUBTOPIC: t_subt,
                       COREF_TYPE: STRICT,
                       DESCRIPTION: None,
                       CONLL_DOC_KEY: "-/"+t_subt+"/"+doc_id
                       }

            # add to mentions list
            entity_mentions_local.append(mention)

            summary_df.loc[len(summary_df)] = {
                DOC_ID: doc_id,
                COREF_CHAIN: coref_id,
                DESCRIPTION: "",
                MENTION_TYPE: "MIS",
                MENTION_FULL_TYPE: "MISC",
                MENTION_ID: mention_identifier,
                TOKENS_STR: token_str
            }

    # --> output files

    if len(need_manual_review_mention_head):
        LOGGER.warning(
            f'Mentions ignored: {len(need_manual_review_mention_head)}. The ignored mentions are available here for a manual review: '
            f'{os.path.join(OUT_PATH, MANUAL_REVIEW_FILE)}')
        with open(os.path.join(OUT_PATH, MANUAL_REVIEW_FILE), "w", encoding='utf-8') as file:
            json.dump(need_manual_review_mention_head, file)

    with open(os.path.join(OUT_PATH, MENTIONS_ENTITIES_JSON), "w", encoding='utf-8') as file:
        json.dump(entity_mentions_local, file)

    summary_df.drop(columns=[MENTION_ID], inplace=True)
    summary_df.to_csv(os.path.join(OUT_PATH, MENTIONS_ALL_CSV))

    # --> create an output conll string from the conll_df in our format

    conll_df = conll_df.reset_index(drop=True)
    final_output_str = ""
    LOGGER.info("Generating conll string...")
    for i, row in tqdm(conll_df.iterrows(), total=conll_df.shape[0]):
        conll_df.iloc[i][TOPIC_SUBTOPIC] = "-/" + row[TOPIC_SUBTOPIC] + "/" + row[DOC_ID]
        if not "(" in row[REFERENCE] and not ")" in row[REFERENCE]:
            conll_df.iloc[i][REFERENCE] = "-"

    conll_df = conll_df.drop(columns=[DOC_ID, "doc_token_id", "title_text_id"])
    outputdoc_str = ""
    for (topic_local), topic_df in conll_df.groupby(by=[TOPIC_SUBTOPIC]):
        outputdoc_str += f'#begin document ({topic_local}); part 000\n'

        for (sent_id_local), sent_df in topic_df.groupby(by=[SENT_ID], sort=[SENT_ID]):
            np.savetxt(os.path.join(GVC_PARSING_FOLDER, "tmp.txt"), sent_df.values, fmt='%s', delimiter="\t",
                       encoding="utf-8")
            with open(os.path.join(GVC_PARSING_FOLDER, "tmp.txt"), "r", encoding="utf8") as file:
                saved_lines = file.read()
            outputdoc_str += saved_lines + "\n"

        outputdoc_str += "#end document\n"

    # --> Check if the brackets ( ) are correct
    try:
        brackets_1 = 0
        brackets_2 = 0
        for i, row in conll_df.iterrows():  # only count brackets in reference column (exclude token text)
            brackets_1 += str(row[REFERENCE]).count("(")
            brackets_2 += str(row[REFERENCE]).count(")")
        LOGGER.info(
            f"Amount of mentions in this topic: {str(len(entity_mentions_local))}")
        LOGGER.info(f"brackets '(' , ')' : {str(brackets_1)}, {str(brackets_2)}")
        assert brackets_1 == brackets_2
    except AssertionError:
        print(outputdoc_str)
        LOGGER.warning(f'Number of opening and closing brackets in conll does not match!')
        conll_df.to_csv(os.path.join(OUT_PATH, CONLL_CSV))
        with open(os.path.join(OUT_PATH, 'gvc.conll'), "w", encoding='utf-8') as file:
            file.write(outputdoc_str)
        # sys.exit()

    # --> output conll in out format

    conll_df.to_csv(os.path.join(OUT_PATH, CONLL_CSV))
    with open(os.path.join(OUT_PATH, 'gvc.conll'), "w", encoding='utf-8') as file:
        file.write(outputdoc_str)

if __name__ == '__main__':
    LOGGER.info(f"Processing GVC {source_path[-34:].split('_')[2]}.")
    conv_files(source_path)
