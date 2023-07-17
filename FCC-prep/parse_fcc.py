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
pd.set_option('display.max_columns', None)

path_sample = os.path.join(DATA_PATH, "_sample_doc.json")  # ->root/data/original/_sample_doc.json
FCC_PARSING_FOLDER = os.path.join(os.getcwd())
OUT_PATH = os.path.join(FCC_PARSING_FOLDER, OUTPUT_FOLDER_NAME)
source_path = os.path.join(FCC_PARSING_FOLDER, FCC_FOLDER_NAME)
result_path = os.path.join(OUT_PATH, 'test_parsing')

CONTEXT_COMB_SPAN = 5

nlp = spacy.load('en_core_web_sm')
LOGGER.info("Spacy model loaded.")

# opens and loads the newsplease-format out of the json file: _sample_doc.json
with open(path_sample, "r") as file:
    newsplease_format = json.load(file)


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
    Converts the given dataset into the desired format.
    :param path: The path desired to process
    """
    conll_df = pd.DataFrame(
        columns=[TOPIC_SUBTOPIC, DOC_ID, SENT_ID, TOKEN_ID, "doc_token_id", TOKEN, "title_text_id", REFERENCE])
    summary_df = pd.DataFrame(
        columns=[DOC_ID, COREF_CHAIN, DESCRIPTION, MENTION_TYPE, MENTION_FULL_TYPE, MENTION_ID, TOKENS_STR])

    # --> process the datasets tokens file into a pd dataframe for processing
    LOGGER.info("Reading tokens.csv...")
    for split in ["dev", "test", "train"]:
        LOGGER.info("Reading " + split + " split...")
        tokens_df = pd.read_csv(os.path.join(source_path, "2020-10-05_FCC_cleaned", split, "tokens.csv"))
        for i, row in tqdm(tokens_df.iterrows(), total=tokens_df.shape[0]):
            conll_df = pd.concat([conll_df, pd.DataFrame({
                TOPIC_SUBTOPIC: "-",
                DOC_ID: row["doc-id"],
                SENT_ID: int(row["sentence-idx"]),
                TOKEN_ID: int(row["token-idx"]),
                "doc_token_id": int(row["token-idx"]),
                TOKEN: row["token"],
                "title_text_id": row["sentence-type"].upper(),
                REFERENCE: "-"
            }, index=[0])])

            #if i > 1000:
            #    break
    conll_df.reset_index(drop=True, inplace=True)

    LOGGER.info("Reading mentions_cross_subtopic.csv...")
    mention_identifiers = []
    for split in ["dev", "test", "train"]:
        LOGGER.info("Reading " + split + " split...")
        mentions_cross_subt_df = pd.read_csv(
            os.path.join(source_path, "2020-10-05_FCC_cleaned", split, "mentions_cross_subtopic.csv"))
        for i, row in tqdm(mentions_cross_subt_df.iterrows(), total=mentions_cross_subt_df.shape[0]):
            # format: corefID_docID_sentID_mentionID
            mention_identifiers.append(
                str(row["event"]).replace("_", "") + "_" + str(row["doc-id"]).replace("_", "") + "_" + str(
                    row["sentence-idx"]).replace("_", "") + "_" + str(row["mention-id"]))

    LOGGER.info("Reading documents data (including seminal events / subtopics)...")
    docs_seminal_df = pd.DataFrame()
    for split in ["dev", "test", "train"]:
        LOGGER.info("Reading " + split + " split...")
        docs_seminal_df = pd.concat([docs_seminal_df, pd.read_csv(os.path.join(source_path, "2020-10-05_FCC_cleaned",
                                                                               split, "documents.csv"))])
    LOGGER.info("Found " + str(docs_seminal_df["seminal-event"].nunique()) + " different seminal events.")

    LOGGER.info("Updating conll dataframe with the correct subtopics...")
    for i, row in tqdm(conll_df.iterrows(), total=conll_df.shape[0]):
        seminal_event = docs_seminal_df[docs_seminal_df["doc-id"] == row[DOC_ID]].iloc[0]["seminal-event"]
        conll_df.at[i, TOPIC_SUBTOPIC] = "0/"+str(seminal_event)+"/"+str(row[DOC_ID])   # main topic 0 "football"

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

        newsplease_custom["filename"] = "tokens.csv"
        newsplease_custom["text"] = doc_str
        newsplease_custom["source_domain"] = doc_id
        newsplease_custom["language"] = "en"
        newsplease_custom["title"] = title_str
        if newsplease_custom["title"] != "":
            if newsplease_custom["title"][-1] not in string.punctuation:
                newsplease_custom["title"] += "."

        doc_files[doc_id] = newsplease_custom

        seminal_event = str(docs_seminal_df[docs_seminal_df["doc-id"] == doc_id].iloc[0]["seminal-event"])   # subtopic
        seminal_event = seminal_event.replace(":", "_")    # needed to be a valid file path
        os.makedirs(os.path.join(result_path, seminal_event), exist_ok=True)

        with open(os.path.join(result_path, seminal_event, newsplease_custom["source_domain"] + ".json"),
                  "w") as file:
            json.dump(newsplease_custom, file)

    event_mentions = []

    # --> generate coref mention files
    # go through every mention identifier and determine its attributes to save into event_mentions
    LOGGER.info("Generating the mentions file...")
    for i, mention_identifier in tqdm(enumerate(mention_identifiers), total=len(mention_identifiers)):
        coref_id = mention_identifier.split("_")[0]
        doc_id = mention_identifier.split("_")[1]
        sent_id = int(mention_identifier.split("_")[2])

        # get only the conll data that corresponds to the sentence we want to process
        sent_conll = conll_df[(conll_df[SENT_ID] == sent_id) & (conll_df[DOC_ID] == doc_id)]
        mention_tokenized = sent_conll[TOKEN_ID].tolist()
        split_mention_text = sent_conll[TOKEN].tolist()
        if sent_conll.shape[0] == 0:
            LOGGER.info("Skipping mention: " + str(mention_identifier) + ", ignore this if this is a limited test run.")
            continue

        sentence_str = ""
        for i, row in sent_conll.iterrows():
            sentence_str, word_fixed, no_whitespace = append_text(sentence_str, row[TOKEN])
        mention_text = sentence_str

        doc = nlp(sentence_str)
        token_str = mention_text

        if conll_df[conll_df[DOC_ID] == doc_id].shape[0] == 0:
            LOGGER.info("Skipping mention: " + str(mention_identifier) + ", ignore this if this is a limited test run.")
            continue
        t_subt = conll_df[conll_df[DOC_ID] == doc_id].iloc[0][TOPIC_SUBTOPIC]

        # whole mention string processed, look for the head
        for t in doc:
            ancestors_in_mention = len(list(t.ancestors))
            if ancestors_in_mention == 0 and t.text not in string.punctuation:  # puncts should not be heads
                # head within the mention found
                mention_head = t
                break

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

        #[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]

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

        subtopic_name = t_subt.split("/")[1]

        # add attributes to mentions
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
                   SUBTOPIC: subtopic_name,
                   TOPIC_SUBTOPIC: subtopic_name,
                   COREF_TYPE: STRICT,
                   DESCRIPTION: None,
                   CONLL_DOC_KEY: t_subt
                   }

        # add to mentions list
        event_mentions.append(mention)

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
    with open(os.path.join(OUT_PATH, MENTIONS_EVENTS_JSON), "w", encoding='utf-8') as file:
        json.dump(event_mentions, file)

    summary_df.drop(columns=[MENTION_ID], inplace=True)
    summary_df.to_csv(os.path.join(OUT_PATH, MENTIONS_ALL_CSV))

    # create a conll string from the conll_df
    LOGGER.info("Generating conll string...")
    for i, row in tqdm(conll_df.iterrows(), total=conll_df.shape[0]):
        if row[REFERENCE] is None:
            reference_str = "-"
        else:
            reference_str = row[REFERENCE]

        for mention in event_mentions:
            if mention[CONLL_DOC_KEY] == row[TOPIC_SUBTOPIC] and mention[SENT_ID] == row[SENT_ID] and row[TOKEN_ID] in mention[TOKENS_NUMBER]:
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

        conll_df.at[i, REFERENCE] = reference_str

    for i, row in conll_df.iterrows():  # remove the leading characters if necessary (left from initialization)
        if row[REFERENCE].startswith("-| "):
            conll_df.at[i, REFERENCE] = row[REFERENCE][3:]

    conll_df = conll_df.drop(columns=[DOC_ID, "title_text_id", "doc_token_id"])

    outputdoc_str = ""
    for (topic_local), topic_df in conll_df.groupby(by=[TOPIC_SUBTOPIC]):
        outputdoc_str += f'#begin document ({topic_local}); part 000\n'

        for (sent_id_local), sent_df in topic_df.groupby(by=[SENT_ID], sort=[SENT_ID]):
            np.savetxt(os.path.join(FCC_PARSING_FOLDER, "tmp.txt"), sent_df.values, fmt='%s', delimiter="\t",
                       encoding="utf-8")
            with open(os.path.join(FCC_PARSING_FOLDER, "tmp.txt"), "r", encoding="utf8") as file:
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
            f"Amount of mentions in this topic: {str(len(event_mentions))}")
        LOGGER.info(f"brackets '(' , ')' : {str(brackets_1)}, {str(brackets_2)}")
        assert brackets_1 == brackets_2
    except AssertionError:
        print(outputdoc_str)
        LOGGER.warning(f'Number of opening and closing brackets in conll does not match!')
        conll_df.to_csv(os.path.join(OUT_PATH, CONLL_CSV))
        with open(os.path.join(OUT_PATH, 'fcc.conll'), "w", encoding='utf-8') as file:
            file.write(outputdoc_str)
        # sys.exit()

    # --> output conll in out format

    conll_df.to_csv(os.path.join(OUT_PATH, CONLL_CSV))
    with open(os.path.join(OUT_PATH, 'fcc.conll'), "w", encoding='utf-8') as file:
        file.write(outputdoc_str)

    LOGGER.info(f'Parsing of FCC done!')

if __name__ == '__main__':
    LOGGER.info(f"Processing FCC {source_path[-34:].split('_')[2]}.")
    conv_files(source_path)
