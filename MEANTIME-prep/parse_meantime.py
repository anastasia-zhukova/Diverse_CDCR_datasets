import xml.etree.ElementTree as ET
import string
import copy
import re
import pandas as pd
import numpy as np
from nltk import Tree
import shortuuid
from tqdm import tqdm
from setup import *
from utils import *
from logger import LOGGER

path_sample = os.path.join(DATA_PATH, "_sample_doc.json") # ->root/data/original/_sample_doc.json
MEANTIME_PARSING_FOLDER = os.path.join(os.getcwd())
OUT_PATH = os.path.join(MEANTIME_PARSING_FOLDER, OUTPUT_FOLDER_NAME)
CONTEXT_RANGE = 250

# opens and loads the newsplease-format out of the json file: _sample_doc.json
with open(path_sample, "r") as file:
    newsplease_format = json.load(file)


lang_paths = {
    EN: {
        "source": os.path.join(MEANTIME_PARSING_FOLDER, MEANTIME_FOLDER_NAME, MEANTIME_FOLDER_NAME_ENGLISH),
        "nlp": spacy.load(SPACY_EN)},
    ES: {
        "source": os.path.join(MEANTIME_PARSING_FOLDER, MEANTIME_FOLDER_NAME,  MEANTIME_FOLDER_NAME_SPANISH),
        "nlp": spacy.load(SPACY_ES)},
    NL: {
        "source": os.path.join(MEANTIME_PARSING_FOLDER, MEANTIME_FOLDER_NAME,  MEANTIME_FOLDER_NAME_DUTCH),
        "nlp": spacy.load(SPACY_NL)},
    IT: {
        "source": os.path.join(MEANTIME_PARSING_FOLDER, MEANTIME_FOLDER_NAME, MEANTIME_FOLDER_NAME_ITALIAN),
        "nlp": spacy.load(SPACY_IT)}
}

meantime_types = {"PRO": "PRODUCT",
                  "FIN": "FINANCE",
                  "LOC": "LOCATION",
                  "ORG": "ORGANIZATION",
                  "OTH": "OTHER",
                  "PER": "PERSON",
                  "GRA": "GRAMMATICAL",
                  "SPE": "SPEECH_COGNITIVE",
                  "MIX": "MIXTURE"}


def to_nltk_tree(node):
    """
        Converts a sentence to a visually helpful tree-structure output.
        Can be used to double-check if a determined head is correct.
    """
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


# def conv_files(paths, result_path, out_path, language, nlp):
def conv_files():
    """
        Converts the given dataset for a specified language into the desired format.
        :param paths: The paths desired to process (intra- & cross_intra annotations)
        :param result_path: the test_parsing folder for that language
        :param out_path: the output folder for that language
        :param language: the language to process
        :param nlp: the spacy model that fits the desired language
    """
    out_path = os.path.join(OUT_PATH, "test_parsing")
    if "test_parsing" not in os.listdir(OUT_PATH):
        os.mkdir(out_path)

    doc_files = {}
    entity_mentions = []
    event_mentions = []
    topic_list = set()
    conll_df = pd.DataFrame(columns=[TOPIC_SUBTOPIC_DOC, DOC_ID, SENT_ID, TOKEN_ID, TOKEN, REFERENCE])
    need_manual_review_mention_head = {}
    coref_dict = {}

    for lang_key in lang_paths.keys():
        source_path = lang_paths[lang_key]["source"]
        nlp = lang_paths[lang_key]["nlp"]
        language = source_path[-34:].split("_")[2]

        LOGGER.info(f"Processing MEANTIME language {language}.")
        annotation_folders = [
            os.path.join(source_path, 'intra_cross-doc_annotation'),
            # os.path.join(source_path, 'intra-doc_annotation')
        ]

        for path in annotation_folders:
            dirs = os.listdir(path)

            for topic_id, topic in enumerate(dirs):  # for each topic folder
                LOGGER.info(f"Parsing of {topic} ({language}) [{path[-20:]}]. Please wait...")
                topic_id_compose = f'{topic_id}_{topic}'
                topic_files = os.listdir(os.path.join(path, topic))
                topic_list.add(topic_id_compose)

                result_topic_path = os.path.join(out_path, topic_id_compose)
                if topic_id_compose not in os.listdir(out_path):
                    os.mkdir(result_topic_path)

                for doc_file in tqdm(topic_files):
                    tree = ET.parse(os.path.join(path, topic, doc_file))
                    root = tree.getroot()
                    subtopic_full = doc_file.split(".")[0]
                    subtopic = subtopic_full.split("_")[0]
                    doc_id_full = f'{language}{doc_file.split(".")[0]}'
                    doc_id = f'{language}{subtopic}'
                    topic_subtopic_doc = f'{topic_id}/{subtopic}/{doc_id}'

                    result_subtopic_path = os.path.join(result_topic_path, subtopic)
                    if subtopic not in os.listdir(result_topic_path):
                        os.mkdir(result_subtopic_path)

                    token_dict, mentions, mentions_map = {}, {}, {}

                    t_id = -1
                    old_sent = 0

                    for elem in root:
                        if elem.tag == "token":
                            try:
                                if old_sent == int(elem.attrib["sentence"]):
                                    t_id += 1
                                else:
                                    old_sent = int(elem.attrib["sentence"])
                                    t_id = 0
                                token_dict[elem.attrib["t_id"]] = {"text": elem.text, "sent": elem.attrib["sentence"],
                                                                   "id": t_id}

                                if elem.tag == "token" and len(conll_df.loc[(conll_df[TOPIC_SUBTOPIC_DOC] == topic_subtopic_doc) &
                                                                            (conll_df[DOC_ID] == doc_id) &
                                                                            (conll_df[SENT_ID] == int(
                                                                                elem.attrib["sentence"])) &
                                                                            (conll_df[TOKEN_ID] == t_id)]) < 1:
                                    conll_df.loc[len(conll_df)] = {
                                        TOPIC_SUBTOPIC_DOC: topic_subtopic_doc,
                                        DOC_ID: doc_id,
                                        SENT_ID: int(elem.attrib["sentence"]),
                                        TOKEN_ID: t_id,
                                        TOKEN: elem.text,
                                        REFERENCE: "-"
                                    }

                            except KeyError as e:
                                LOGGER.warning(f'Value with key {e} not found and will be skipped from parsing.')

                        if elem.tag == "Markables":
                            for i, subelem in enumerate(elem):
                                if "SIGNAL" in subelem.tag:
                                    continue

                                tokens = [token.attrib[T_ID] for token in subelem]
                                tokens.sort(key=int)  # sort tokens by their id
                                sent_tokens = [int(token_dict[t]["id"]) for t in tokens]

                                # skip if the token is contained more than once within the same mention
                                # (i.e. ignore entries with error in meantime tokenization)
                                if len(tokens) != len(list(set(tokens))):
                                    continue

                                mention_text = ""
                                for t in tokens:
                                    mention_text, _, _ = append_text(mention_text, token_dict[t][TEXT])

                                # if "tokens" has values -> fill the "mention" dict with the value of the corresponding m_id
                                if len(tokens):
                                    sent_id = int(token_dict[tokens[0]][SENT])

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
                                    tolerance = 1
                                    mention_doc_ids = []
                                    mention_id = topic_subtopic_doc + "-" + str(sent_id) + "-" + mention_text
                                    while True:
                                        if counter > 50:  # an error must have occurred, so break and add to manual review
                                            need_manual_review_mention_head[mention_id] = {
                                                "mention_text": mention_text,
                                                "sentence_str": sentence_str,
                                                "mention_head": "unknown",
                                                "mention_tokens_amount": len(tokens),
                                                "tolerance": tolerance
                                            }
                                            LOGGER.warning(
                                                f"Document {doc_id}: Mention with ID {str(topic_subtopic_doc)}_{str(mention_text)} needs manual review. Could not determine the mention head automatically. {str(tolerance)}")
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
                                    # mention_head = None
                                    if mention_id not in need_manual_review_mention_head:
                                        mention_head = doc[0]
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
                                        if mention_id not in need_manual_review_mention_head:
                                            need_manual_review_mention_head[mention_id] = \
                                                {
                                                    "mention_text": mention_text,
                                                    "sentence_str": sentence_str,
                                                    "mention_head": str(mention_head),
                                                    "mention_tokens_amount": len(tokens),
                                                    "tolerance": tolerance
                                                }
                                            with open(os.path.join(TMP_PATH, MANUAL_REVIEW_FILE.replace(".json",
                                                                                                        "_" + language + ".json")),
                                                      "w",
                                                      encoding='utf-8') as file:
                                                json.dump(need_manual_review_mention_head, file)
                                            LOGGER.warning(
                                                f"Document {doc_id}: Mention with ID {mention_id} needs manual review. Could not determine the mention head automatically. {str(tolerance)}")

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

                                    # add to mentions if the variables are correct ( do not add for manual review needed )
                                    if mention_id not in need_manual_review_mention_head:
                                        mentions[subelem.attrib["m_id"]] = {"type": subelem.tag,
                                                                            "text": " ".join(
                                                                                [token_dict[t]["text"] for t in tokens]),
                                                                            "sent_doc": doc,
                                                                            "source": path.split("\\")[-1].split("-")[0],
                                                                            LANGUAGE: language,
                                                                            MENTION_ID: mention_id,
                                                                            MENTION_NER: mention_ner,
                                                                            MENTION_HEAD_POS: mention_head_pos,
                                                                            MENTION_HEAD_LEMMA: mention_head_lemma,
                                                                            MENTION_HEAD: mention_head.text,
                                                                            MENTION_HEAD_ID: mention_head.i,
                                                                            TOKENS_NUMBER: sent_tokens,
                                                                            TOKENS_TEXT: [str(token_dict[t]["text"]) for t in tokens],
                                                                            DOC_ID: doc_id,
                                                                            DOC: doc_id_full,
                                                                            SENT_ID: int(sent_id),
                                                                            MENTION_CONTEXT: mention_context_str,
                                                                            SUBTOPIC: subtopic_full,
                                                                            TOPIC_SUBTOPIC_DOC: topic_subtopic_doc,
                                                                            TOPIC: topic_id_compose}

                                else:
                                    # form coreference chain
                                    # m_id points to the target

                                    if "ent_type" in subelem.attrib:
                                        mention_type_annot = meantime_types.get(subelem.attrib["ent_type"], "")
                                    elif "class" in subelem.attrib:
                                        mention_type_annot = subelem.attrib["class"]
                                    elif "type" in subelem.attrib:
                                        mention_type_annot = subelem.attrib["type"]
                                    else:
                                        mention_type_annot = ""

                                    if "instance_id" in subelem.attrib:
                                        id_ = subelem.attrib["instance_id"]
                                    else:
                                        descr = subelem.attrib["TAG_DESCRIPTOR"]
                                        id_ = ""

                                        for coref_id, coref_vals in coref_dict.items():
                                            if coref_vals["descr"] == descr and coref_vals["type"] == mention_type_annot \
                                                    and coref_vals["subtopic"] == subtopic and mention_type_annot:
                                                id_ = coref_id
                                                break

                                        if not len(id_):
                                            LOGGER.warning(f"Document {doc_id}: {subelem.attrib} doesn\'t have attribute instance_id. It will be created")
                                            if "ent_type" in subelem.attrib:
                                                id_ = subelem.attrib["ent_type"] + shortuuid.uuid()[:17]
                                            elif "class" in subelem.attrib:
                                                id_ = subelem.attrib["class"][:3] + shortuuid.uuid()[:17]
                                            elif "type" in subelem.attrib:
                                                id_ = subelem.attrib["type"][:3] + shortuuid.uuid()[:17]
                                            else:
                                                id_ = ""

                                        if not len(id_):
                                            continue

                                        subelem.attrib["instance_id"] = id_

                                    if not len(id_):
                                        continue

                                    if id_ not in coref_dict:
                                        coref_dict[id_] = {"descr": subelem.attrib["TAG_DESCRIPTOR"],
                                                           "type": mention_type_annot,
                                                           "subtopic": subtopic}

                        if elem.tag == "Relations":
                            mentions_map = {m: False for m in list(mentions)}
                            # use only REFERS_TO
                            for i, subelem in enumerate(elem):
                                if subelem.tag != "REFERS_TO":
                                    continue

                                tmp_instance_id = "None"
                                for j, subsubelm in enumerate(subelem):
                                    if subsubelm.tag == "target":
                                        for prevelem in root:
                                            if prevelem.tag != "Markables":
                                                continue

                                            for k, prevsubelem in enumerate(prevelem):
                                                if prevsubelem.get("instance_id") is None:
                                                    continue

                                                if subsubelm.attrib["m_id"] == prevsubelem.attrib["m_id"]:
                                                    tmp_instance_id = prevsubelem.attrib["instance_id"]
                                                    break

                                if tmp_instance_id == "None":
                                    LOGGER.warning(f"Document {doc_id}: not found target for: {str([v.attrib for v in subelem])}")
                                    continue
                                try:
                                    if "r_id" not in coref_dict[tmp_instance_id]:
                                        coref_dict[tmp_instance_id].update({
                                            "r_id": subelem.attrib["r_id"],
                                            "coref_type": subelem.tag,
                                            "mentions": {mentions[m.attrib["m_id"]][MENTION_ID]: mentions[m.attrib["m_id"]] for m in subelem if
                                                         m.tag == "source"}
                                        })
                                    else:
                                        for m in subelem:
                                            if m.tag == "source":
                                                mention_id = mentions[m.attrib["m_id"]][MENTION_ID]
                                                if mentions[m.attrib["m_id"]][MENTION_ID] in coref_dict[tmp_instance_id]["mentions"]:
                                                    continue

                                                coref_dict[tmp_instance_id]["mentions"][mention_id] = mentions[m.attrib["m_id"]]
                                except KeyError as e:
                                    LOGGER.warning(f'Document {doc_id}: Mention with ID {str(e)} is not amoung the Markables and will be skipped.')
                                for m in subelem:
                                    mentions_map[m.attrib["m_id"]] = True


        LOGGER.info(f'Parsing of MEANTIME annotation with language {language} done!')

    for chain_index, (chain_id, chain_vals) in enumerate(coref_dict.items()):
        if chain_vals.get("mentions") is None or chain_id == "":
            LOGGER.warning(f"Chain {chain_id}, {chain_vals} doesn\'t have any mentions and will be excluded.")
            continue

        for m_d, m in chain_vals["mentions"].items():
            token_numbers = [int(t) for t in m[TOKENS_NUMBER]]
            mention_id = f'{m["doc_id"]}_{str(chain_id)}_{str(m["sent_id"])}_{str(m[MENTION_HEAD_ID])}_{m["source"]}'

            try:
                mention_type = meantime_types[chain_id[:3]]
            except KeyError:
                mention_type = "OTHER"

            mention = {COREF_CHAIN: chain_id,
                       MENTION_NER: m["mention_ner"],
                       MENTION_HEAD_POS: m["mention_head_pos"],
                       MENTION_HEAD_LEMMA: m["mention_head_lemma"],
                       MENTION_HEAD: m["mention_head"],
                       MENTION_HEAD_ID: m["mention_head_id"],
                       DOC_ID: m[DOC_ID],
                       DOC: m[DOC],
                       IS_CONTINIOUS: True if token_numbers == list(
                           range(token_numbers[0], token_numbers[-1] + 1))
                       else False,
                       IS_SINGLETON: len(chain_vals["mentions"]) == 1,
                       MENTION_ID: mention_id,
                       MENTION_TYPE: chain_id[:3],
                       MENTION_FULL_TYPE: mention_type,
                       SCORE: -1.0,
                       SENT_ID: m["sent_id"],
                       MENTION_CONTEXT: m[MENTION_CONTEXT],
                       TOKENS_NUMBER: token_numbers,
                       TOKENS_STR: m["text"],
                       TOKENS_TEXT: m[TOKENS_TEXT],
                       TOPIC_ID: m[TOPIC].split("_")[0],
                       TOPIC: m[TOPIC],
                       SUBTOPIC_ID: m[TOPIC_SUBTOPIC_DOC].split("/")[1],
                       SUBTOPIC: m[SUBTOPIC],
                       COREF_TYPE: IDENTITY,
                       DESCRIPTION: chain_vals["descr"],
                       LANGUAGE: m[LANGUAGE],
                       CONLL_DOC_KEY: m[TOPIC_SUBTOPIC_DOC],
                       }
            if "EVENT" in m["type"]:
                event_mentions.append(mention)
            else:
                entity_mentions.append(mention)

    # create a csv. file out of the mentions summary_df
    df_all_mentions = pd.DataFrame()
    for mention in tqdm(entity_mentions + event_mentions):
        df_all_mentions = pd.concat([df_all_mentions, pd.DataFrame({
            attr: str(value) if type(value) == list else value for attr, value in mention.items()
        }, index=[mention[MENTION_ID]])], axis=0)
    df_all_mentions.to_csv(os.path.join(OUT_PATH, MENTIONS_ALL_CSV))

    with open(os.path.join(OUT_PATH, MENTIONS_ENTITIES_JSON), "w",
              encoding='utf-8') as file:
        json.dump(entity_mentions, file)

    with open(os.path.join(OUT_PATH, MENTIONS_EVENTS_JSON), "w",
              encoding='utf-8') as file:
        json.dump(event_mentions, file)

    # save full corpus
    # make_save_conll(conll_df, entity_mentions, event_mentions, out_path)
    make_save_conll(conll_df, df_all_mentions, OUT_PATH)


    LOGGER.info(
        "Mentions that need manual review to define the head and its attributes have been saved to: " +
        MANUAL_REVIEW_FILE.replace(".json",
                                    ".json - Total: " + str(len(need_manual_review_mention_head))))

    with open(os.path.join(TMP_PATH, MANUAL_REVIEW_FILE.replace(".json",  ".json")), "w",
              encoding='utf-8') as file:
        json.dump(need_manual_review_mention_head, file)

    LOGGER.info(f'Done! \nNumber of unique mentions: {len(df_all_mentions)} '
                f'\nNumber of unique chains: {len(set(df_all_mentions[COREF_CHAIN].values))} ')

    LOGGER.info(f'Parsing of MEANTIME annotation done!')


if __name__ == '__main__':
    conv_files()
