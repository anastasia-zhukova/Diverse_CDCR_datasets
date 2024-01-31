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

MEANTIME_PARSING_FOLDER = os.path.join(os.getcwd())
OUT_PATH = os.path.join(MEANTIME_PARSING_FOLDER, OUTPUT_FOLDER_NAME)


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


def conv_files(languages_to_parse: List[str]=[EN, ES, NL, IT]):
    doc_files = {}
    entity_mentions = []
    event_mentions = []
    topic_list = set()
    conll_df = pd.DataFrame(columns=[TOPIC_SUBTOPIC_DOC, DOC_ID, SENT_ID, TOKEN_ID, TOKEN, REFERENCE])
    need_manual_review_mention_head = {}
    coref_dict = {}

    for lang_key in lang_paths.keys():
        if lang_key not in languages_to_parse:
            continue

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

                for doc_file in tqdm(topic_files):
                    tree = ET.parse(os.path.join(path, topic, doc_file))
                    root = tree.getroot()
                    subtopic_full = doc_file.split(".")[0]
                    subtopic = subtopic_full.split("_")[0]
                    doc_id_full = f'{language}{doc_file.split(".")[0]}'
                    doc_id = f'{language}{subtopic}'
                    topic_subtopic_doc = f'{topic_id}/{subtopic}/{doc_id}'

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

                                mention_tokens_ids_global = [token.attrib[T_ID] for token in subelem]
                                mention_tokens_ids_global.sort(key=int)  # sort tokens by their id
                                sent_tokens = [int(token_dict[t]["id"]) for t in mention_tokens_ids_global]

                                # skip if the token is contained more than once within the same mention
                                # (i.e. ignore entries with error in meantime tokenization)
                                if len(mention_tokens_ids_global) != len(list(set(mention_tokens_ids_global))):
                                    continue

                                tokens_str = ""
                                for t in mention_tokens_ids_global:
                                    tokens_str, _, _ = append_text(tokens_str, token_dict[t][TEXT])
                                doc_df = conll_df[(conll_df[TOPIC_SUBTOPIC_DOC] == topic_subtopic_doc)]

                                # if "tokens" has values -> fill the "mention" dict with the value of the corresponding m_id
                                if len(mention_tokens_ids_global):
                                    sent_id = int(token_dict[mention_tokens_ids_global[0]][SENT])

                                    # generate sentence doc with spacy
                                    sentence_str = ""
                                    for t in root:
                                        if t.tag == TOKEN and t.attrib[SENTENCE] == str(sent_id):
                                            sentence_str, _, _ = append_text(sentence_str, t.text)

                                    doc = nlp(sentence_str)
                                    mention_id = f'{topic_subtopic_doc}-{sent_id}-{tokens_str.replace(" ", "_")}'

                                    # tokenize the mention text
                                    tokens_text, token_ids = [], []
                                    for t_id in mention_tokens_ids_global:
                                        tokens_text.append(token_dict[t_id][TEXT])
                                        token_ids.append(int(token_dict[t_id][ID]))

                                    tolerance = 0
                                    token_found = {t: None for t in token_ids}
                                    prev_id = 0
                                    for t_id, t in zip(token_ids, tokens_text):
                                        to_break = False
                                        for tolerance in range(10):
                                            for token in doc[max(0, int(t_id) - tolerance): int(t_id) + tolerance + 1]:
                                                if token.i < prev_id - 1:
                                                    continue
                                                if token.text == t:
                                                    token_found[t_id] = token.i
                                                    prev_id = token.i
                                                    to_break = True
                                                    break
                                                elif t.startswith(token.text):
                                                    token_found[t_id] = token.i
                                                    prev_id = token.i
                                                    to_break = True
                                                    break
                                                elif token.text.startswith(t):
                                                    token_found[t_id] = token.i
                                                    prev_id = token.i
                                                    to_break = True
                                                    break
                                                elif t.endswith(token.text):
                                                    token_found[t_id] = token.i
                                                    prev_id = token.i
                                                    to_break = True
                                                    break
                                                elif token.text.endswith(t):
                                                    token_found[t_id] = token.i
                                                    prev_id = token.i
                                                    to_break = True
                                                    break
                                            if to_break:
                                                break

                                    # whole mention string processed, look for the head
                                    mention_head_id = token_ids[0]
                                    mention_head = None
                                    if mention_id not in need_manual_review_mention_head:
                                        found_mentions_tokens_ids = list(
                                            [t for t in token_found.values() if t is not None])
                                        found_mentions_tokens = []
                                        if len(found_mentions_tokens_ids):
                                            found_mentions_tokens = doc[min(found_mentions_tokens_ids): max(
                                                found_mentions_tokens_ids) + 1]
                                            if len(found_mentions_tokens) == 1:
                                                mention_head = found_mentions_tokens[0]
                                                # remap the mention head back to the np4e original tokenization to get the ID for the output
                                                for t_orig, t_mapped in token_found.items():
                                                    if t_mapped == mention_head.i:
                                                        mention_head_id = t_orig
                                                        break

                                        if mention_head is None:
                                            # found_mentions_tokens_ids = set([t.i for t in found_mentions_tokens])
                                            for i, t in enumerate(found_mentions_tokens):
                                                if t.head.i == t.i:
                                                    # if a token is a root, it is a candidate for the head
                                                    pass

                                                elif t.head.i >= min(found_mentions_tokens_ids) and t.head.i <= max(
                                                        found_mentions_tokens_ids):
                                                    # check if a head the candidate head is outside the mention's boundaries
                                                    if t.head.text in tokens_text:
                                                        # a head of a candiate head cannot be in the text of the mention
                                                        continue

                                                mention_head = t
                                                if mention_head.pos_ == "DET":
                                                    mention_head = None
                                                    continue

                                                to_break = False
                                                # remap the mention head back to the np4e original tokenization to get the ID for the output
                                                for t_orig, t_mapped in token_found.items():
                                                    if t_mapped == mention_head.i:
                                                        mention_head_id = t_orig
                                                        to_break = True
                                                        break
                                                if to_break:
                                                    break

                                    # add to manual review if the resulting token is not inside the mention
                                    # (error must have happened)
                                    if mention_head is None:  # also "if is None"
                                        if mention_id not in need_manual_review_mention_head:
                                            need_manual_review_mention_head[mention_id] = \
                                                {
                                                    "mention_text": list(zip(mention_tokens_ids_global, tokens_text)),
                                                    "sentence_tokens": list(enumerate(sent_tokens)),
                                                    "spacy_sentence_tokens": [(i, t.text) for i, t in enumerate(doc)],
                                                    "tolerance": tolerance
                                                }
                                            LOGGER.warning(
                                                f"Mention with ID {doc_id}/{mention_id} ({tokens_str}) needs manual review. Could not "
                                                f"determine the mention head automatically. {str(tolerance)}")

                                    token_mention_start_id = int(mention_tokens_ids_global[0])
                                    # context_min_id = 0 if token_mention_start_id - CONTEXT_RANGE < 0 else token_mention_start_id - CONTEXT_RANGE

                                    doc_df.loc[:, "token_id_global"] = list(range(len(doc_df)))

                                    if token_mention_start_id - CONTEXT_RANGE < 0:
                                        context_min_id = 0
                                        tokens_number_context = list(
                                            doc_df[(doc_df[SENT_ID] == sent_id) & (doc_df[TOKEN_ID].isin(token_ids))][
                                                "token_id_global"])
                                    else:
                                        context_min_id = token_mention_start_id - CONTEXT_RANGE
                                        global_token_ids = list(
                                            doc_df[(doc_df[SENT_ID] == sent_id) & (doc_df[TOKEN_ID].isin(token_ids))][
                                                "token_id_global"])
                                        tokens_number_context = [int(t - context_min_id) for t in global_token_ids]

                                    context_max_id = min(token_mention_start_id + CONTEXT_RANGE, len(doc_df))
                                    mention_context_str = list(doc_df.iloc[context_min_id:context_max_id][TOKEN].values)

                                    # add to mentions if the variables are correct ( do not add for manual review needed )
                                    if mention_id not in need_manual_review_mention_head:
                                        mention_ner = mention_head.ent_type_ if mention_head.ent_type_ != "" else "O"
                                        mention_text = ""
                                        for t in mention_tokens_ids_global:
                                            mention_text, _, _ = append_text(mention_text, token_dict[t]["text"])

                                        mentions[subelem.attrib["m_id"]] = {"type": subelem.tag,
                                                                            "text": mention_text,
                                                                            "sent_doc": doc,
                                                                            "source": path.split("\\")[-1].split("-")[0],
                                                                            LANGUAGE: language,
                                                                            MENTION_ID: mention_id,
                                                                            MENTION_NER: mention_ner,
                                                                            MENTION_HEAD_POS: mention_head.pos_,
                                                                            MENTION_HEAD_LEMMA: mention_head.lemma_,
                                                                            MENTION_HEAD: mention_head.text,
                                                                            MENTION_HEAD_ID: mention_head_id,
                                                                            TOKENS_NUMBER: sent_tokens,
                                                                            TOKENS_TEXT: [str(token_dict[t]["text"]) for t in mention_tokens_ids_global],
                                                                            DOC_ID: doc_id,
                                                                            DOC: doc_id_full,
                                                                            SENT_ID: int(sent_id),
                                                                            MENTION_CONTEXT: mention_context_str,
                                                                            TOKENS_NUMBER_CONTEXT: tokens_number_context,
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
            mention_id = f'{m["doc_id"]}_{str(chain_id)}_{str(m["sent_id"])}_{shortuuid.uuid(m["text"])}'

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
                       TOKENS_NUMBER_CONTEXT: m[TOKENS_NUMBER_CONTEXT],
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
    conll_df_labeled = make_save_conll(conll_df, df_all_mentions, OUT_PATH)

    LOGGER.info("Splitting the dataset into train/val/test subsets...")

    with open("train_val_test_split.json", "r") as file:
        train_val_test_dict = json.load(file)

    for split, topic_ids in train_val_test_dict.items():
        conll_df_split = pd.DataFrame()
        for topic_id in topic_ids:
            conll_df_split = pd.concat([conll_df_split,
                                        conll_df_labeled[conll_df_labeled[TOPIC_SUBTOPIC_DOC].str.contains(f'^{topic_id}/')]])
        event_mentions_split = [m for m in event_mentions if any([m[TOPIC_ID] in topic_name for topic_name in topic_ids])]
        entity_mentions_split = [m for m in entity_mentions if any([m[TOPIC_ID] in topic_name for topic_name in topic_ids])]

        output_folder_split = os.path.join(OUT_PATH, split)
        if not os.path.exists(output_folder_split):
            os.mkdir(output_folder_split)

        with open(os.path.join(output_folder_split, MENTIONS_EVENTS_JSON), 'w', encoding='utf-8') as file:
            json.dump(event_mentions_split, file)

        with open(os.path.join(output_folder_split, MENTIONS_ENTITIES_JSON), 'w', encoding='utf-8') as file:
            json.dump(entity_mentions_split, file)

        make_save_conll(conll_df_split, event_mentions_split+entity_mentions_split, output_folder_split, assign_reference_labels=False)


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
    conv_files([EN])
