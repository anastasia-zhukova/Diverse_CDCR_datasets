import re
import shortuuid
import pandas as pd
import os
import random
from collections import Counter
from utils import *
from setup import *

source_path = os.path.join(os.getcwd(), CEREC_FOLDER_NAME)
output_path = os.path.join(os.getcwd(), OUTPUT_FOLDER_NAME)
nlp = spacy.load('en_core_web_sm')


def parse_conll():
    conll_df = pd.DataFrame()
    topic_id = "0"
    topic_name = "0_emails"
    need_manual_review_mention_head = {}
    entity_mentions = []
    doc_enumeration = -1
    random.seed(41)

    for split, source_file in zip(["train", "test", "val"], ["cerec.conll", "cerec.validation.14.conll",
                                                              "cerec.validation.20.conll"
                                                              ]):
        LOGGER.info(f"Reading CoNLL for {split} split...")
        mentions_dict = {}
        coref_dict = {}
        conll_df_split = pd.DataFrame()
        entity_mentions_split = []

        with open(os.path.join(source_path, source_file), "r", encoding="utf-8") as file:
            conll_text = file.readlines()

        if split == "train":
            # make a subset of all data
            train_subtopic_ids = []
            topic_counter = 0
            for line_id, line in enumerate(conll_text):
                if line.startswith("#begin"):
                    train_subtopic_ids.append(topic_counter)
                    topic_counter += 1
            selected_subtopics = random.sample(train_subtopic_ids, k=100)
        else:
            selected_subtopics = list(range(int(source_file.split(".")[-2])))

        subtopic = ""
        subtopic_id = ""
        sent_id = 0
        doc_id_prev = ""
        mention_counter = 0
        token_id = 0
        mention_id_list = []
        orig_sent_id_prev = ""
        subtopic_counter = -1

        for line_id, line in tqdm(enumerate(conll_text), total=len(conll_text)):
            if line.startswith("#begin"):
                subtopic = re.sub("#begin document ", "", line)
                subtopic_id = shortuuid.uuid(subtopic)
                subtopic_counter += 1
                continue

            if subtopic_counter not in selected_subtopics:
                continue

            if line.startswith("#end"):
                sent_id = 0
                continue

            if line.startswith("\n"):
                sent_id += 1
                continue

            token, doc_id_orig, _, speaker, _, _, reference = line.replace("\n", "").split("\t\t")

            if speaker == "-":
                speaker = shortuuid.uuid(str(doc_enumeration))[:4]

            if speaker == "SYSTEM":
                continue

            if f'{doc_id_orig}' not in doc_id_prev:
                doc_enumeration += 1
            # if doc_id != doc_id_prev:
                sent_id = 0
                token_id = 0
            else:
                if orig_sent_id_prev != sent_id:
                    # sent_id += 1
                    token_id = 0

            doc_id = f'{doc_id_orig}_{speaker}_{doc_enumeration}'
            topic_subtopic_doc = f'{topic_id}/{subtopic_id}/{doc_id}'

            # add a token to all open mentions
            for mention_id in mention_id_list:
                mentions_dict[mention_id]["words"].append((token, token_id))

            for chain_value in reference.split("|"):
                chain_id = re.sub("\D+", "", chain_value)
                # continue

                chain_composed_id = f'{subtopic.split(".")[0]}_{chain_id}'

                if chain_value.strip() == f'({chain_id}' or chain_value.strip() == f'({chain_id})':
                    coref_dict[chain_composed_id] = coref_dict.get(chain_composed_id, 0) + 1
                    mention_id = shortuuid.uuid(f'{chain_composed_id}_{split}')

                    # if chain_value.strip() == f'({chain_id})':
                    mention_id_compose = f"{mention_id}_{mention_counter}"
                    mention_counter += 1
                    # else:
                    #     mention_id_compose = mention_id

                    # if mention_id_compose not in mentions_dict:
                    mentions_dict[mention_id_compose] = {
                        COREF_CHAIN: chain_composed_id,
                        # DESCRIPTION: "",
                        MENTION_ID: mention_id_compose,
                        DOC_ID: doc_id,
                        DOC: doc_id,
                        SENT_ID: int(sent_id),
                        SUBTOPIC_ID: str(subtopic_id),
                        SUBTOPIC: subtopic.split(".")[0],
                        TOPIC_ID: topic_id,
                        TOPIC: topic_name,
                        COREF_TYPE: IDENTITY,
                        CONLL_DOC_KEY: topic_subtopic_doc,
                        "words": []}
                    mentions_dict[mention_id_compose]["words"].append((token, token_id))

                    if chain_value.strip() != f'({chain_id})':
                        mention_id_list.append(mention_id_compose)

                elif chain_value.strip() == f'{chain_id})':
                    mention_id_base = shortuuid.uuid(f'{chain_composed_id}_{split}')
                    # there is a weird reference encoding with folded reference for the same entity for which I need a workaround
                    # try:
                    mention_id_compose = ""
                    for v in list(mention_id_list)[::-1]:
                        # stack principle
                        if mention_id_base in v:
                            mention_id_compose = v
                            break

                    # mentions_dict[mention_id_compose]["words"].append((token, token_id))
                    # mention = mentions_dict[mention_id_compose]
                    # mention_id_compose = f"{mention_id}_{mention_counter}"
                    # mention[MENTION_ID] = mention_id_compose
                    # mentions_dict.pop(mention_id_compose)
                    # mentions_dict[mention_id_compose] = mention
                    mention_id_list.pop(mention_id_list.index(mention_id_compose))
                    # mention_counter += 1

                    # except KeyError:
                    #     # attack token to the last matching mention

                    #     if len(mention_id_compose):
                    #         mention = mentions_dict[mention_id_compose]
                    #         mention["words"].append((token, token_id))
                    #         mention[MENTION_ID] = mention_id_compose
                    #         if mention_id in mention_id_list:
                    #             mention_id_list.pop(mention_id_list.index(mention_id))

            conll_df_split = pd.concat([conll_df_split, pd.DataFrame({
                TOPIC_SUBTOPIC_DOC: topic_subtopic_doc,
                DOC_ID: doc_id,
                SENT_ID: sent_id,
                TOKEN_ID: token_id,
                TOKEN: token.replace("NEWLINE", "//n"),
                REFERENCE: "-"
            }, index=[f'{subtopic_id}/{doc_id}/{sent_id}/{token_id}'])])
            token_id += 1
            doc_id_prev = doc_id
            orig_sent_id_prev = sent_id

        # parse information about mentions
        coref_type_dict = {}
        LOGGER.info(f"Processing mentions for {split} split...")
        for mention_id, mention in tqdm(mentions_dict.items()):

            orig_token_ids = [w[1] for w in mention["words"]]
            word_start = mention["words"][0][1]
            word_end = mention["words"][-1][1]
            # coref_id = mention_orig["entity"]
            doc_id = mention[DOC_ID]
            sent_id = mention[SENT_ID]
            subtopic_id = mention[SUBTOPIC_ID]
            # markable_df = conll_df_split.loc[f'{subtopic_id}/{doc_id}/{sent_id}/{word_start}': f'{subtopic_id}/{doc_id}/{sent_id}/{word_end}']
            # markable_df = conll_df_split.loc[f'{subtopic_id}/{doc_id}/{sent_id}/{word_start}': f'{subtopic_id}/{doc_id}/{sent_id}/{word_end}']
            markable_df = conll_df_split[(conll_df_split[TOPIC_SUBTOPIC_DOC].str.contains(f'/{subtopic_id}')) & (conll_df_split[DOC_ID] == doc_id)
                                         & (conll_df_split[SENT_ID] == sent_id) & (conll_df_split[TOKEN_ID].isin(orig_token_ids))]
            if not len(markable_df):
                continue

            # mention attributes
            token_ids = [int(t) for t in list(markable_df[TOKEN_ID].values)]
            token_str = ""
            tokens_text = list(markable_df[TOKEN].values)
            for token in tokens_text:
                token_str, word_fixed, no_whitespace = append_text(token_str, token)

            # sent_id = list(markable_df[SENT_ID].values)[0]

            # determine the sentences as a string

            sent_tokens = list(conll_df_split[(conll_df_split[DOC_ID] == doc_id) & (conll_df_split[SENT_ID] == sent_id)][TOKEN])
            sentence_str = " ".join(sent_tokens)
            # for t in sent_tokens:
            #     sentence_str, _, _ = append_text(sentence_str, t)

            # pass the string into spacy
            doc = nlp(sentence_str)

            tolerance = 0
            token_found = {t: None for t in token_ids}
            prev_id = 0
            for t_id, t in zip(token_ids, tokens_text):
                to_break = False
                for tolerance in range(10):
                    for token in doc[max(0, t_id - tolerance): t_id + tolerance + 1]:
                        if token.i < prev_id:
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
                    if to_break:
                        break
            # whole mention string processed, look for the head
            mention_head_id = None
            mention_head = None
            if f'{doc_id}/{mention_id}' not in need_manual_review_mention_head:
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
            if mention_head_id is None or mention_head is None:  # also "if is None"
                if f'{doc_id}/{mention_id}' not in need_manual_review_mention_head:
                    need_manual_review_mention_head[f'{doc_id}/{mention_id}'] = \
                        {
                            "mention_text": list(zip(token_ids, tokens_text)),
                            "sentence_tokens": list(enumerate(sent_tokens)),
                            "spacy_sentence_tokens": [(i, t.text) for i, t in enumerate(doc)],
                            "tolerance": tolerance
                        }

                    LOGGER.warning(
                        f"Mention with ID {doc_id}/{mention_id} ({token_str}) needs manual review. Could not "
                        f"determine the mention head automatically. {str(tolerance)}")

            doc_df = conll_df_split[conll_df_split[DOC_ID] == doc_id]
            token_mention_start_id = list(doc_df.index).index(f'{subtopic_id}/{doc_id}/{sent_id}/{word_start}')
            doc_df.loc[:, "token_id_global"] = list(range(len(doc_df)))

            if token_mention_start_id - CONTEXT_RANGE < 0:
                context_min_id = 0
                tokens_number_context = list(
                    doc_df[(doc_df[SENT_ID] == sent_id) & (doc_df[TOKEN_ID].isin(token_ids))]["token_id_global"])
            else:
                context_min_id = token_mention_start_id - CONTEXT_RANGE
                global_token_ids = list(
                    doc_df[(doc_df[SENT_ID] == sent_id) & (doc_df[TOKEN_ID].isin(token_ids))]["token_id_global"])
                tokens_number_context = [int(t - context_min_id) for t in global_token_ids]

            context_max_id = min(token_mention_start_id + CONTEXT_RANGE, len(doc_df))
            mention_context_str = list(doc_df.iloc[context_min_id:context_max_id][TOKEN].values)

            # add to mentions if the variables are correct ( do not add for manual review needed )
            if f'{doc_id}/{mention_id}' not in need_manual_review_mention_head:
                mention_ner = mention_head.ent_type_ if mention_head.ent_type_ != "" else "O"
                # mention_type = "OCCURRENCE"
                if mention[COREF_CHAIN] not in coref_type_dict:
                    coref_type_dict[mention[COREF_CHAIN]] = {"mentions": [], "ner": [], "heads": []}
                coref_type_dict[mention[COREF_CHAIN]]["ner"].append(mention_ner)
                coref_type_dict[mention[COREF_CHAIN]]["mentions"].append(mention_id)
                if mention_head.pos_ != "PRON":
                    coref_type_dict[mention[COREF_CHAIN]]["heads"].append(mention_head.text)

                mention.update({
                    MENTION_NER: mention_ner,
                    MENTION_HEAD_POS: mention_head.pos_,
                    MENTION_HEAD_LEMMA: mention_head.lemma_,
                    MENTION_HEAD: mention_head.text,
                    MENTION_HEAD_ID: int(mention_head_id),
                    # DOC: "_".join(doc_title_dict[doc_id]),
                    IS_CONTINIOUS: token_ids == list(
                        range(token_ids[0], token_ids[-1] + 1)),
                    IS_SINGLETON: coref_dict[mention[COREF_CHAIN]] == 1,
                    # MENTION_TYPE: mention_type[:3],
                    # MENTION_FULL_TYPE: mention_type,
                    SCORE: -1.0,
                    MENTION_CONTEXT: mention_context_str,
                    TOKENS_NUMBER_CONTEXT: tokens_number_context,
                    TOKENS_NUMBER: token_ids,
                    TOKENS_STR: token_str,
                    TOKENS_TEXT: tokens_text
                })
                mention.pop("words")
                entity_mentions_split.append(mention)

        for coref_chain, values in coref_type_dict.items():
            ner_list = [ner for ner in values["ner"] if ner != 'O']
            if not len(ner_list):
                mention_type = "OTHER"
            else:
                ner_dict = Counter(ner_list)
                # the values are already sorted
                mention_type = list(ner_dict)[0]

            head_dict = Counter(values["heads"])
            if len(head_dict):
                head = list(head_dict)[0]
            else:
                head = ""

            for m_id in values["mentions"]:
                # the mention gets automatically updated in the list of mentions due to the reference variables
                mentions_dict[m_id].update({
                    COREF_CHAIN: f'{mentions_dict[m_id][COREF_CHAIN]}_{mention_type[:3]}',
                    MENTION_TYPE: mention_type[:3],
                    MENTION_FULL_TYPE: mention_type,
                    DESCRIPTION: head
                })

        entity_mentions.extend(entity_mentions_split)

        save_path = os.path.join(output_path, split)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        conll_df_labels = make_save_conll(conll_df_split, entity_mentions_split, save_path)
        conll_df = pd.concat([conll_df, conll_df_labels])

        with open(os.path.join(save_path, MENTIONS_ENTITIES_JSON), "w") as file:
            json.dump(entity_mentions_split, file)

        with open(os.path.join(save_path, MENTIONS_EVENTS_JSON), "w") as file:
            json.dump([], file)

    df_all_mentions = pd.DataFrame()
    for mention in tqdm(entity_mentions):
        df_all_mentions = pd.concat([df_all_mentions, pd.DataFrame({
            attr: str(value) if type(value) == list else value for attr, value in mention.items()
        }, index=[mention[MENTION_ID]])], axis=0)
    df_all_mentions.to_csv(os.path.join(output_path, MENTIONS_ALL_CSV))

    with open(os.path.join(output_path, MENTIONS_ENTITIES_JSON), "w") as file:
        json.dump(entity_mentions, file)

    with open(os.path.join(output_path, MENTIONS_EVENTS_JSON), "w") as file:
        json.dump([], file)

    make_save_conll(conll_df, df_all_mentions, output_path, assign_reference_labels=False)

    LOGGER.info(
        "Mentions that need manual review to define the head and its attributes have been saved to: " +
        MANUAL_REVIEW_FILE + " - Total: " + str(len(need_manual_review_mention_head)))

    with open(os.path.join(TMP_PATH, MANUAL_REVIEW_FILE), "w", encoding='utf-8') as file:
        json.dump(need_manual_review_mention_head, file)

    LOGGER.info(f'Done! \nNumber of unique mentions: {len(df_all_mentions)} '
                f'\nNumber of unique chains: {len(set(df_all_mentions[COREF_CHAIN].values))} ')
    a = 1


if __name__ == '__main__':
    LOGGER.info(f"Processing CEREC {source_path[-34:].split('_')[2]}.")
    parse_conll()
