import os
import json
import sys
import string
import spacy
import re
import pandas as pd
import shortuuid
from utils import *
from tqdm import tqdm
from setup import *
from logger import LOGGER

FCC_PARSING_FOLDER = os.path.join(os.getcwd())
source_path = os.path.join(FCC_PARSING_FOLDER, FCC_FOLDER_NAME)
nlp = spacy.load('en_core_web_sm')
LOGGER.info("Spacy model loaded.")


def conv_files():
    topic_name = "0_football"
    topic_id = "0"
    documents_df = pd.DataFrame()
    conll_df_fcc = pd.DataFrame()
    conll_df_fcc_t = pd.DataFrame()
    all_mentions_dict = {"event": [], "entity": [], "event_sentence": []}
    need_manual_review_mention_head = {}

    for split in ["dev", "test", "train"]:
        LOGGER.info(f"Reading {split} split...")
        all_mentions_dict_local = {"event": [], "entity": [], "event_sentence": []}
        documents_df = pd.concat([documents_df,
                                  pd.read_csv(os.path.join(source_path, "2020-10-05_FCC_cleaned", split, "documents.csv"), index_col=[0]).fillna("")])

        tokens_df = pd.read_csv(os.path.join(source_path, "2020-10-05_FCC_cleaned", split, "tokens.csv"))

        conll_df_local = pd.DataFrame()
        for index, row in tqdm(tokens_df.iterrows(), total=tokens_df.shape[0]):
            conll_df_local = pd.concat([conll_df_local, pd.DataFrame({
                TOPIC_SUBTOPIC_DOC: f"{topic_id}/{documents_df.loc[row['doc-id'], 'collection']}/{row['doc-id']}",
                DOC_ID: row["doc-id"],
                SENT_ID: int(row["sentence-idx"]),
                TOKEN_ID: int(row["token-idx"]),
                TOKEN: row["token"],
                REFERENCE: "-"
            }, index=[f'{row["doc-id"]}/{row["sentence-idx"]}/{row["token-idx"]}'])])

        mentions_sent_level_df_local = pd.read_csv(
            os.path.join(source_path, "2020-10-05_FCC_cleaned", split, "mentions_cross_subtopic.csv"))

        event_mentions_df = pd.read_csv(os.path.join(source_path, "2020-10-05_FCC-T", split, "with_stacked_actions",
                                                     "cross_subtopic_mentions_action.csv"))
        event_mentions_df["event"] = [re.sub("other_event", "other_event_" + shortuuid.uuid(f'{row["doc-id"]}/{row["mention-id"]}')[:6], row["event"])
                                      if "other_event" in row["event"] else row["event"]
                                      for index, row in event_mentions_df.iterrows()]

        semantic_roles_df = pd.read_csv(
            os.path.join(source_path, "2020-10-05_FCC-T", split, "with_stacked_actions", "cross_subtopic_semantic_roles.csv"))

        entity_mentions_df_init = pd.DataFrame()
        for file_name in ["cross_subtopic_mentions_location.csv", "cross_subtopic_mentions_participants.csv",
                          "cross_subtopic_mentions_time.csv"]:
            entity_mentions_df_local = pd.read_csv(
                os.path.join(source_path, "2020-10-05_FCC-T", split, "with_stacked_actions", file_name))
            entity_mentions_df_init = pd.concat([entity_mentions_df_init, entity_mentions_df_local])

        entity_mentions_df = pd.merge(entity_mentions_df_init, semantic_roles_df.rename(columns={"mention-id": "event-mention-id"}),
                                      how="left", left_on=["doc-id", "mention-id"],
                                      right_on=["doc-id", "component-mention-id"])
        entity_mentions_df = pd.merge(entity_mentions_df, event_mentions_df[["doc-id", "mention-id", "event"]].rename(columns={"mention-id": "event-mention-id"}),
                                      how="left", left_on=["doc-id", "event-mention-id"],
                                      right_on=["doc-id", "event-mention-id"])

        for mention_annot_type, mention_init_df in zip(["event", "entity", "event_sentence"],
                                                            [event_mentions_df, entity_mentions_df, mentions_sent_level_df_local]):
            LOGGER.info(f"Parsing {mention_annot_type} mentions...")
            for index, mention_row in tqdm(mention_init_df.iterrows(), total=mention_init_df.shape[0]):
                doc_id = mention_row["doc-id"]
                mention_id_global = f'{mention_row["doc-id"]}/{mention_row["mention-id"]}/{shortuuid.uuid()[:4]}'
                # mention_id = mention_row["mention-id"]

                sentence_df = conll_df_local[(conll_df_local[DOC_ID] == mention_row["doc-id"])
                                             & ((conll_df_local[SENT_ID] == mention_row["sentence-idx"]))]

                if "token-idx-from" in mention_init_df.columns:
                    # tokenized version
                    markable_df = conll_df_local[(conll_df_local[DOC_ID] == mention_row["doc-id"])
                                                 & (conll_df_local[SENT_ID] == mention_row["sentence-idx"])
                                                 & (conll_df_local[TOKEN_ID] >= mention_row["token-idx-from"])
                                                 & (conll_df_local[TOKEN_ID] < mention_row["token-idx-to"])]
                else:
                    markable_df = sentence_df

                token_str = ""
                tokens_text = list(markable_df[TOKEN].values)
                token_ids = list(markable_df[TOKEN_ID].values)
                for token in tokens_text:
                    token_str, word_fixed, no_whitespace = append_text(token_str, token)

                # determine the sentences as a string
                sentence_str = ""
                sentence_tokens = list(sentence_df[TOKEN].values)
                for t in sentence_tokens:
                    sentence_str, _, _ = append_text(sentence_str, t)

                # pass the string into spacy
                doc = nlp(sentence_str)

                tolerance = 0
                token_found = {t: None for t in token_ids}
                prev_id = 0
                for t_id, t in zip(token_ids, tokens_text):
                    to_break = False
                    for tolerance in range(10):
                        for token in doc[max(0, t_id-tolerance): t_id+tolerance+1]:
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
                if mention_id_global not in need_manual_review_mention_head:
                    found_mentions_tokens = doc[min([t for t in token_found.values()]): max([t for t in token_found.values()]) + 1]
                    if len(found_mentions_tokens) == 1:
                        mention_head = found_mentions_tokens[0]
                        # remap the mention head back to the np4e original tokenization to get the ID for the output
                        for t_orig, t_mapped in token_found.items():
                            if t_mapped == mention_head.i:
                                mention_head_id = t_orig
                                break

                    if mention_head is None:
                        found_mentions_tokens_ids = list(token_found.values())
                        # found_mentions_tokens_ids = set([t.i for t in found_mentions_tokens])
                        for i, t in enumerate(found_mentions_tokens):
                            if t.head.i == t.i:
                                # if a token is a root, it is a candidate for the head
                                pass

                            elif t.head.i >= min(found_mentions_tokens_ids) and t.head.i <= max(found_mentions_tokens_ids):
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
                if mention_head_id is None:  # also "if is None"
                    if mention_id_global not in need_manual_review_mention_head:
                        need_manual_review_mention_head[mention_id_global] = \
                            {
                                "mention_text": list(zip(token_ids, tokens_text)),
                                "sentence_tokens": list(enumerate(tokens_text)),
                                "spacy_sentence_tokens": [(i,t.text) for i,t in enumerate(doc)],
                                "tolerance": tolerance
                            }

                        LOGGER.warning(
                            f"Mention with ID {mention_id_global} ({token_str}) needs manual review. Could not "
                            f"determine the mention head automatically. {str(tolerance)}")

                word_start = mention_row["token-idx-from"] if "token-idx-from" in mention_init_df.columns else 0
                doc_df = conll_df_local[conll_df_local[DOC_ID] == doc_id]
                token_mention_start_id = list(doc_df.index).index(f'{doc_id}/{mention_row["sentence-idx"]}/{word_start}')
                context_min_id = 0 if token_mention_start_id-CONTEXT_RANGE < 0 else token_mention_start_id-CONTEXT_RANGE
                context_max_id = min(token_mention_start_id + CONTEXT_RANGE, len(doc_df))

                mention_context_str = list(doc_df.iloc[context_min_id:context_max_id][TOKEN].values)

                # add to mentions if the variables are correct ( do not add for manual review needed )
                if mention_id_global not in need_manual_review_mention_head:
                    # mention_id = mention_row["mention-id"]
                    mention_ner = mention_head.ent_type_ if mention_head.ent_type_ != "" else "O"
                    srl_type = mention_row["mention-type-coarse"] if "mention-type-coarse" in mention_init_df.columns else ""
                    chain_id = mention_row["event"] if "event" in mention_annot_type \
                        else f'{mention_row["event"]}/{srl_type}/{shortuuid.uuid(mention_head.lemma_)[:4]}'

                    subtopic_id = documents_df.loc[doc_id, "collection"]
                    mention_type = mention_row["mention-type"] if "mention-type" in mention_init_df.columns else "OCCURENCE"
                    is_singleton = len(mention_init_df[mention_init_df["event"] == chain_id]) == 1 if "event" in mention_init_df.columns else True

                    mention = {COREF_CHAIN: chain_id,
                               MENTION_NER: mention_ner,
                               MENTION_HEAD_POS:  mention_head.pos_,
                               MENTION_HEAD_LEMMA: mention_head.lemma_,
                               MENTION_HEAD: mention_head.text,
                               MENTION_HEAD_ID: int(mention_head_id),
                               DOC_ID: doc_id,
                               DOC: doc_id,
                               IS_CONTINIOUS: True if token_ids == list(range(token_ids[0], token_ids[-1] + 1)) else False,
                               IS_SINGLETON: is_singleton,
                               MENTION_ID: mention_id_global,
                               MENTION_TYPE: mention_type[:3],
                               MENTION_FULL_TYPE: mention_type,
                               SCORE: -1.0,
                               SENT_ID: mention_row["sentence-idx"],
                               MENTION_CONTEXT: mention_context_str,
                               TOKENS_NUMBER: [int(t) for t in token_ids],
                               TOKENS_STR: token_str,
                               TOKENS_TEXT: tokens_text,
                               TOPIC_ID: topic_id,
                               TOPIC: topic_name,
                               SUBTOPIC_ID: subtopic_id,
                               SUBTOPIC: subtopic_id,
                               COREF_TYPE: IDENTITY,
                               DESCRIPTION: mention_row["event"] if "event" in mention_init_df.columns else "",
                               CONLL_DOC_KEY: f'{topic_id}/{subtopic_id}/{doc_id}',
                               }
                    all_mentions_dict_local[mention_annot_type].append(mention)
                    all_mentions_dict[mention_annot_type].append(mention)

        conll_df_local.reset_index(drop=True, inplace=True)
        save_folder_fcc_t = os.path.join(FCC_PARSING_FOLDER, f'{OUTPUT_FOLDER_NAME}_FCC-T', split)
        if not os.path.exists(os.path.join(FCC_PARSING_FOLDER, f'{OUTPUT_FOLDER_NAME}_FCC-T')):
            os.mkdir(os.path.join(FCC_PARSING_FOLDER, f'{OUTPUT_FOLDER_NAME}_FCC-T'))

        if not os.path.exists(save_folder_fcc_t):
            os.mkdir(save_folder_fcc_t)

        save_folder_fcc = os.path.join(FCC_PARSING_FOLDER, f'{OUTPUT_FOLDER_NAME}_FCC', split)
        if not os.path.exists(os.path.join(FCC_PARSING_FOLDER, f'{OUTPUT_FOLDER_NAME}_FCC')):
            os.mkdir(os.path.join(FCC_PARSING_FOLDER, f'{OUTPUT_FOLDER_NAME}_FCC'))

        if not os.path.exists(save_folder_fcc):
            os.mkdir(save_folder_fcc)

        conll_df_fcc_t = pd.concat([conll_df_fcc_t, make_save_conll(conll_df=conll_df_local,
                                        mentions=all_mentions_dict_local["event"] + all_mentions_dict_local["entity"],
                                        output_folder=save_folder_fcc_t)])

        conll_df_fcc = pd.concat([conll_df_fcc, make_save_conll(conll_df=conll_df_local,
                                        mentions=all_mentions_dict_local["event_sentence"],
                                        output_folder=save_folder_fcc)])

        with open(os.path.join(save_folder_fcc_t, MENTIONS_ENTITIES_JSON), "w", encoding='utf-8') as file:
            json.dump(all_mentions_dict_local["entity"], file)

        with open(os.path.join(save_folder_fcc_t, MENTIONS_EVENTS_JSON), "w", encoding='utf-8') as file:
            json.dump(all_mentions_dict_local["event"], file)

        with open(os.path.join(save_folder_fcc, MENTIONS_EVENTS_JSON), "w", encoding='utf-8') as file:
            json.dump(all_mentions_dict_local["event_sentence"], file)

        with open(os.path.join(save_folder_fcc, MENTIONS_ENTITIES_JSON), "w", encoding='utf-8') as file:
            json.dump([], file)

    # conll_df_fcc.reset_index(drop=True, inplace=True)
    save_folder_fcc_t = os.path.join(FCC_PARSING_FOLDER, f'{OUTPUT_FOLDER_NAME}_FCC-T')
    save_folder_fcc = os.path.join(FCC_PARSING_FOLDER, f'{OUTPUT_FOLDER_NAME}_FCC')
    make_save_conll(conll_df=conll_df_fcc_t,
                    mentions=all_mentions_dict["event"] + all_mentions_dict["entity"],
                    output_folder=save_folder_fcc_t, assign_reference_labels=False)

    make_save_conll(conll_df=conll_df_fcc,
                    mentions=all_mentions_dict["event_sentence"],
                    output_folder=save_folder_fcc, assign_reference_labels=False)

    with open(os.path.join(save_folder_fcc_t, MENTIONS_ENTITIES_JSON), "w", encoding='utf-8') as file:
        json.dump(all_mentions_dict["entity"], file)

    with open(os.path.join(save_folder_fcc_t, MENTIONS_EVENTS_JSON), "w", encoding='utf-8') as file:
        json.dump(all_mentions_dict["event"], file)

    with open(os.path.join(save_folder_fcc, MENTIONS_EVENTS_JSON), "w", encoding='utf-8') as file:
        json.dump(all_mentions_dict["event_sentence"], file)

    with open(os.path.join(save_folder_fcc, MENTIONS_ENTITIES_JSON), "w", encoding='utf-8') as file:
        json.dump([], file)

    df_all_mentions = pd.DataFrame()
    for mention in all_mentions_dict["event"] + all_mentions_dict["entity"]:
        df_all_mentions = pd.concat([df_all_mentions, pd.DataFrame({
            attr: str(value) if type(value) == list else value for attr, value in mention.items()
        }, index=[mention[MENTION_ID]])], axis=0)
    df_all_mentions.to_csv(os.path.join(save_folder_fcc_t, MENTIONS_ALL_CSV))

    df_all_mentions_sent = pd.DataFrame()
    for mention in all_mentions_dict["event_sentence"]:
        df_all_mentions_sent = pd.concat([df_all_mentions_sent, pd.DataFrame({
            attr: str(value) if type(value) == list else value for attr, value in mention.items()
        }, index=[mention[MENTION_ID]])], axis=0)
    df_all_mentions_sent.to_csv(os.path.join(save_folder_fcc, MENTIONS_ALL_CSV))

    LOGGER.info(f'Parsing of FCC and FCC-T is done!')
    LOGGER.info(
        f'\nNumber of unique mentions in FCC: {len(df_all_mentions_sent)} '
        f'\nNumber of unique event chains: {len(set(df_all_mentions_sent[COREF_CHAIN].values))} ')

    LOGGER.info(
        f'\nNumber of unique mentions: {len(df_all_mentions)} '
        f'\nNumber of unique chains: {len(set(df_all_mentions[COREF_CHAIN].values))} ')

    a = 1


if __name__ == '__main__':
    LOGGER.info(f"Processing FCC corpus...")
    conv_files()
