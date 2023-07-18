import pandas as pd
import json
import os
import shortuuid
from tqdm import tqdm

from utils import *
from setup import *

source_folder = os.path.join(os.getcwd(), CD2CR_FOLDER_NAME)
output_folder = os.path.join(os.getcwd(), OUTPUT_FOLDER_NAME)
nlp = spacy.load('en_core_web_sm')


def conv_files():
    conll_df = pd.DataFrame()
    entity_mentions = []
    topic_id = "0"
    topic_name = "0_science_technology"
    subtopic_id_global = 0
    subtopic_map_dict = {}
    need_manual_review_mention_head = {}
    chain_dict = {}

    with open(os.path.join(source_folder, "sci_papers.json"), "r") as file:
        sci_papers = json.load(file)

    with open(os.path.join(source_folder, "news_urls.json"), "r") as file:
        news_urls = json.load(file)

    for split, conll_file, mentions_file in zip(["train", "val", "test"],
                                                ["train.conll", "dev.conll", "test.conll"],
                                                ["train_entities.json", "dev_entities.json", "test_entities.json"]):
        if split not in subtopic_map_dict:
            subtopic_map_dict[split] = {}

        # read conll file

        LOGGER.info(f'Reading CoNLL files of {split} split...')
        conll_split_df = pd.DataFrame()
        entity_mentions_split = []
        with open(os.path.join(source_folder, conll_file), "r", encoding="utf-8") as file:
            conll_txt = file.readlines()

        sent_id_prev = -1
        doc_id_prev = -1
        token_id = 0
        for line in tqdm(conll_txt):
            if line.startswith("#"):
                continue

            subtopic_id, _, doc_id, sent_id, token_id_global, token, is_headline, chain_value = line.strip().split("\t")
            if subtopic_id not in subtopic_map_dict[split]:
                subtopic_map_dict[split][subtopic_id] = subtopic_id_global
                subtopic_id_global += 1

            chain_id = re.sub("\D+", "", chain_value)

            if chain_value.strip() == f'({chain_id}' or chain_value.strip() == f'({chain_id})':
                chain_dict[chain_id] = chain_dict.get(chain_id, 0) + 1

            if sent_id != sent_id_prev or doc_id != doc_id_prev:
                token_id = 0

            conll_split_df = pd.concat([conll_split_df, pd.DataFrame({
                TOPIC_SUBTOPIC_DOC: f"{topic_id}/{subtopic_map_dict[split][subtopic_id]}/{doc_id}",
                DOC_ID: doc_id,
                SENT_ID: int(sent_id),
                TOKEN_ID: int(token_id),
                "token_id_global": int(token_id_global),
                TOKEN: token,
                # REFERENCE: chain_value
                REFERENCE: '-' #we will need to reassign the references because not mentions are in conll
            }, index=[f'{doc_id}/{token_id_global}'])])

            token_id += 1
            sent_id_prev = sent_id
            doc_id_prev = doc_id

        # read mentions file
        coref_info_dict = {}
        LOGGER.info(f'Reading mentions of {split} split...')
        with open(os.path.join(source_folder, mentions_file), "r") as file:
            entity_mention_orig = json.load(file)

        for mention_orig in tqdm(entity_mention_orig):
            mention_id = shortuuid.uuid(f'{mention_orig[DOC_ID]}/{mention_orig["tokens"]}')
            doc_id = mention_orig[DOC_ID]
            sent_id = mention_orig["sentence_id"]
            token_ids_global = mention_orig["tokens_ids"]
            markable_df = conll_split_df[(conll_split_df[DOC_ID] == doc_id) & (conll_split_df[SENT_ID] == sent_id) &
                                         (conll_split_df["token_id_global"].isin(token_ids_global))]

            if not len(markable_df):
                LOGGER.warning(f'Mention {mention_orig} is not found and skipped.' )
                continue

            if len(token_ids_global) != len(markable_df):
                token_ids_global = list(markable_df["token_id_global"].values)

            # mention attributes
            token_ids = [int(t) for t in markable_df[TOKEN_ID].values]
            token_str = ""
            tokens_text = list(markable_df[TOKEN].values)
            for token in tokens_text:
                token_str, word_fixed, no_whitespace = append_text(token_str, token)

            # determine the sentences as a string
            sentence_str = ""
            sent_tokens = list(conll_split_df[(conll_split_df[DOC_ID] == doc_id) & (conll_split_df[SENT_ID] == sent_id)][TOKEN])
            for t in sent_tokens:
                sentence_str, _, _ = append_text(sentence_str, t)

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
            if mention_head_id is None:  # also "if is None"
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

            doc_df = conll_split_df[conll_split_df[DOC_ID] == doc_id]
            # doc_df.loc[:, "token_id_global"] = list(range(len(doc_df)))
            token_mention_start_id = token_ids_global[0]
            # token_mention_start_id = list(doc_df.index).index(f'{doc_id}/{token_ids[0]}')
            # context_min_id = 0 if token_mention_start_id - CONTEXT_RANGE < 0 else token_mention_start_id - CONTEXT_RANGE
            # context_max_id = min(token_mention_start_id + CONTEXT_RANGE, len(doc_df))

            if token_mention_start_id - CONTEXT_RANGE < 0:
                context_min_id = 0
                tokens_number_context = list(doc_df[(doc_df[SENT_ID] == sent_id) & (doc_df[TOKEN_ID].isin(token_ids))]["token_id_global"])
            else:
                context_min_id = token_mention_start_id - CONTEXT_RANGE
                global_token_ids = list(doc_df[(doc_df[SENT_ID] == sent_id) & (doc_df[TOKEN_ID].isin(token_ids))]["token_id_global"])
                tokens_number_context = [int(t - context_min_id) for t in global_token_ids]

            context_max_id = min(token_mention_start_id + CONTEXT_RANGE, len(doc_df))
            mention_context_str = list(doc_df.iloc[context_min_id:context_max_id][TOKEN].values)

            # add to mentions if the variables are correct ( do not add for manual review needed )
            if f'{doc_id}/{mention_id}' not in need_manual_review_mention_head:
                mention_ner = mention_head.ent_type_ if mention_head.ent_type_ != "" else "O"
                mention_type = mention_ner if mention_ner != "O" else "OTHER"
                subtopic_id = str(subtopic_map_dict[split][str(mention_orig["topic"])])

                doc_id_key = doc_id.split("_")[-1]
                if doc_id_key in sci_papers:
                    doc_name = sci_papers[doc_id_key]["doi"]
                elif doc_id_key in news_urls:
                    doc_name = news_urls[doc_id_key]["url"].split("/")[-1]
                else:
                    doc_name = doc_id

                coref_id = str(mention_orig["cluster_id"])
                if coref_id not in coref_info_dict:
                    coref_info_dict[coref_id] = set()
                # required to create chain descriptions later
                coref_info_dict[coref_id].add(mention_head.text)

                mention = {COREF_CHAIN: coref_id,
                           MENTION_NER: mention_ner,
                           MENTION_HEAD_POS: mention_head.pos_,
                           MENTION_HEAD_LEMMA: mention_head.lemma_,
                           MENTION_HEAD: mention_head.text,
                           MENTION_HEAD_ID: mention_head_id,
                           DOC_ID: doc_id,
                           DOC: doc_name,
                           IS_CONTINIOUS: True if token_ids == list(range(token_ids[0], token_ids[-1] + 1)) else False,
                           IS_SINGLETON: coref_id not in chain_dict, # singletons were not reflected in the conll files
                           MENTION_ID: mention_id,
                           MENTION_TYPE: mention_type[:3],
                           MENTION_FULL_TYPE: mention_type,
                           SCORE: -1.0,
                           SENT_ID: sent_id,
                           MENTION_CONTEXT: mention_context_str,
                           TOKENS_NUMBER_CONTEXT: tokens_number_context,
                           TOKENS_NUMBER: token_ids,
                           TOKENS_STR: mention_orig["tokens"],
                           TOKENS_TEXT: tokens_text,
                           TOPIC_ID: topic_id,
                           TOPIC: topic_name,
                           SUBTOPIC_ID: subtopic_id,
                           SUBTOPIC: subtopic_id,
                           COREF_TYPE: IDENTITY,
                           DESCRIPTION: "", # the original description is "something" for all chains, so we will generate it below
                           CONLL_DOC_KEY: f'{topic_id}/{subtopic_id}/{doc_id}',
                           }
                entity_mentions_split.append(mention)

        #add chain descriptions
        for mention in entity_mentions_split:
            mention[DESCRIPTION] = "_".join(coref_info_dict[mention[COREF_CHAIN]])

        save_path = os.path.join(output_folder, split)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        with open(os.path.join(save_path, MENTIONS_EVENTS_JSON), "w") as file:
            json.dump([], file)

        with open(os.path.join(save_path, MENTIONS_ENTITIES_JSON), "w") as file:
            json.dump(entity_mentions_split, file)

        conll_split_df_labeled = make_save_conll(conll_split_df.drop(columns="token_id_global"), entity_mentions_split, save_path)
        conll_df = pd.concat([conll_df, conll_split_df_labeled])
        entity_mentions.extend(entity_mentions_split)

    df_all_mentions = pd.DataFrame()
    for mention in tqdm(entity_mentions):
        df_all_mentions = pd.concat([df_all_mentions, pd.DataFrame({
            attr: str(value) if type(value) == list else value for attr, value in mention.items()
        }, index=[mention[MENTION_ID]])], axis=0)
    df_all_mentions.to_csv(os.path.join(output_folder, MENTIONS_ALL_CSV))

    with open(os.path.join(output_folder, MENTIONS_EVENTS_JSON), "w") as file:
        json.dump([], file)

    with open(os.path.join(output_folder, MENTIONS_ENTITIES_JSON), "w") as file:
        json.dump(entity_mentions, file)

    LOGGER.info(
        "Mentions that need manual review to define the head and its attributes have been saved to: " +
        MANUAL_REVIEW_FILE + " - Total: " + str(len(need_manual_review_mention_head)))

    with open(os.path.join(TMP_PATH, MANUAL_REVIEW_FILE), "w", encoding='utf-8') as file:
        json.dump(need_manual_review_mention_head, file)

    make_save_conll(conll_df, entity_mentions, output_folder, assign_reference_labels=False)
    LOGGER.info(
        f'\nNumber of unique mentions: {len(entity_mentions)} '
        f'\nNumber of unique chains: {len(set(df_all_mentions[COREF_CHAIN].values))} ')
    LOGGER.info(f'Parsing of CD2CR is done!')


if __name__ == '__main__':
    LOGGER.info(f"Processing CD2CR {source_folder[-34:].split('_')[2]}.")
    conv_files()
