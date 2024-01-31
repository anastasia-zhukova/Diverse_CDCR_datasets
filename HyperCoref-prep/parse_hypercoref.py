import os

import pandas as pd
from utils import *
import shortuuid
import itertools
import random
import math

from setup import *

source_path = os.path.join(os.getcwd(), HYPERCOREF_FOLDER_NAME)
output_path = os.path.join(os.getcwd(), OUTPUT_FOLDER_NAME)
nlp = spacy.load('en_core_web_sm')
CHUNK_SIZE = 1000
MENTION_TYPE_EVENT = "EVENT"
TRAIN_SPLIT_SIZE = 25000
DEV_ABC_SPLIT_SIZE = 1700
DEV_BBC_SPLIT_SIZE = 4200
random.seed(42)


def divide_chunks(l, n):
    '''
    Yield successive n-sized chunks from l.
    Args:
        l: list
        n: size of chunk

    Returns: chuncked list

    '''
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_subtopic(doc_orig: str):
    if doc_orig.endswith("/story"):
        return "_".join(doc_orig.split("/")[1: -2])
    return "_".join(doc_orig.split("/")[1: -1])


def downscale_mentions(mentions_input_df: pd.DataFrame, split_name: str, topic_name: str) -> pd.DataFrame:
    """
    Filter the mentions from train and dev. According to Bugert et al 2021, they used 25K of mentions per publisher
    (topic) and only from the chains of the sized 2-6 mentions.
    We follow the same methodology but 1) focus on selecting chains with 3+ mentions rather than 2,
    2) keep some singletons to be used to build negative pairs
    For dev, they used 1,7K for ABC and 4,2K for BBC. We will downscale to the same numbers.
    Args:
        mentions_input_df: original dataframe with mentions
        split_name: name of a split
        topic_name: origin of mentions

    Returns:

    """
    if split_name == "test":
        return mentions_input_df

    # default value
    downsample_size = TRAIN_SPLIT_SIZE
    if split_name == "val":
        if topic_name.startswith("abcnews"):
            downsample_size = DEV_ABC_SPLIT_SIZE
        elif topic_name.startswith("bbc"):
            downsample_size = DEV_BBC_SPLIT_SIZE

    mention_df_downsampled = pd.DataFrame()
    chain_size_df = mentions_input_df.groupby("to-url-normalized").count().sort_values(by=["url-normalized"],
                                                                                              ascending=[False])
    # start collecting mentions from the largest chains
    for chain_size in chain_size_df["url-normalized"].unique():
        fitting_chains_df = chain_size_df[chain_size_df["url-normalized"] == chain_size]

        # check how much more mentions can still be taken
        if fitting_chains_df["url-normalized"].sum() + len(mention_df_downsampled) <= downsample_size:
            selected_chain_names = list(fitting_chains_df.index)
        else:
            remaining_number = downsample_size - len(mention_df_downsampled)
            selected_chain_names = random.sample(list(fitting_chains_df.index),
                                                 math.floor(remaining_number / chain_size))

        selected_mentions_df = mentions_input_df[
            mentions_input_df["to-url-normalized"].isin(selected_chain_names)]
        mention_df_downsampled = pd.concat([mention_df_downsampled, selected_mentions_df])
    return mention_df_downsampled


def parse_hypercoref_data():
    # merge all train/dev/test parts
    # replace dev with val
    mentions_event_global_dict = {} # for the splits
    conll_df_dict = {} # also for the splits
    need_manual_review_mention_head = {}

    for publisher_folder_name in os.listdir(source_path):
        if publisher_folder_name.startswith("abc"):
            continue

        if os.path.isfile(os.path.join(source_path, publisher_folder_name)):
            continue

        LOGGER.info(f'Processing {publisher_folder_name} topic...')
        topic_id = publisher_folder_name

        data_folder = os.path.join(source_path, publisher_folder_name, "6_CreateSplitsStage_create_splits")
        for split_name_orig in os.listdir(data_folder):

            if split_name_orig == "dev":
                split_name = "val"
            else:
                split_name = split_name_orig

            if split_name not in mentions_event_global_dict:
                mentions_event_global_dict[split_name] = []
                conll_df_dict[split_name] = []

            doc_chunk_id_prev = len(conll_df_dict[split_name])

            save_path = os.path.join(output_path, split_name)
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            LOGGER.info(f"Reading data from {split_name_orig} split...")
            split_folder = os.path.join(data_folder, split_name_orig)
            mention_df_orig = pd.read_parquet(os.path.join(split_folder, 'hyperlinks.parquet'), engine='pyarrow')
            tokens_df_orig = pd.read_parquet(os.path.join(split_folder, 'tokens.parquet'), engine='pyarrow')
            sentences_df_orig = pd.read_parquet(os.path.join(split_folder, 'sentences.parquet'), engine='pyarrow')

            # remove mentions that are lonely single in a document (documents of such singletons are balast in the dataset)
            mentions_per_doc_df = mention_df_orig.groupby("url-normalized").count()
            chain_size_df = mention_df_orig.groupby("to-url-normalized").count()

            mention_df_orig_filtered_list = []
            for index, row in mention_df_orig.iterrows():
                mentions_in_doc = mentions_per_doc_df.loc[row["url-normalized"], "to-url"]
                chain_size = chain_size_df.loc[row["to-url-normalized"], "to-url"]
                if mentions_in_doc == 1 and chain_size == 1:
                    continue

                mention_df_orig_filtered_list.append(row)

            mention_df_orig_filtered = pd.DataFrame(mention_df_orig_filtered_list)

            def __create_conll_key(doc_orig: str) -> Tuple[str, str]:
                """
                Creates a CoNLL key of topic_id/subtopic_id/doc_id and returns it together with the doc_id generated
                based on the doc name
                Args:
                    doc_orig: an original document name

                Returns: conll_key and doc_id
                """
                subtopic_id = get_subtopic(doc_orig)
                doc_id = shortuuid.uuid(doc_orig.replace("/", "_"))
                return f"{topic_id}/{subtopic_id}/{doc_id}", doc_id

            # conll format
            conll_df_split = tokens_df_orig.reset_index()
            conll_df_split.rename(columns={"url-normalized": DOC, "sentence-idx": SENT_ID, "token-idx": TOKEN_ID}, inplace=True)
            conll_df_split[REFERENCE] = "-"
            conll_df_split[TOPIC_SUBTOPIC_DOC], conll_df_split[DOC_ID] = zip(*conll_df_split[DOC].map(__create_conll_key))
            conll_df_split.index = conll_df_split.apply(lambda v: f'{v[DOC_ID]}/{v[SENT_ID]}/{v[TOKEN_ID]}', axis=1)
            conll_df_split = conll_df_split[[TOPIC_SUBTOPIC_DOC, DOC_ID, DOC, SENT_ID, TOKEN_ID, TOKEN, REFERENCE]]

            mention_df_orig_downscaled = downscale_mentions(mention_df_orig_filtered, split_name=split_name, topic_name=publisher_folder_name)

            doc_with_selected_mentions = set(mention_df_orig_downscaled["url-normalized"].values)
            conll_df_split_filtered = conll_df_split[conll_df_split[DOC].isin(doc_with_selected_mentions)]
            doc_group_dict = {group_name: group for group_name, group in conll_df_split_filtered.groupby(DOC)}
            coref_chain_dict = {group_name: group for group_name, group in mention_df_orig_downscaled.groupby("to-url-normalized")}

            # the files are too big, we will chunk them
            doc_chunks = list(divide_chunks(list(doc_group_dict.items()), CHUNK_SIZE))

            for doc_chunk_id, doc_chunk in enumerate(doc_chunks, start=doc_chunk_id_prev):
                LOGGER.info(f"Creating conll for the documents from {split_name_orig}: chunk #{doc_chunk_id} out of {len(doc_chunks) + doc_chunk_id_prev - 1}...")
                conll_df_chunk = pd.DataFrame()
                event_mentions_split_chunk = []

                for doc, doc_df in tqdm(doc_chunk, total=len(doc_chunk)):
                    mention_orig_chunk = mention_df_orig_downscaled[mention_df_orig_downscaled["url-normalized"] == doc]
                    doc_df.loc[:, "token_id_global"] = list(range(len(doc_df)))

                    for mention_index, mention_row in mention_orig_chunk.iterrows():
                        mention_id_sent_table = (mention_row["url-normalized"], mention_row["sentence-idx"])
                        mention_id = f'{publisher_folder_name}_{mention_index}'
                        coref_chain = mention_row["to-url-normalized"]
                        doc = mention_row["url-normalized"].replace("/", "_")
                        doc_id = shortuuid.uuid(doc)
                        sent_id = mention_row["sentence-idx"]
                        sentence_str = sentences_df_orig.loc[mention_id_sent_table]["sentence"]
                        tokens_str = sentence_str[mention_row["chars-start"]: mention_row["chars-end"]]
                        tokens_id_start = (mention_row["url-normalized"], mention_row["sentence-idx"], mention_row["token-idx-from"])
                        tokens_id_end = (mention_row["url-normalized"], mention_row["sentence-idx"], mention_row["token-idx-to"] - 1)
                        tokens_text = list(tokens_df_orig.loc[tokens_id_start: tokens_id_end]["token"].values)
                        token_ids = list(range(mention_row["token-idx-from"], mention_row["token-idx-to"]))
                        subtopic_id = get_subtopic(mention_row["url-normalized"])
                        subtopic = subtopic_id

                        token_mention_start_id_global = \
                        doc_df[(doc_df[SENT_ID] == sent_id) & (doc_df[TOKEN_ID] == token_ids[0])]["token_id_global"][0]
                        if token_mention_start_id_global - CONTEXT_RANGE < 0:
                            context_min_id = 0
                            tokens_number_context = list(
                                doc_df[(doc_df[SENT_ID] == sent_id) & (doc_df[TOKEN_ID].isin(token_ids))][
                                    "token_id_global"])
                        else:
                            context_min_id = token_mention_start_id_global - CONTEXT_RANGE
                            global_token_ids = list(
                                doc_df[(doc_df[SENT_ID] == sent_id) & (doc_df[TOKEN_ID].isin(token_ids))][
                                    "token_id_global"])
                            tokens_number_context = [int(t - context_min_id) for t in global_token_ids]

                        context_max_id = min(token_mention_start_id_global + CONTEXT_RANGE, len(doc_df))
                        mention_context_str = list(doc_df.iloc[context_min_id:context_max_id][TOKEN].values)

                        # pass the string into spacy
                        doc_spacy = nlp(sentence_str)

                        tolerance = 0
                        token_found = {t: None for t in token_ids}
                        prev_id = 0
                        for t_id, t in zip(token_ids, tokens_text):
                            to_break = False
                            for tolerance in range(10):
                                for token in doc_spacy[max(0, t_id-tolerance): t_id+tolerance+1]:
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
                        if f'{doc_id}/{mention_id}' not in need_manual_review_mention_head and not any([t is None for t  in token_found.values()]):
                            found_mentions_tokens = doc_spacy[min([t for t in token_found.values()]): max([t for t in token_found.values()]) + 1]
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
                            if f'{doc_id}/{mention_id}' not in need_manual_review_mention_head:
                                sent_tokens = list(
                                    conll_df_split[(conll_df_split[DOC_ID] == doc_id) & (conll_df_split[SENT_ID] == sent_id)][TOKEN])
                                need_manual_review_mention_head[f'{doc_id}/{mention_id}'] = \
                                    {
                                        "mention_text": list(zip(token_ids, tokens_text)),
                                        "sentence_tokens": list(enumerate(sent_tokens)),
                                        "spacy_sentence_tokens": [(i,t.text) for i,t in enumerate(doc_spacy)],
                                        "tolerance": tolerance
                                    }

                                LOGGER.warning(
                                    f"Mention with ID {doc_id}/{mention_id} ({tokens_str}) needs manual review. Could not "
                                    f"determine the mention head automatically. {str(tolerance)}")

                        if f'{doc_id}/{mention_id}' not in need_manual_review_mention_head:
                            mention_ner =  mention_head.ent_type_ if mention_head.ent_type_ != "" else "O"
                            mention = {COREF_CHAIN: coref_chain,
                                       MENTION_NER: mention_ner,
                                       MENTION_HEAD_POS: mention_head.pos_,
                                       MENTION_HEAD_LEMMA: mention_head.lemma_,
                                       MENTION_HEAD: mention_head.text,
                                       MENTION_HEAD_ID: mention_head_id,
                                       DOC_ID: doc_id,
                                       DOC: doc,
                                       IS_CONTINIOUS: True,
                                       IS_SINGLETON: len(coref_chain_dict[coref_chain]) == 1,
                                       MENTION_ID: mention_id,
                                       MENTION_TYPE: MENTION_TYPE_EVENT[:3],
                                       MENTION_FULL_TYPE: MENTION_TYPE_EVENT,
                                       SCORE: -1.0,
                                       SENT_ID: mention_row["sentence-idx"],
                                       MENTION_CONTEXT: mention_context_str,
                                       TOKENS_NUMBER_CONTEXT: tokens_number_context,
                                       TOKENS_NUMBER: token_ids,
                                       TOKENS_STR: tokens_str,
                                       TOKENS_TEXT: tokens_text,
                                       TOPIC_ID: topic_id,
                                       TOPIC: publisher_folder_name,
                                       SUBTOPIC_ID: subtopic_id,
                                       SUBTOPIC: subtopic,
                                       COREF_TYPE: IDENTITY,
                                       DESCRIPTION: coref_chain,
                                       CONLL_DOC_KEY: f'{topic_id}/{subtopic_id}/{doc_id}',
                                       }
                            event_mentions_split_chunk.append(mention)
                    conll_df_chunk = pd.concat([conll_df_chunk, doc_df.drop(columns=["token_id_global"])])

                mentions_event_global_dict[split_name].extend(event_mentions_split_chunk)

                # backup the mentions after each chunk
                with open(os.path.join(save_path, MENTIONS_EVENTS_JSON), "w") as file:
                    json.dump(mentions_event_global_dict[split_name], file)

                with open(os.path.join(save_path, MENTIONS_ENTITIES_JSON), "w") as file:
                    json.dump([], file)

                conll_df_local_chunk_labeled = make_save_conll(conll_df_chunk, event_mentions_split_chunk, save_path,
                                                                   part_id=doc_chunk_id)
                conll_df_dict[split_name].append(conll_df_local_chunk_labeled)

    event_mentions = list(itertools.chain.from_iterable(mentions_event_global_dict.values()))
    with open(os.path.join(output_path, MENTIONS_EVENTS_JSON), "w") as file:
        json.dump(event_mentions, file)

    with open(os.path.join(output_path, MENTIONS_ENTITIES_JSON), "w") as file:
        json.dump([], file)

    df_all_mentions = pd.DataFrame()
    for mention in event_mentions:
        df_all_mentions = pd.concat([df_all_mentions, pd.DataFrame({
            attr: str(value) if type(value) == list else value for attr, value in mention.items()
        }, index=[mention[MENTION_ID]])], axis=0)
    df_all_mentions.to_csv(os.path.join(output_path, MENTIONS_ALL_CSV))

    conll_df = pd.concat(list(itertools.chain.from_iterable(conll_df_dict.values())))
    conll_df.to_csv(os.path.join(output_path, CONLL_CSV))
    # make_save_conll(conll_df, df_all_mentions, OUT_PATH, assign_reference_labels=False)

    LOGGER.info(
        f'\nNumber of unique mentions: {len(event_mentions)} '
        f'\nNumber of unique chains: {len(set(df_all_mentions[COREF_CHAIN].values))} ')
    LOGGER.info(f'Parsing of WEC-Eng is done!')


if __name__ == '__main__':
    parse_hypercoref_data()
