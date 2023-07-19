import os
from typing import Tuple, Union, List
import re
import shutil
from tqdm import tqdm
from setup import  *
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None


def make_save_conll(conll_df: pd.DataFrame, mentions: Union[List, pd.DataFrame], output_folder: str,
                    assign_reference_labels=True, part_id: int = None) -> pd.DataFrame:
    """
    Universal function that converst a dataframe into a simplified ConLL format for coreference resolution.
    Args:
        conll_df: a conll format in the dataframe format
        mentions: a list or a dataframe of mentions
        output_folder: a path where the files will be saved
        assign_reference_labels: if to perform assignment of the reference labels (last column of CoNLL) or just form a conll file
    Returns:

    """
    conll_df = conll_df.reset_index(drop=True)

    if assign_reference_labels:
        if type(mentions) == pd.DataFrame:
            df_all_mentions = mentions
        else:
            LOGGER.info("Creating a dataframe of mentions...")
            df_all_mentions = pd.DataFrame()
            for mention in tqdm(mentions):
                df_all_mentions = pd.concat([df_all_mentions, pd.DataFrame({
                    attr: str(value) if type(value) == list else value for attr, value in mention.items()
                }, index=[mention[MENTION_ID]])], axis=0)

            if part_id is None:
                df_all_mentions.to_csv(os.path.join(output_folder, MENTIONS_ALL_CSV))
            else:
                df_all_mentions.to_csv(os.path.join(output_folder, f"all_mentions_{part_id}.csv"))

        df_all_mentions[SENT_ID] = df_all_mentions[SENT_ID].astype(int)

        # assigning reference labels to the tokens
        LOGGER.info("Assigning reference labels to the tokens...")
        for i, row in tqdm(conll_df.iterrows(), total=conll_df.shape[0]):
            if row[REFERENCE] is None:
                reference_str = "-"
            else:
                reference_str = row[REFERENCE]

            mention_candidates_df = df_all_mentions[(df_all_mentions[CONLL_DOC_KEY] == row[TOPIC_SUBTOPIC_DOC]) &
                                                    (df_all_mentions[SENT_ID] == row[SENT_ID])]
            mention_candidates_df["start_token"] = 0
            mention_candidates_df["end_token"] = 0
            mention_candidates_df["diff"] = 0
            for mentions_row_id, mention_row in mention_candidates_df.iterrows():
                mention = mention_row.to_dict()
                # the token_ids in the mention are of type int64 and need to be converted
                token_numbers = [int(v) for v in re.sub(r'[\[,\]]+', "", mention[TOKENS_NUMBER]).split(" ")]
                mention_candidates_df.loc[mentions_row_id, "start_token"] = token_numbers[0]
                mention_candidates_df.loc[mentions_row_id, "end_token"] = token_numbers[-1]
                mention_candidates_df.loc[mentions_row_id, "diff"] = token_numbers[-1] - token_numbers[0]

            mention_candidates_df.sort_values(by=["end_token", "diff"], ascending=[False, True], inplace=True)
            for mentions_row_id, mention_row in mention_candidates_df.iterrows():
                mention = mention_row.to_dict()
                token_numbers = [int(v) for v in re.sub(r'[\[,\]]+', "", mention[TOKENS_NUMBER]).split(" ")]
                if row[TOKEN_ID] not in token_numbers:
                    continue

                chain = mention[COREF_CHAIN]
                # one and only token
                if len(token_numbers) == 1 and token_numbers[0] == row[TOKEN_ID]:
                    reference_str = reference_str + '| (' + str(chain) + ')'
                # one of multiple tokes
                elif len(token_numbers) > 1 and token_numbers[0] == row[TOKEN_ID]:
                    reference_str = reference_str + '| (' + str(chain)
                elif len(token_numbers) > 1 and token_numbers[len(token_numbers) - 1] == row[TOKEN_ID]:
                    reference_str = reference_str + '| ' + str(chain) + ')'

            # remove the leading characters if necessary (left from initialization)
            if reference_str.startswith("-| "):
                reference_str = reference_str[3:]
            conll_df.at[i, REFERENCE] = reference_str

    if DOC_ID in conll_df.columns:
        conll_df = conll_df.drop(columns=[DOC_ID])

    # create a conll string from the conll_df
    LOGGER.info("Generating conll string...")
    outputdoc_str = ""
    if part_id is None:
        part_id_to_use = 0
    else:
        part_id_to_use = part_id

    for (topic_local), topic_df in conll_df.groupby(by=[TOPIC_SUBTOPIC_DOC]):
        outputdoc_str += f'#begin document ({topic_local}); part {part_id_to_use}\n'

        for (sent_id_local), sent_df in topic_df.groupby(by=[SENT_ID], sort=[SENT_ID]):
            np.savetxt(os.path.join(TMP_PATH, "tmp.txt"), sent_df.values, fmt='%s', delimiter="\t",
                       encoding="utf-8")

            with open(os.path.join(TMP_PATH, "tmp.txt"), "r", encoding="utf8") as file:
                saved_lines = file.read()

            outputdoc_str += saved_lines + "\n"

        outputdoc_str += "#end document\n"

    # Check if the brackets ( ) are correct
    brackets_1 = 0
    brackets_2 = 0
    doc_id_prev = ""
    for i, row in conll_df.iterrows():
        if row[TOPIC_SUBTOPIC_DOC] != doc_id_prev:
            if brackets_1 != brackets_2:
                LOGGER.warning(f'Number of opening and closing brackets in conll does not match!')
                LOGGER.warning(f"brackets '(' , ')' : {str(brackets_1)}, {str(brackets_2)} at row {doc_id_prev} "
                               f"in the file {output_folder}")
            brackets_1 = 0
            brackets_2 = 0
            doc_id_prev = row[TOPIC_SUBTOPIC_DOC]
            # only count brackets in reference column (exclude token text)
        brackets_1 += str(row[REFERENCE]).count("(")
        brackets_2 += str(row[REFERENCE]).count(")")

    if part_id is None:
        conll_df.to_csv(os.path.join(output_folder, CONLL_CSV))
        with open(os.path.join(output_folder, 'dataset.conll'), "w", encoding='utf-8') as file:
            file.write(outputdoc_str)
    else:
        conll_df.to_csv(os.path.join(output_folder, f"conll_{part_id}.csv"))
        with open(os.path.join(output_folder, f'dataset_{part_id}.conll'), "w", encoding='utf-8') as file:
            file.write(outputdoc_str)
    return conll_df


def append_text(text, word) -> Tuple[str, str, bool]:
    """
    Decides which whitespace to add when addiding a token to the already concatenated tokens.
    :param text: Preceding part to the word of the sentence
    :param word: a word to concatenate
    :return: A sentences with the concatenated word, modifications to this word, and a whitespace delimiter.
    """
    word = "\"" if word == "``" else word

    if not len(text):
        return word, word, True

    space = "" if word in ".,?!)]`\"\'" or word == "'s" else " "
    space = " " if word[0].isupper() and not len(text) else space
    space = " " if word in ["-", "(","\""] else space

    if len(text):
        space = " " if text[-1] in ["\""] and text.count("\"") % 2 == 0 else space  # first "
        space = "" if text[-1] in ["\""] and text.count("\"") % 2 != 0 else space   # second "
        space = "" if text[-1] in ["(", "``", "["] else space
        space = " " if text[-1] in [".,?!)]`\"\'"] else space
        space = "" if text[-1] in ["\""] and word.istitle() else space
        space = "" if word in ["com", "org"] else space

    if len(text) > 1:
        space = "" if word.isupper() and text[-1] == "." and text[-2].isupper() else space
        space = " " if not len(re.sub(r'\W+', "", text[-2:])) and \
                       len(text[-2:]) == len(text[-2:].replace(" ", "")) else space
        space = "" if text[-1] == "." and text[-2] in "0123456789" and len(
            set(word).intersection(set("0123456789"))) > 0 else space

    return text + space + word, word, space == ""


def form_benchmark():
    """
    Takes all specified datasets and places into one folder for upload.
    """
    LOGGER.info("Creating a CDCR benchmark (protected)...")
    benchmark_folder = os.path.join(TMP_PATH, "CDCR_benchmark")
    if not os.path.exists(benchmark_folder):
        os.mkdir(benchmark_folder)

    for dataset_name, source_folder in tqdm(DIRECTORIES_TO_SUMMARIZE.items()):
        dataset = dataset_name.split("-")[0].replace("_", "-")
        output_folder = os.path.join(benchmark_folder, dataset)
        shutil.copytree(source_folder, output_folder)

    LOGGER.info("Archiving the datasets...")
    shutil.make_archive(os.path.join(TMP_PATH, "CDCR_benchmark"), 'zip', os.path.join(TMP_PATH, "CDCR_benchmark"))

    LOGGER.info("Creating a CDCR benchmark (public)...")
    benchmark_folder = os.path.join(TMP_PATH, "CDCR_benchmark_public")
    if not os.path.exists(benchmark_folder):
        os.mkdir(benchmark_folder)

    for dataset_name, source_folder in tqdm(DIRECTORIES_TO_SUMMARIZE.items()):
        dataset = dataset_name.split("-")[0].replace("_", "-")
        if dataset in ["NewsWCL50", "NiDENT", "FCC", "FCC-T"]:
            continue

        output_folder = os.path.join(benchmark_folder, dataset)
        shutil.copytree(source_folder, output_folder)

    LOGGER.info("Archiving the datasets...")
    shutil.make_archive(os.path.join(TMP_PATH, "CDCR_benchmark_public"), 'zip', os.path.join(TMP_PATH, "CDCR_benchmark_public"))
    LOGGER.info("Completed creating a CDCR benchmark!")


def check_mention_attributes(mention: dict, dataset_name: str):
    required_fields = {COREF_CHAIN, MENTION_NER, MENTION_HEAD_POS, MENTION_HEAD_LEMMA, MENTION_HEAD, MENTION_HEAD_ID, DOC_ID, DOC,
     IS_CONTINIOUS, IS_SINGLETON, MENTION_ID, MENTION_TYPE, MENTION_FULL_TYPE, SCORE, SENT_ID, MENTION_CONTEXT,
     TOKENS_NUMBER_CONTEXT, TOKENS_NUMBER, TOKENS_STR, TOKENS_TEXT, TOPIC_ID, TOPIC, SUBTOPIC_ID, SUBTOPIC, COREF_TYPE,
     DESCRIPTION, CONLL_DOC_KEY}

    diff = required_fields - set(mention)
    if len(diff):
        LOGGER.warning(f'Mentions from the dataset {dataset_name} doesn\'t have the following attributes: '
                       f'{diff}. Reparse the dataset to include this attribute to match the dataset output format!')
    else:
        LOGGER.info(f'Success: Dataset {dataset_name} adheres to the output format.')


def conll_to_newsplease_json(path: str):
    # TODO export conll into a collection of json files
    newsplease_sample = {
      "authors": [],
      "date_download": None,
      "date_modify": None,
      "date_publish": None,
      "description": "",
      "filename": None,
      "image_url": None,
      "language": "en",
      "localpath": None,
      "source_domain": "",
      "text": "",
      "title": "",
      "title_page": None,
      "title_rss": None,
      "url": None
    }
    pass

if __name__ == '__main__':
    form_benchmark()
