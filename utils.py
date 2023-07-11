from typing import Tuple, Union, List
import re
from tqdm import tqdm
from setup import  *
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None


def make_save_conll(conll_df: pd.DataFrame, mentions: Union[List, pd.DataFrame], output_folder: str, assign_reference_labels=True) -> pd.DataFrame:
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
            df_all_mentions.to_csv(os.path.join(output_folder, MENTIONS_ALL_CSV))

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
    for (topic_local), topic_df in conll_df.groupby(by=[TOPIC_SUBTOPIC_DOC]):
        outputdoc_str += f'#begin document ({topic_local}); part 000\n'

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
    try:
        for i, row in conll_df.iterrows():  # only count brackets in reference column (exclude token text)
            brackets_1 += str(row[REFERENCE]).count("(")
            brackets_2 += str(row[REFERENCE]).count(")")
        assert brackets_1 == brackets_2

    except AssertionError:
        LOGGER.warning(f'Number of opening and closing brackets in conll does not match!')
        LOGGER.warning(f"brackets '(' , ')' : {str(brackets_1)}, {str(brackets_2)}")

    conll_df.to_csv(os.path.join(output_folder, CONLL_CSV))
    with open(os.path.join(output_folder, 'dataset.conll'), "w", encoding='utf-8') as file:
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
