import xml.etree.ElementTree as ET
import os
import json
import string
import spacy
import copy
import re
import pandas as pd
import numpy as np
from insert_whitespace import append_text
from nltk import Tree
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from setup import *
from logger import LOGGER

ECB_PARSING_FOLDER = os.path.join(os.getcwd())
ECBPLUS_FILE = "ecbplus.xml"
ECB_FILE = "ecb.xml"
IS_TEXT, TEXT = "is_text", TEXT

source_path = os.path.join(ECB_PARSING_FOLDER, ECBPLUS_FOLDER_NAME)
result_path = os.path.join(ECB_PARSING_FOLDER, OUTPUT_FOLDER_NAME, "test_parsing")
out_path = os.path.join(ECB_PARSING_FOLDER, OUTPUT_FOLDER_NAME)
path_sample = os.path.join(os.getcwd(), "..", SAMPLE_DOC_JSON)

nlp = spacy.load('en_core_web_sm')

with open(path_sample, "r") as file:
    newsplease_format = json.load(file)


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


def convert_files(topic_number_to_convert=3, check_with_list=True):
    doc_files = {}
    coref_dics = {}

    selected_topics = os.listdir(source_path)[:topic_number_to_convert]

    conll_df = pd.DataFrame()
    final_output_str = ""
    entity_mentions = []
    event_mentions = []
    topic_names = []

    for topic_folder in selected_topics:
        LOGGER.info(f'Converting topic {topic_folder}')
        diff_folders = {ECB_FILE: [], ECBPLUS_FILE: []}

        # assign the different folders according to the topics in the variable "diff_folders"
        for topic_file in os.listdir(os.path.join(source_path, topic_folder)):
            if ECBPLUS_FILE in topic_file:
                diff_folders[ECBPLUS_FILE].append(topic_file)
            else:
                diff_folders[ECB_FILE].append(topic_file)

        for annot_folders in list(diff_folders.values()):
            t_number = annot_folders[0].split(".")[0].split("_")[0]
            t_name = re.search(r'[a-z]+', annot_folders[0].split(".")[0])[0]
            topic_name = t_number + t_name
            topic_names.append(topic_name)
            coref_dict = {}
            doc_sent_map = {}

            # for every themed-file in "commentated files"
            for topic_file in tqdm(annot_folders):
                mention_counter_got, mentions_counter_found = 0, 0
                info_t_name = re.search(r'[\d+]+', topic_file.split(".")[0].split("_")[1])[0]
                t_subt = f'{topic_folder}/{topic_name}/{info_t_name}'

                # import the XML-Datei topic_file
                tree = ET.parse(os.path.join(source_path, topic_folder, topic_file))
                root = tree.getroot()

                title, text, url, time, time2, time3 = "", "", "", "", "", ""

                token_dict, mentions, mentions_map = {}, {}, {}

                t_id = -1
                old_sent = -1
                sent_dict = {}

                for elem in root:
                    try:
                        # increase t_id value by 1 if the sentence value in the xml element ''equals the value of old_sent
                        if old_sent == int(elem.attrib[SENTENCE]):
                            t_id += 1
                            # else set old_sent to the value of sentence and t_id to 0
                        else:
                            old_sent = int(elem.attrib[SENTENCE])
                            t_id = 0

                        # fill the token-dictionary with fitting attributes
                        token_dict[elem.attrib[T_ID]] = {TEXT: elem.text, SENT: elem.attrib[SENTENCE], ID: t_id}

                        prev_word = ""
                        if t_id >= 0:
                            prev_word = root[t_id-1].text

                        if elem.tag == TOKEN:
                            _, word, space = append_text(prev_word, elem.text)
                            conll_df = conll_df.append(pd.DataFrame({
                                TOPIC_SUBTOPIC: t_subt,
                                DOC_ID: topic_name,
                                SENT_ID: int(token_dict[elem.attrib[T_ID]][SENT]),
                                TOKEN_ID: int(t_id),
                                TOKEN: word,
                                # "after": "\"" + space + "\"",
                                REFERENCE: "-"
                            }, index=[elem.attrib[T_ID]]))

                        if ECB_FILE in topic_file:
                            #set the title-attribute of all words of the first sentence together und the text-attribute of the rest
                            if int(elem.attrib[SENTENCE]) == 0:
                                title, _, _ = append_text(title, elem.text)
                            else:
                                text, _, _ = append_text(text, elem.text)

                        if ECBPLUS_FILE in topic_file:
                            #add a string with the complete sentence to the sentence-dictionary for every different sentence
                            new_text, _, _ = append_text(sent_dict.get(int(elem.attrib[SENTENCE]), ""), elem.text)
                            sent_dict[int(elem.attrib[SENTENCE])] = new_text

                    except KeyError as e:
                        pass

                    if elem.tag == "Markables":

                        for i, subelem in enumerate(elem):
                            tokens = [token.attrib[T_ID] for token in subelem]
                            mention_text = ""
                            for t in tokens:
                                mention_text, _, _ = append_text(mention_text, token_dict[t][TEXT])
                            #if "tokens" has values -> fill the "mention" dict with the value of the correspondant m_id
                            if len(tokens):
                                mention_counter_got += 1
                                sent_id = int(token_dict[tokens[0]][SENT])

                                #generate sentence doc with spacy
                                sentence = ""
                                iterationsWithFittingToken = 0
                                for t in root:
                                    if t.tag == TOKEN:
                                        if t.attrib[SENTENCE] == str(sent_id):
                                            if iterationsWithFittingToken != 0 and t.text != "'s" and t.text != "." and \
                                                    t.text != "'" and t.text != "\"" and t.text != "," and t.text != ":" \
                                                    and t.text != ";" and t.text != "'s" and t.text != "?" and t\
                                                    .text != "!":
                                                sentence = sentence + " "
                                            if t.text != "``" and t.text != "''":
                                                sentence = sentence + str(t.text)
                                                iterationsWithFittingToken = iterationsWithFittingToken + 1
                                sentence = sentence.replace("  ", " ")
                                doc = nlp(sentence)
                                tokens_str = []

                                first_token = token_dict[tokens[0]][TEXT]

                                if first_token[0] == "'":
                                    first_token = first_token[1:]
                                if first_token[-1] == "'":
                                    first_token = first_token[:-1]
                                if first_token[-1] == '"':
                                    first_token = first_token[:-1]
                                if first_token[0] == '"':
                                    first_token = first_token[1:]

                                first_token = first_token.replace(",", " ,").replace("'", " '").replace("\'", " '").replace("#", "# ").replace(":", " :").replace("-", " - ").replace("\t", "").replace("(", "( ").replace(")", " )")
                                if len(first_token) > 2:
                                    first_token = first_token.replace(".", " .")

                                indiv_tokens = first_token.split(" ")
                                for it in indiv_tokens:
                                    if it != "":
                                        tokens_str.append(it)

                                token_ids_in_doc = []

                                #Redetermine the ids of the token within the sentence because it is not guaranteed that "numbers" from the ECB+ XML does match the spacy interpretation of tokenization
                                for t_id in tokens[1:]:
                                    t = token_dict[t_id]
                                    if t[TEXT] != "``" and t[TEXT] != "''":
                                        t[TEXT] = t[TEXT].replace(",", " ,").replace("'", " '").replace("\'", " '").replace("#", "# ").replace(":", " :").replace("-", " - ").replace("\t", "").replace("(", "( ").replace(")", " )")
                                        # account for name abbreviations, i.e. "Robert R."
                                        if len(t[TEXT])>2 and t[TEXT] != "p.m." and t[TEXT] != "a.m.":
                                            t[TEXT] = t[TEXT].replace(".", " .")
                                        indiv_tokens = t[TEXT].split(" ")
                                        for it in indiv_tokens:
                                            if it != "":
                                                tokens_str.append(it)

                                for to in doc:
                                    if len(token_ids_in_doc) == len(tokens):
                                        break
                                    for ts in tokens_str:
                                        if len(token_ids_in_doc) == len(tokens):
                                            break
                                        if (to.text.startswith(ts) or to.text.endswith(ts)):

                                            if len(token_ids_in_doc) > 0:
                                                if (to.i not in token_ids_in_doc and abs(to.i - token_ids_in_doc[-1]) <= 2):
                                                    # account for small differences in tokenization
                                                    token_ids_in_doc.append(to.i)
                                                if abs(to.i - token_ids_in_doc[-1]) > 2 and len(token_ids_in_doc) < len(tokens):
                                                    token_ids_in_doc = [to.i]   #reset
                                            else:
                                                token_ids_in_doc.append(to.i)

                                #determine the head
                                for i in token_ids_in_doc:
                                    ancestors_in_mention = 0
                                    for a in doc[i].ancestors:
                                        if a.i in token_ids_in_doc:
                                            ancestors_in_mention = ancestors_in_mention + 1
                                            break   #one is enough to make the token unviable as a head
                                    if ancestors_in_mention == 0:
                                        #head within the mention
                                        mention_head = doc[i]

                                mention_head_lemma = mention_head.lemma_
                                mention_head_pos = mention_head.pos_

                                mention_ner = mention_head.ent_type_
                                if mention_ner == "":
                                    mention_ner = "O"

                                #get the context
                                tokens_int = [int(x) for x in tokens]
                                context_min_id , context_max_id = [0 if int(min(tokens_int)) - CONTEXT_RANGE < 0 else
                                                                   int(min(tokens_int)) - CONTEXT_RANGE, len(token_dict)-1
                                                                    if int(max(tokens_int)) + CONTEXT_RANGE > len(token_dict)
                                                                    else int(max(tokens_int)) + CONTEXT_RANGE]

                                mention_context_str = []
                                for t in root:
                                    if t.tag == TOKEN and int(t.attrib[T_ID]) >= context_min_id and int(t.attrib[T_ID]) <= context_max_id:
                                        mention_context_str.append(t.text)
                                mentions[subelem.attrib[M_ID]] = {MENTION_TYPE: subelem.tag,
                                                                    MENTION_FULL_TYPE: subelem.tag,
                                                                    TOKENS_STR: mention_text.strip(),
                                                                    "sent_doc": doc,
                                                                    MENTION_NER: mention_ner,
                                                                    MENTION_HEAD_POS: mention_head_pos,
                                                                    MENTION_HEAD_LEMMA: mention_head_lemma,
                                                                    MENTION_HEAD: mention_head.text,
                                                                    MENTION_HEAD_ID: mention_head.i,
                                                                    TOKENS_NUMBER: [int(token_dict[t][ID]) for t in tokens],
                                                                    TOKENS_TEXT: [token_dict[t][TEXT] for t in tokens],
                                                                    "token_doc_numbers": token_ids_in_doc,
                                                                    DOC_ID: topic_file.split(".")[0],
                                                                    SENT_ID: sent_id,
                                                                    MENTION_CONTEXT: mention_context_str,
                                                                    TOPIC: t_subt}
                                mentions_counter_found += 1
                            else:
                                try:
                                    # if there are no t_ids (token is empty) and the "instance_id" is not in coref-dict:
                                    if subelem.attrib["instance_id"] not in coref_dict:
                                        # save in coref_dict to the instance_id
                                        coref_dict[subelem.attrib["instance_id"]] = {
                                            DESCRIPTION: subelem.attrib["TAG_DESCRIPTOR"]}
                                    # m_id points to the target
                                except KeyError as e:
                                    pass

                    if elem.tag == "Relations":
                        # for every false create a false-value in "mentions_map"
                        mentions_map = {m: False for m in list(mentions)}
                        for i, subelem in enumerate(elem):
                            try:
                                if "r_id" not in coref_dict[subelem.attrib["note"]]:
                                    # update the coref_dict-element and add "r_id", "coref_type" and MENTIONS
                                    # just elements of the subelements with tag "sourse" are added to mentions
                                    coref_dict[subelem.attrib["note"]].update({
                                        "r_id": subelem.attrib["r_id"],
                                        COREF_TYPE: subelem.tag,
                                        MENTIONS: [mentions[m.attrib[M_ID]] for m in subelem if m.tag == "source"]
                                    })
                                else:
                                    coref_dict[subelem.attrib["note"]][MENTIONS].extend(
                                        [mentions[m.attrib[M_ID]] for m in subelem if m.tag == "source"])
                            except KeyError:
                                pass

                            for m in subelem:
                                mentions_map[m.attrib[M_ID]] = True

                        for i, (m_id, used) in enumerate(mentions_map.items()):
                            if used:
                                continue

                            m = mentions[m_id]
                            if "Singleton_" + m[MENTION_TYPE][:4] + "_" + str(m_id) + "_" + m[DOC_ID] not in coref_dict:
                                coref_dict["Singleton_" + m[MENTION_TYPE][:4] + "_" + str(m_id) + "_" + m[DOC_ID]] = {
                                    "r_id": str(10000 + i),
                                    COREF_TYPE: "Singleton",
                                    MENTIONS: [m],
                                    DESCRIPTION: ""
                                }
                            else:
                                coref_dict["Singleton_" + m["type"][:4] + "_" + str(m_id) + "_" + m[DOC_ID]].update(
                                    {
                                        "r_id": str(10000 + i),
                                        COREF_TYPE: "Singleton",
                                        MENTIONS: [m],
                                        DESCRIPTION: ""
                                    })
                    a = 1

                newsplease_custom = copy.copy(newsplease_format)

                if ECBPLUS_FILE in topic_file:
                    sent_df = pd.DataFrame(columns=[IS_TEXT, TEXT])
                    for sent_key, text in sent_dict.items():
                        #expand the dataframe
                        # IS_TEXT is 0 (if the (number of number-signs in Text)/(Textlength) >= 0.1 or Counter(sent_key) = 0. else 1
                        # TEXT to text
                        # index to sent_key
                        sent_df = sent_df.append(pd.DataFrame({
                            IS_TEXT: 0 if len(re.sub(r'[\D]+', "", text)) / len(text) >= 0.1 or sent_key == 0 else 1,
                            TEXT: text
                        }, index=[sent_key]))
                    doc_sent_map[topic_file.split(".")[0]] = sent_df

                    small_df = sent_df[sent_df[IS_TEXT] == 0]
                    # create/ add values to DataFrame news_please_custom :
                    # "date_published" as " " + all other TEXT- values of the Dataframe w.o. the first. "" if not there
                    # "title" with first value of TEXT, of wich IS_TEXT = 1 (<= 10% Zahlen)
                    # set Variable TEXT to " "+ rest of values (w.o. 1st one) that consist <= 10% of the numbers

                    # URL
                    url_space_text = ""
                    if len(small_df) > 0:
                        url_space_list = str(list(small_df[TEXT].values)[0]).split(" ")
                        if len(url_space_list) > 1:
                            for c in url_space_list:
                                url_space_text, _, space = append_text(url_space_text, c)

                    newsplease_custom["url"] = url_space_text
                    newsplease_custom["date_publish"] = " ".join(list(small_df[TEXT].values)[1:]) if len(
                        small_df) > 1 else ""
                    newsplease_custom["title"] = list(sent_df[sent_df[IS_TEXT] == 1][TEXT].values)[0]
                    text = " ".join(list(sent_df[sent_df[IS_TEXT] == 1][TEXT].values)[1:])
                else:
                    newsplease_custom["title"] = title
                    newsplease_custom["date_publish"] = None

                if len(text):
                    text = text if text[-1] != "," else text[:-1] + "."
                # create/ add values toDataFrame news_please_custom:
                # TEXT = text
                # "source_domain" = name of file
                newsplease_custom[TEXT] = text
                newsplease_custom[SOURCE_DOMAIN] = topic_file.split(".")[0]
                if newsplease_custom[TITLE][-1] not in string.punctuation:
                    newsplease_custom["title"] += "."

                doc_files[topic_file.split(".")[0]] = newsplease_custom
                if topic_name not in os.listdir(result_path):
                    os.mkdir(os.path.join(result_path, topic_name))

                with open(os.path.join(result_path, topic_name, newsplease_custom[SOURCE_DOMAIN] + ".json"),
                          "w") as file:
                    json.dump(newsplease_custom, file)

                # LOGGER.info(f'GOT:   {mention_counter_got}')
                # LOGGER.info(f'FOUND: {mentions_counter_found}\n')
            coref_dics[topic_folder] = coref_dict

            entity_mentions_local = []
            event_mentions_local = []
            mentions_local = []

            for chain_id, chain_vals in coref_dict.items():
                not_unique_heads = []

                for m in chain_vals[MENTIONS]:

                    if ECBPLUS_FILE.split(".")[0] not in m[DOC_ID]:
                        sent_id = int(m[SENT_ID])
                    else:
                        df = doc_sent_map[m[DOC_ID]]
                        # sent_id = np.sum([df.iloc[:list(df.index).index(int(m[SENT_ID])) + 1][IS_TEXT].values])
                        sent_id = int(m[SENT_ID])
                    # converts TOKENS_NUMBER of m to an array TOKENS_NUMBER with int values

                    # create variable "mention_id" out of doc_id+_+chain_id+_+sent_id+_+first value of TOKENS_NUMBER of "m"
                    token_numbers = [int(t) for t in m[TOKENS_NUMBER]]
                    mention_id = m[DOC_ID] + "_" + str(chain_id) + "_" + str(m[SENT_ID]) + "_" + str(
                        m[TOKENS_NUMBER][0])

                    not_unique_heads.append(m[MENTION_HEAD_LEMMA])

                    #create the dict. "mention" with all corresponding values
                    mention = {COREF_CHAIN: chain_id,
                                MENTION_NER: m[MENTION_NER],
                                MENTION_HEAD_POS: m[MENTION_HEAD_POS],
                                MENTION_HEAD_LEMMA: m[MENTION_HEAD_LEMMA],
                                MENTION_HEAD: m[MENTION_HEAD],
                                MENTION_HEAD_ID: m[MENTION_HEAD_ID],
                                DOC_ID: m[DOC_ID],
                                # DOC_ID_FULL: m[DOC_ID],
                                IS_CONTINIOUS: bool(token_numbers == list(range(token_numbers[0], token_numbers[-1] + 1))),
                                IS_SINGLETON: bool(len(chain_vals[MENTIONS]) == 1),
                                MENTION_ID: mention_id,
                                MENTION_TYPE: m[MENTION_TYPE][:3],
                                MENTION_FULL_TYPE: m[MENTION_TYPE],
                                SCORE: -1.0,
                                SENT_ID: sent_id,
                                MENTION_CONTEXT: m[MENTION_CONTEXT],
                                # now the token numbers based on spacy tokenization, not ecb+ tokenization
                                TOKENS_NUMBER: m["token_doc_numbers"],
                                TOKENS_STR: m[TOKENS_STR],
                                TOKENS_TEXT: m[TOKENS_TEXT],
                                TOPIC_ID: int(t_number),
                                TOPIC: topic_name,
                                # COREF_TYPE: chain_vals[COREF_TYPE],
                                COREF_TYPE: STRICT,
                                DESCRIPTION: chain_vals[DESCRIPTION]
                                }

                    # if the first two entries of chain_id are "ACT" or "NEG", add the "mention" to the array "event_mentions_local"
                    if chain_id[:3] in ["ACT", "NEG"]:
                        event_mentions_local.append(mention)
                    # else add the "mention" to the array "event_mentions_local" and add the following values to the DF "summary_df"
                    else:
                        entity_mentions_local.append(mention)

                    if not mention[IS_SINGLETON]:
                        mentions_local.append(mention)

                    t_subt = m[TOPIC]
                    mark_counter = chain_id
                    # todo: once with (number) as 1st entry the placeholder stays as "- ," before the (number).
                    if token_numbers[0] == token_numbers[len(token_numbers) - 1]:
                        if str(conll_df.loc[(conll_df[SENT_ID] == sent_id) & (
                                conll_df[TOKEN_ID] == token_numbers[0]) & (conll_df[TOPIC_SUBTOPIC] == t_subt), REFERENCE].values[
                                   0]) == "-":

                            conll_df.loc[(conll_df[SENT_ID] == sent_id) & (
                                    conll_df[TOKEN_ID] == token_numbers[0]) & (conll_df[TOPIC_SUBTOPIC] == t_subt), REFERENCE] = '(' + str(
                                mark_counter) + ')'

                        else:
                            conll_df.loc[(conll_df[SENT_ID] == sent_id) & (
                                    conll_df[TOKEN_ID] == token_numbers[0]) & (conll_df[TOPIC_SUBTOPIC] == t_subt), REFERENCE] = str(
                                conll_df.loc[(conll_df[SENT_ID] == sent_id) & (
                                        conll_df[TOKEN_ID] == token_numbers[0] & (
                                            conll_df[TOPIC_SUBTOPIC] == t_subt)), REFERENCE].values[
                                    0]) + '| (' + str(mark_counter) + ')'

                    else:
                        # if there stands "-" at the location with the df with the right token and sentence and topic-file, hence first entry, then for the 1st token of the arrays:
                        if str(conll_df.loc[(conll_df[SENT_ID] == sent_id) & (
                                conll_df[TOKEN_ID] == token_numbers[0]) & (conll_df[TOPIC_SUBTOPIC] == t_subt), REFERENCE].values[
                                   0]) == "-":
                            conll_df.loc[(conll_df[SENT_ID] == sent_id) & (
                                    conll_df[TOKEN_ID] == token_numbers[0]) & (conll_df[TOPIC_SUBTOPIC] == t_subt), REFERENCE] = '(' + str(
                                mark_counter)

                        else:
                            conll_df.loc[(conll_df[SENT_ID] == sent_id) & (
                                    conll_df[TOKEN_ID] == token_numbers[0]) & (conll_df[TOPIC_SUBTOPIC] == t_subt), REFERENCE] = str(
                                conll_df.loc[(conll_df[SENT_ID] == sent_id) & (
                                        conll_df[TOKEN_ID] == token_numbers[0]) & (conll_df[TOPIC_SUBTOPIC] == t_subt), REFERENCE].values[
                                    0]) + '| (' + str(mark_counter)

                        # if there stands "-" at the location with the df with the right token and sentence and topic-file, hence first entry, then for the last token of the arrays:
                        if str(conll_df.loc[(conll_df[SENT_ID] == sent_id) & (
                                conll_df[TOKEN_ID] == token_numbers[
                            len(token_numbers) - 1]) & (conll_df[
                                                            TOPIC_SUBTOPIC] == t_subt), REFERENCE].values[0]) == "-":
                            conll_df.loc[(conll_df[SENT_ID] == sent_id) & (
                                    conll_df[TOKEN_ID] == token_numbers[
                                len(token_numbers) - 1]) & (conll_df[TOPIC_SUBTOPIC] == t_subt), REFERENCE] = str(
                                mark_counter) + ')'
                        else:
                            conll_df.loc[(conll_df[SENT_ID] == sent_id) & (
                                        conll_df[TOKEN_ID] == token_numbers[len(token_numbers) - 1]) &
                                               (conll_df[TOPIC_SUBTOPIC] == t_subt), REFERENCE] = str(
                                conll_df.loc[(conll_df[SENT_ID] == sent_id)
                                                   & (conll_df[TOKEN_ID] == token_numbers[
                                    len(token_numbers) - 1]) &
                                                   (conll_df[TOPIC_SUBTOPIC] == t_subt), REFERENCE].values[
                                    0]) + '| ' + str(mark_counter) + ')'

            # create annot_path and file-structure (if not already) for the output of the annotations
            annot_path = os.path.join(result_path, topic_name, "annotation", "original")
            # ->root/data/ECBplus-prep/test_parsing/topicName/annotation/original
            if topic_name not in os.listdir(os.path.join(result_path)):
                os.mkdir(os.path.join(result_path, topic_name))

            if "annotation" not in os.listdir(os.path.join(result_path, topic_name)):
                os.mkdir(os.path.join(result_path, topic_name, "annotation"))
                os.mkdir(annot_path)
            # create the entity-mentions and event-mentions - .json files out of the arrays
            with open(os.path.join(annot_path, f'{ENTITY}_{MENTIONS}_{topic_name}.json'), "w") as file:
                json.dump(entity_mentions_local, file)
            entity_mentions.extend(entity_mentions_local)

            with open(os.path.join(annot_path, f'{EVENT}_{MENTIONS}_{topic_name}.json'), "w") as file:
                json.dump(event_mentions_local, file)
            event_mentions.extend(event_mentions_local)

            conll_topic_df = conll_df[conll_df[TOPIC_SUBTOPIC].str.contains(f'{topic_name}/')].drop(columns=[DOC_ID])

            outputdoc_str = ""
            for (topic_local), topic_df in conll_topic_df.groupby(by=[TOPIC_SUBTOPIC]):
                outputdoc_str += f'#begin document ({topic_local}); part 000\n'

                for (sent_id_local), sent_df in topic_df.groupby(by=[SENT_ID], sort=[SENT_ID]):
                    np.savetxt(os.path.join(ECB_PARSING_FOLDER, "tmp.txt"), sent_df.values, fmt='%s', delimiter="\t",
                               encoding="utf-8")
                    with open(os.path.join(ECB_PARSING_FOLDER, "tmp.txt"), "r", encoding="utf8") as file:
                        saved_lines = file.read()
                    outputdoc_str += saved_lines + "\n"

                outputdoc_str += "#end document\n"
            final_output_str += outputdoc_str

            LOGGER.info("Checking equal brackets in conll (if unequal, the result may be incorrect):")
            try:
                assert final_output_str.count("(") == final_output_str.count(")")
            except AssertionError:
                LOGGER.warning(f'Number of opening and closing brackets in conll does not match! ')

            with open(os.path.join(annot_path, f'{topic_name}.conll'), "w", encoding='utf-8') as file:
                file.write(outputdoc_str)

            #Average number of unique head lemmas within a cluster
            #(excluding singletons for fair comparison)
            for m in mentions_local:
                head_lemmas = [m[MENTION_HEAD_LEMMA]]
                uniques = 1
                for m2 in mentions_local:
                    if m2[MENTION_HEAD_LEMMA] not in head_lemmas and m2[TOKENS_STR] == m[TOKENS_STR] and m2[TOPIC_ID] == m[TOPIC_ID]:
                        head_lemmas.append(m[MENTION_HEAD_LEMMA])
                        uniques = uniques+1
                m["mention_head_lemma_uniques"] = uniques
           

    conll_df.to_csv(os.path.join(out_path, CONLL_CSV))

    with open(os.path.join(out_path, ECB_PLUS.split("-")[0] + '.conll'), "w", encoding='utf-8') as file:
        file.write(final_output_str)

    with open(os.path.join(out_path, MENTIONS_ENTITIES_JSON), "w", encoding='utf-8') as file:
        json.dump(entity_mentions, file)

    with open(os.path.join(out_path, MENTIONS_EVENTS_JSON), "w", encoding='utf-8') as file:
        json.dump(event_mentions, file)

    # create a csv. file out of the mentions summary_df
    df_all_mentions = pd.DataFrame()
    for mention in entity_mentions + event_mentions:
        df_all_mentions = pd.concat([df_all_mentions, pd.DataFrame({
                   attr: str(value) if type(value) == list else value for attr, value in mention.items()
            }, index=[mention[MENTION_ID]])], axis=0)

    df_all_mentions.to_csv(os.path.join(os.getcwd(), OUTPUT_FOLDER_NAME, MENTIONS_ALL_CSV))
    LOGGER.info(f'Done! \nNumber of unique mentions: {len(df_all_mentions)} '
                f'\nNumber of unique chains: {len(set(df_all_mentions[COREF_CHAIN].values))} ')


# main function for the input which topics of the ecb corpus are to be converted
if __name__ == '__main__':
    topic_num = 45
    convert_files(topic_num)
    LOGGER.info("\nConversion of {0} topics from xml to newsplease format and to annotations in a json file is "
          "done. \n\nFiles are saved to {1}. \n.".format(str(topic_num), result_path))
