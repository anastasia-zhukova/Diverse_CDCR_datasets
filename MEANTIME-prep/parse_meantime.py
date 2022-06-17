import xml.etree.ElementTree as ET
import os
import json
import string
import copy
import re
import pandas as pd
import numpy as np
from datetime import datetime
from nltk.corpus import wordnet
import sys
from nltk import Tree
import spacy
from tqdm import tqdm
from setup import *
from insert_whitespace import append_text
from config import DATA_PATH, TMP_PATH
from logger import LOGGER

path_sample = os.path.join(DATA_PATH, "_sample_doc.json")  # ->root/data/original/_sample_doc.json
MEANTIME_PARSING_FOLDER = os.path.join(DATA_PATH, "MEANTIME-prep")
OUT_PATH = os.path.join(TMP_PATH, "output_data")
CONTEXT_RANGE = 250

nlp = spacy.load('en_core_web_sm')

# opens and loads the newsplease-format out of the json file: _sample_doc.json
with open(path_sample, "r") as file:
    newsplease_format = json.load(file)

import os
source_path = os.path.join(MEANTIME_PARSING_FOLDER, 'MEANTIME')
result_path = os.path.join(MEANTIME_PARSING_FOLDER, 'test_parsing')
result_path2 = os.path.join(MEANTIME_PARSING_FOLDER, 'test_parsing2')
intra = os.path.join(source_path, 'intra-doc_annotation')
intra_cross = os.path.join(source_path, 'intra_cross-doc_annotation')

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
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


def conv_files(path):
    doc_files = {}
    # coref_dics = {}
    entity_mentions = []
    event_mentions = []
    summary_df = pd.DataFrame(columns=[DOC_ID, COREF_CHAIN, DESCRIPTION, MENTION_TYPE, MENTION_FULL_TYPE, TOKENS_STR])
    summary_conversion_df = pd.DataFrame()
    conll_df = pd.DataFrame(columns=[TOPIC_SUBTOPIC, DOC_ID, SENT_ID, TOKEN_ID, TOKEN, REFERENCE])
    final_output_str = ""

    dirs = os.listdir(path)
    cnt = 0
    mark_counter = 0

    for topic_dir in dirs:
        print("Parsing of " + topic_dir + ". Please wait ...")

        # information_df = pd.DataFrame()

        cnt = cnt + 1
        topic_files = os.listdir(os.path.join(path, topic_dir))
        if "_cross-doc_annotation" in path:
            topic_name = str(cnt) + "MEANTIME" + "cross"
            # doc_type = "intra_cross"
        else:
            topic_name = str(cnt) + "MEANTIME"
            # doc_type = "intra"

        coref_dict = {}
        entity_mentions_local = []
        event_mentions_local = []

        for topic_file in topic_files:
            tree = ET.parse(os.path.join(path, topic_dir, topic_file))
            root = tree.getroot()
            title, text, date, url, time, time2, time3 = "", "", "", "", "", "", ""

            info_t_name = topic_file.split(".")[0].split("_")[0]
            t_subt = topic_dir + "/" + info_t_name

            token_dict, mentions, mentions_map = {}, {}, {}

            t_id = -1
            old_sent = 0
            sent_dict = {}

            for elem in root:
                try:
                    if old_sent == int(elem.attrib["sentence"]):
                        t_id += 1
                    else:
                        old_sent = int(elem.attrib["sentence"])
                        t_id = 0
                    token_dict[elem.attrib["t_id"]] = {"text": elem.text, "sent": elem.attrib["sentence"], "id": t_id}

                    if int(elem.attrib["sentence"]) == 0:
                        title, word, no_whitespace = append_text(title, elem.text)
                    elif int(elem.attrib["sentence"]) == 1:
                        title, word, no_whitespace = append_text(date, elem.text)
                    else:
                        title, word, no_whitespace = append_text(text, elem.text)

                    prev_word = ""
                    if t_id >= 0:
                        prev_word = root[t_id - 1].text

                    if elem.tag == "token":
                        conll_df.loc[len(conll_df)] = {
                            TOPIC_SUBTOPIC: t_subt,
                            DOC_ID: topic_name,
                            SENT_ID: int(elem.attrib["sentence"]),
                            TOKEN_ID: t_id,
                            TOKEN: elem.text,
                            # "after": "\"" + append_text(prev_word, elem.text, "space") + "\"",
                            REFERENCE: "-"
                        }

                except KeyError:
                    pass

                if elem.tag == "Markables":
                    for i, subelem in enumerate(elem):
                        tokens = [token.attrib["t_id"] for token in subelem]
                        if len(tokens):
                            sen_id = token_dict[tokens[0]]["sent"]
                            token_nums = [token_dict[t]["id"] for t in tokens]

                            #generate sentence doc with spacy
                            sentence = ""
                            iterationsWithFittingToken = 0
                            for t in root:
                                if t.tag == "token":
                                    if t.attrib["sentence"] == str(sen_id):
                                        if iterationsWithFittingToken != 0 and t.text != "'s" and t.text != "." and t.text != "'" and t.text != "\"" and t.text != "," and t.text != ":" and t.text != ";" and t.text != "'s" and t.text != "?" and t.text != "!": 
                                            sentence = sentence + " "
                                        if t.text != "``" and t.text != "''":     
                                            sentence = sentence + str(t.text)
                                            iterationsWithFittingToken = iterationsWithFittingToken + 1
                            sentence = sentence.replace("  ", " ")
                            doc = nlp(sentence)
                            tokens_str = []

                            first_token = token_dict[tokens[0]]["text"]

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
                                if t["text"] != "``" and t["text"] != "''":
                                    t["text"] = t["text"].replace(",", " ,").replace("'", " '").replace("\'", " '").replace("#", "# ").replace(":", " :").replace("-", " - ").replace("\t", "").replace("(", "( ").replace(")", " )")
                                    if len(t["text"])>2 and t["text"] != "p.m." and t["text"] != "a.m.":    #account for name abbreviations, i.e. "Robert R."
                                        t["text"] = t["text"].replace(".", " .")
                                    indiv_tokens = t["text"].split(" ")
                                    for it in indiv_tokens:
                                        if it != "":
                                            tokens_str.append(it)
                                
                            for to in doc:
                                if len(token_ids_in_doc) == len(tokens):
                                    break
                                for i, ts in enumerate(tokens_str):
                                    if len(token_ids_in_doc) == len(tokens):
                                        break
                                    if (to.text.startswith(ts) or to.text.endswith(ts)):

                                        if len(token_ids_in_doc) > 0:
                                            if i >= 1 and i < len(tokens):
                                                diff_tokens = int(tokens[i])-int(tokens[i-1])   #the difference between tokens in mention in datasets tokenization
                                            else:
                                                diff_tokens = 1

                                            if (to.i not in token_ids_in_doc and abs(to.i - token_ids_in_doc[-1]) <= diff_tokens+1 ):  #account for small differences in tokenization 
                                                token_ids_in_doc.append(to.i)
                                            if abs(to.i - token_ids_in_doc[-1]) > diff_tokens+1 and len(token_ids_in_doc) < len(tokens):
                                                #print("RESET NECCESARY AT ID " + str(to.i))
                                                token_ids_in_doc = [to.i]   #reset
                                        else:
                                            token_ids_in_doc.append(to.i)

                            if len(token_ids_in_doc) == 0:  #if no token has been found, set the condition more broadly
                                for to in doc:
                                    for ts in tokens_str:
                                        if to.text in ts or ts in to.text:
                                            token_ids_in_doc.append(to.i)
                                            break
                            
                            for i in token_ids_in_doc:
                                ancestors_in_mention = 0
                                for a in doc[i].ancestors:
                                    if a.i in token_ids_in_doc:
                                        ancestors_in_mention = ancestors_in_mention + 1
                                        break   #one is enough to make the token unviable as a head
                                if ancestors_in_mention == 0:
                                    #head within the mention
                                    mention_head = doc[i]
                            
                            #this can be used to track inaccuracies with the retokenization (debugging)
                            #if len(token_ids_in_doc) != len(tokens):
                                #print("UNEQUAL LENGTH")
                                #print(sentence)
                                #[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]
                                #print(tokens_str)
                                #print(doc)
                                #print(token_ids_in_doc)
                                #print(tokens)
                                #print("Determined head: " + str(mention_head))

                            mention_head_lemma = mention_head.lemma_
                            mention_head_pos = mention_head.pos_
                            
                            mention_ner = mention_head.ent_type_
                            if mention_ner == "":
                                mention_ner = "O"

                            #get the context
                            tokens_int = [int(x) for x in tokens]
                            context_min_id , context_max_id = [0 if int(min(tokens_int)) - CONTEXT_RANGE < 0 else int(min(tokens_int)) - CONTEXT_RANGE, len(token_dict)-1 if int(max(tokens_int)) + CONTEXT_RANGE > len(token_dict) else int(max(tokens_int)) + CONTEXT_RANGE]
                            
                            mention_context_str = []
                            for t in root:
                                if t.tag == "token" and int(t.attrib["t_id"]) >= context_min_id and int(t.attrib["t_id"]) <= context_max_id:
                                    mention_context_str.append(t.text)

                            mentions[subelem.attrib["m_id"]] = {"type": subelem.tag,
                                                                "text": " ".join(
                                                                    [token_dict[t]["text"] for t in tokens]),
                                                                "sent_doc": doc,
                                                                MENTION_NER: mention_ner,
                                                                MENTION_HEAD_POS: mention_head_pos,
                                                                MENTION_HEAD_LEMMA: mention_head_lemma,
                                                                MENTION_HEAD: mention_head.text,
                                                                MENTION_HEAD_ID: mention_head.i,
                                                                TOKENS_NUMBER: token_nums,
                                                                "token_doc_numbers": token_ids_in_doc,
                                                                DOC_ID: topic_file.split(".")[0],
                                                                SENT_ID: int(sen_id),
                                                                "mention_context": mention_context_str,
                                                                TOPIC_SUBTOPIC: t_subt}

                        else:
                            try:
                                if subelem.attrib["instance_id"] not in coref_dict:
                                    coref_dict[subelem.attrib["instance_id"]] = {
                                        "descr": subelem.attrib["TAG_DESCRIPTOR"]}
                                # m_id points to the target
                            except KeyError:
                                pass
                                # unknown_tag = subelem.tag
                                # coref_dict[unknown_tag + subelem.attrib["m_id"]] = {"descr": subelem.attrib["TAG_DESCRIPTOR"]}

                if elem.tag == "Relations":
                    mentions_map = {m: False for m in list(mentions)}
                    for i, subelem in enumerate(elem):
                        for j, subsubelm in enumerate(subelem):
                            if subsubelm.tag == "target":
                                for prevelem in root:
                                    if prevelem.tag == "Markables":
                                        for k, prevsubelem in enumerate(prevelem):
                                            if prevsubelem.get("instance_id") is not None:
                                                if subsubelm.attrib["m_id"] == prevsubelem.attrib["m_id"]:
                                                    tmp_instance_id = prevsubelem.attrib["instance_id"]
                                            else:
                                                tmp_instance_id = "None"

                        try:
                            if "r_id" not in coref_dict[tmp_instance_id]:
                                coref_dict[tmp_instance_id].update({
                                    "r_id": subelem.attrib["r_id"],
                                    "coref_type": subelem.tag,
                                    "mentions": [mentions[m.attrib["m_id"]] for m in subelem if
                                                 m.tag == "source"]
                                })
                            else:
                                coref_dict[tmp_instance_id]["mentions"].extend(
                                    [mentions[m.attrib["m_id"]] for m in subelem if
                                     m.tag == "source"])
                        except KeyError:
                            pass
                        for m in subelem:
                            mentions_map[m.attrib["m_id"]] = True

            newsplease_custom = copy.copy(newsplease_format)

            newsplease_custom["title"] = title
            newsplease_custom["date_publish"] = None

            # if len(text):
            #     text = text if text[-1] != "," else text[:-1] + "."
            newsplease_custom["filename"] = topic_file
            newsplease_custom["text"] = text
            newsplease_custom["source_domain"] = topic_file.split(".")[0]
            if newsplease_custom["title"][-1] not in string.punctuation:
                newsplease_custom["title"] += "."

            doc_files[topic_file.split(".")[0]] = newsplease_custom
            if topic_name not in os.listdir(result_path):
                os.mkdir(os.path.join(result_path, topic_name))

            with open(os.path.join(result_path, topic_name, newsplease_custom["source_domain"] + ".json"),
                      "w") as file:
                json.dump(newsplease_custom, file)
                print(f'Saved {topic_name}/{newsplease_custom["source_domain"]}')
        # coref_dics[topic_dir] = coref_dict

        for chain_index, (chain_id, chain_vals) in enumerate(coref_dict.items()):
            if chain_vals.get("mentions") is not None:
                for m in chain_vals["mentions"]:

                    sent_id = m["sent_id"]

                    token_numbers = [int(t) for t in m[TOKENS_NUMBER]]
                    mention_id = m["doc_id"] + "_" + str(chain_id) + "_" + str(m["sent_id"]) + "_" + str(
                        m[TOKENS_NUMBER][0])
                    mention = { COREF_CHAIN: chain_id,
                                MENTION_NER: m["mention_ner"],
                                MENTION_HEAD_POS: m["mention_head_pos"],
                                MENTION_HEAD_LEMMA: m["mention_head_lemma"],
                                MENTION_HEAD: m["mention_head"],
                                MENTION_HEAD_ID: m["mention_head_id"],
                                DOC_ID_FULL: m["doc_id"],
                                IS_CONTINIOUS: True if token_numbers == list(
                                    range(token_numbers[0], token_numbers[-1] + 1))
                                else False,
                                IS_SINGLETON: len(chain_vals["mentions"]) == 1,
                                MENTION_ID: mention_id,
                                MENTION_TYPE: chain_id[:3],
                                MENTION_FULL_TYPE: meantime_types[chain_id[:3]],
                                SCORE: -1.0,
                                SENT_ID: sent_id,
                                TOKENS_NUMBER: token_numbers,
                                TOKENS_STR: m["text"],
                                TOPIC_SUBTOPIC: m[TOPIC_SUBTOPIC],
                                COREF_TYPE: chain_vals["coref_type"],
                                DESCRIPTION: chain_vals["descr"]
                               }
                    if "EVENT" in m["type"]:
                        event_mentions_local.append(mention)
                    else:
                        entity_mentions_local.append(mention)
                    summary_df.loc[len(summary_df)] = {
                        DOC_ID: m["doc_id"],
                        COREF_CHAIN: chain_id,
                        DESCRIPTION: chain_vals["descr"],
                        MENTION_TYPE: chain_id[:3],
                        MENTION_FULL_TYPE: m["type"],
                        TOKENS_STR: m["text"]
                    }

        conll_df = conll_df.reset_index(drop=True)

        print(conll_df.head(70))

        # create a conll string from the conll_df
        LOGGER.info("Generating conll string...")
        for i, row in tqdm(conll_df.iterrows(), total=conll_df.shape[0]):
            reference_str = "-"

            for mention in [m for m in event_mentions_local + entity_mentions_local]:
                if mention[TOPIC_SUBTOPIC] == row[TOPIC_SUBTOPIC] and mention[SENT_ID] == row[SENT_ID] and row[TOKEN_ID] in mention[TOKENS_NUMBER]:
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

            if row[DOC_ID] == topic_name:  # do not overwrite conll rows of previous topic iterations
                conll_df.at[i, REFERENCE] = reference_str

        for i, row in conll_df.iterrows():  # remove the leading characters if necessary (left from initialization)
            if row[REFERENCE].startswith("-| "):
                conll_df.at[i, REFERENCE] = row[REFERENCE][3:]

        print(conll_df.head(20))

        annot_path = os.path.join(result_path, topic_name, "annotation",
                                  "original")  # ->root/data/MEANTIME-prep/test_parsing/topicName/annotation/original
        if topic_name not in os.listdir(os.path.join(result_path)):
            os.mkdir(os.path.join(result_path, topic_name))

        if "annotation" not in os.listdir(os.path.join(result_path, topic_name)):
            os.mkdir(os.path.join(result_path, topic_name, "annotation"))
            os.mkdir(annot_path)

        with open(os.path.join(annot_path, "entity_mentions_" + topic_name + ".json"), "w") as file:
            json.dump(entity_mentions_local, file)

        with open(os.path.join(annot_path, "event_mentions_" + topic_name + ".json"), "w") as file:
            json.dump(event_mentions_local, file)

        entity_mentions.extend(entity_mentions_local)
        event_mentions.extend(event_mentions_local)

        conll_topic_df = conll_df[conll_df[TOPIC_SUBTOPIC].str.contains(f'{topic_name}/')].drop(columns=[DOC_ID])

        outputdoc_str = ""
        for (topic_local), topic_df in conll_topic_df.groupby(by=[TOPIC_SUBTOPIC]):
            outputdoc_str += f'#begin document ({topic_local}); part 000\n'

            for (sent_id_local), sent_df in topic_df.groupby(by=[SENT_ID], sort=[SENT_ID]):
                np.savetxt(os.path.join(MEANTIME_PARSING_FOLDER, "tmp.txt"), sent_df.values, fmt='%s', delimiter="\t",
                           encoding="utf-8")
                with open(os.path.join(MEANTIME_PARSING_FOLDER, "tmp.txt"), "r", encoding="utf8") as file:
                    saved_lines = file.read()
                outputdoc_str += saved_lines + "\n"

            outputdoc_str += "#end document\n"
        final_output_str += outputdoc_str

        # Check if the brackets ( ) are correct
        LOGGER.info("Checking equal brackets in conll for " + str(
            topic_name) + " (if unequal, the result may be incorrect):")
        try:
            brackets_1 = 0
            brackets_2 = 0
            for i, row in conll_df.iterrows():  # only count brackets in reference column (exclude token text)
                brackets_1 += str(row[REFERENCE]).count("(")
                brackets_2 += str(row[REFERENCE]).count(")")
            LOGGER.info("Amount of mentions in this topic: " + str(len(event_mentions_local + entity_mentions_local)))
            LOGGER.info("Total mentions parsed (all topics): " + str(len(event_mentions + entity_mentions)))
            LOGGER.info("brackets '(' , ')' : " + str(brackets_1) + " , " + str(brackets_2))
            assert brackets_1 == brackets_2
        except AssertionError:
            LOGGER.warning(
                f'Number of opening and closing brackets in conll does not match! topic: ' + str(topic_name))
            conll_df.to_csv(os.path.join(OUT_PATH, CONLL_CSV))
            with open(os.path.join(annot_path, f'{topic_name}.conll'), "w", encoding='utf-8') as file:
                file.write(outputdoc_str)
            sys.exit()

        conll_df.to_csv(os.path.join(OUT_PATH, CONLL_CSV))

        with open(os.path.join(annot_path, f'{topic_name}.conll'), "w", encoding='utf-8') as file:
            file.write(outputdoc_str)

        summary_conversion_df = summary_conversion_df.append(pd.DataFrame({
            "files": len(topic_files),
            "tokens": len(conll_topic_df),
            "chains": len(coref_dict),
            "event_mentions": len(event_mentions_local),
            "entity_mentions": len(entity_mentions_local),
            "singletons": sum([v["is_singleton"] for v in event_mentions_local]) + sum(
                [v["is_singleton"] for v in entity_mentions_local])
        }, index=[topic_name]))

    with open(os.path.join(OUT_PATH, "conll_as_json" + ".json"), "w", encoding='utf-8') as file:
        json.dump(conll_df.to_dict('records'), file)

    with open(os.path.join(OUT_PATH, 'meantime.conll'), "w", encoding='utf-8') as file:
        file.write(final_output_str)

    with open(os.path.join(OUT_PATH, "entity_mentions" + ".json"), "w", encoding='utf-8') as file:
        json.dump(entity_mentions, file)

    with open(os.path.join(OUT_PATH, "event_mentions" + ".json"), "w", encoding='utf-8') as file:
        json.dump(event_mentions, file)

    summary_df.to_csv(os.path.join(OUT_PATH, "all_mentions.csv"))
    #summary_conversion_df.to_csv(os.path.join(result_path, now.strftime("%Y-%m-%d_%H-%M") + "_" + "dataset_summary.csv"))

    print('Parsing of MEANTIME annotation done!')
    return 'Parsing of MEANTIME annotation done!'


if __name__ == '__main__':

    conv_files(intra_cross)

    # print('Please enter the number of the set, you want to convert:\n'
    #       '   1 MEANTIME intra document annotation\n'
    #       '   2 MEANTIME cross-document annotation\n'
    #       '   3 both')
    #
    #
    # def choose_input():
    #     setnumber = input()
    #     if setnumber == "1":
    #         c_format = "\"MEANTIME intra document annotation\""
    #         print(conv_files(intra))
    #         return c_format
    #     elif setnumber == "2":
    #         c_format = "\"MEANTIME cross-document annotation\""
    #         print(conv_files(intra_cross))
    #         return c_format
    #     elif setnumber == "3":
    #         c_format = "\"MEANTIME intra and intra cross-document annotations\""
    #         print(conv_files(intra))
    #         print(conv_files(intra_cross))
    #         return c_format
    #     else:
    #         print("Please choose one of the 3 numbers!")
    #         return choose_input()


    # co_format = choose_input()

    # print("\nConversion of {0} from xml to newsplease format and to annotations in a json file is "
    #       "done. \n\nFiles are saved to {1}. \nCopy the topics on which you want to execute Newsalyze to "
    #       "{2}.".format(co_format, result_path, DATA_PATH))
