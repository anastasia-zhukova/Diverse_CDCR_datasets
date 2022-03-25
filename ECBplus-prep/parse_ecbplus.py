from operator import attrgetter
import xml.etree.ElementTree as ET
import os
import json
import string
import spacy
import copy
import re
import pandas as pd
import numpy as np
from datetime import datetime
from config import DATA_PATH, TMP_PATH
from insert_whitespace import append_text

#missing: code, mention_ner, mention_head_pos, mention_head_lemma, mention_head, coref_link (same as code), mention_context
#(mention_hhead_lemma_uniques), (tokens_amount)

ECB_PARSING_FOLDER = os.path.join(DATA_PATH, "ECBplus-prep")
ECBPLUS = "ecbplus.xml"
ECB = "ecb.xml"

source_path = os.path.join(ECB_PARSING_FOLDER, "ECB+")
result_path = os.path.join(ECB_PARSING_FOLDER, "test_parsing")
path_sample = os.path.join(DATA_PATH, "_sample.json")

nlp = spacy.load('en_core_web_sm')

with open(path_sample, "r") as file:
    newsplease_format = json.load(file)


def mean(numbers):
    a = 0
    for n in numbers:
        a = a + n
    return a/len(numbers)


def convert_files(topic_number_to_convert=3, check_with_list=True):
    doc_files = {}
    coref_dics = {}

    # with open(os.path.join(ECB_PARSING_FOLDER, "test_all_events.json"), "r") as file:
    #     selected_articles = json.load(file)

    selected_topics = []

    for i in range(topic_number_to_convert + 1):
        for j in os.listdir(source_path):
            if i == int(j):
                selected_topics.append(str(i))

    summary_df = pd.DataFrame()
    summary_conversion_df = pd.DataFrame()
    conll_df = pd.DataFrame()
    final_output_str = ""
    entity_mentions= []
    event_mentions = []

    for topic_folder in selected_topics:
        print(f'Converting {topic_folder}')
        diff_folders = {ECB: [], ECBPLUS: []}

        # assign the different folders according to the topics in the variable "diff_folders"
        for topic_file in os.listdir(os.path.join(source_path, topic_folder)):
            if ECBPLUS in topic_file:
                diff_folders[ECBPLUS].append(topic_file)
            else:
                diff_folders[ECB].append(topic_file)

        for annot_folders in list(diff_folders.values()):
            t_number = annot_folders[0].split(".")[0].split("_")[0]
            t_name = re.search(r'[a-z]+', annot_folders[0].split(".")[0])[0]
            topic_name = t_number + t_name
            coref_dict = {}
            doc_sent_map = {}
            IS_TEXT, TEXT = "is_text", "text"

            # for every themed-file in "commentated files"
            for topic_file in annot_folders:
                print(f'Converting {topic_file}')
                info_t_name = re.search(r'[\d+]+', topic_file.split(".")[0].split("_")[1])[0]
                t_subt = topic_name + "/" + info_t_name

                # import the XML-Datei topic_file
                tree = ET.parse(os.path.join(source_path, topic_folder, topic_file))
                # if the file isnt in the selected articles of the json files continue
                # if check_with_list and topic_file.split(".")[0] not in selected_articles:
                #     continue
                # gets the root (outermost tag)
                root = tree.getroot()

                title, text, url, time, time2, time3 = "", "", "", "", "", ""

                token_dict, mentions, mentions_map = {}, {}, {}

                t_id = -1
                old_sent = -1
                sent_dict = {}

                for elem in root:
                    try:
                        # increase t_id value by 1 if the sentence value in the xml element ''equals the value of old_sent
                        if old_sent == int(elem.attrib["sentence"]):
                            t_id += 1
                            # else set old_sent to the value of sentence and t_id to 0
                        else:
                            old_sent = int(elem.attrib["sentence"])
                            t_id = 0

                        # fill the token-dictionary with fitting attributes
                        token_dict[elem.attrib["t_id"]] = {"text": elem.text, "sent": elem.attrib["sentence"], "id": t_id}

                        prev_word = ""
                        if t_id >= 0:
                            prev_word = root[t_id-1].text

                        if elem.tag == "token":
                            # info_t_number = re.search('[0-9]+', annot_folders[0].split(".")[0].split("_")[1])
                            _, word, space = append_text(prev_word, elem.text)
                            conll_df = conll_df.append(pd.DataFrame({
                                "topic/subtopic_name": t_subt,
                                "sent_id": int(token_dict[elem.attrib["t_id"]]["sent"]),
                                "token_id": int(t_id),
                                "token": word,
                                # "after": "\"" + space + "\"",
                                "coref": "-"
                            }, index=[elem.attrib["t_id"]]))

                        if ECB in topic_file:
                            #set the title-attribute of all words of the first sentence together und the text-attribute of the rest
                            if int(elem.attrib["sentence"]) == 0:
                                title, _, _ = append_text(title, elem.text)
                            else:
                                text, _, _ = append_text(text, elem.text)

                        if ECBPLUS in topic_file:
                            #add a string with the complete sentence to the sentence-dictionary for every different sentence
                            new_text, _, _ = append_text(sent_dict.get(int(elem.attrib["sentence"]), ""), elem.text)
                            sent_dict[int(elem.attrib["sentence"])] = new_text

                    except KeyError as e:
                        pass

                    if elem.tag == "Markables":

                        for i, subelem in enumerate(elem):
                            tokens = [token.attrib["t_id"] for token in subelem]
                            mention_text = ""
                            for t in tokens:
                                mention_text, _, _ = append_text(mention_text, token_dict[t]["text"])
                            #if "tokens" has values -> fill the "mention" dict with the value of the correspondant m_id
                            if len(tokens):
                                sent_id = int(token_dict[tokens[0]]["sent"])

                                #generate sentence doc with spacy
                                sentence = ""
                                iterationsWithFittingToken = 0
                                for t in root:
                                    if t.tag == "token":
                                        if t.attrib["sentence"] == str(sent_id):
                                            if iterationsWithFittingToken != 0 and t.text != "'s" and t.text != "." and t.text != "'" and t.text != "\"" and t.text != "," and t.text != ":" and t.text != ";" and t.text != "'s" and t.text != "?" and t.text != "!": 
                                                sentence = sentence + " "
                                            if t.text != "``" and t.text != "''":     
                                                sentence = sentence + str(t.text)
                                                iterationsWithFittingToken = iterationsWithFittingToken + 1

                                doc = nlp(sentence)

                                #missing: code, mention_ner, mention_head_pos, mention_head_lemma, mention_head, coref_link (same as code), mention_context
                                #(mention_head_lemma_uniques), (tokens_amount)
                                
                                mention_head = doc[token_dict[tokens[0]]["id"]].head
                                mention_head_lemma = mention_head.lemma_
                                mention_head_pos = mention_head.pos_
                                
                                mention_ner = mention_head.ent_type_
                                if mention_ner == "":
                                    mention_ner = "O"

                                #get the context
                                tokens_int = [int(x) for x in tokens]

                                context_min_id = int(min(tokens_int)) - 250
                                context_max_id = int(max(tokens_int)) + 250
                                if context_min_id < 0:
                                    context_min_id = 0
                                if context_max_id > len(token_dict):
                                    context_max_id = len(token_dict)-1

                                mention_context_str = []
                                for t in root:
                                    if t.tag == "token" and int(t.attrib["t_id"]) >= context_min_id and int(t.attrib["t_id"]) <= context_max_id:
                                        mention_context_str.append(t.text)


                                # "type":subelm.tag(e.g.:"HUMAN_PART_PER")
                                # "text":"textposToken1 token_dict+" "+TextposToken2 of tokendict..."
                                # "token_numbers":[all token_dict ID's of the tokens ]
                                # "doc_id":"name of file"
                                # "sent_id":"assosiated token_dict-Sentence-Num of 1. token of the subelement"
                                mentions[subelem.attrib["m_id"]] = {"type": subelem.tag,
                                                                    "text": mention_text.strip(),
                                                                    "sent_doc": doc,
                                                                    "mention_ner": mention_ner,
                                                                    "mention_head_pos": mention_head_pos,
                                                                    "mention_head_lemma": mention_head_lemma,
                                                                    "mention_head": mention_head.text,
                                                                    "token_numbers": [int(token_dict[t]["id"]) for t in tokens],
                                                                    "doc_id": topic_file.split(".")[0],
                                                                    "sent_id": sent_id,
                                                                    "mention_context": mention_context_str,
                                                                    "topic": t_subt}
                                #print("-----------------------")
                                #print(mentions)
                            else:
                                try:
                                    # if there are no t_ids (token is empty) and the "instance_id" is not in coref-dict:
                                    if subelem.attrib["instance_id"] not in coref_dict:
                                        # save in coref_dict to the instance_id
                                        coref_dict[subelem.attrib["instance_id"]] = {
                                            "descr": subelem.attrib["TAG_DESCRIPTOR"]}
                                    # m_id points to the target
                                except KeyError as e:
                                    pass

                    if elem.tag == "Relations":
                        # for every false create a false-value in "mentions_map"
                        mentions_map = {m: False for m in list(mentions)}
                        for i, subelem in enumerate(elem):
                            try:
                                if "r_id" not in coref_dict[subelem.attrib["note"]]:
                                    # update the coref_dict-element and add "r_id", "coref_type" and "mentions"
                                    # just elements of the subelements with tag "sourse" are added to mentions
                                    coref_dict[subelem.attrib["note"]].update({
                                        "r_id": subelem.attrib["r_id"],
                                        "coref_type": subelem.tag,
                                        "mentions": [mentions[m.attrib["m_id"]] for m in subelem if m.tag == "source"]
                                    })
                                else:
                                    coref_dict[subelem.attrib["note"]]["mentions"].extend(
                                        [mentions[m.attrib["m_id"]] for m in subelem if m.tag == "source"])
                            except KeyError:
                                # coref_dict[unknown_tag + "".join([m.attrib["m_id"] for m in subelem if m.tag == "target"])].update({
                                #     "r_id": subelem.attrib["r_id"],
                                #     "coref_type": subelem.tag,
                                #     "mentions": [mentions[m.attrib["m_id"]] for m in subelem if m.tag == "source"]
                                # })
                                pass
                            for m in subelem:
                                mentions_map[m.attrib["m_id"]] = True

                        for i, (m_id, used) in enumerate(mentions_map.items()):
                            if used:
                                continue

                            m = mentions[m_id]
                            if "Singleton_" + m["type"][:4] + "_" + str(m_id) + "_" + m["doc_id"] not in coref_dict:
                                coref_dict["Singleton_" + m["type"][:4] + "_" + str(m_id) + "_" + m["doc_id"]] = {
                                    "r_id": str(10000 + i),
                                    "coref_type": "Singleton",
                                    "mentions": [m],
                                    "descr": ""
                                }
                            else:
                                coref_dict["Singleton_" + m["type"][:4] + "_" + str(m_id) + "_" + m["doc_id"]].update(
                                    {
                                        "r_id": str(10000 + i),
                                        "coref_type": "Singleton",
                                        "mentions": [m],
                                        "descr": ""
                                    })
                    a = 1

                newsplease_custom = copy.copy(newsplease_format)

                if ECBPLUS in topic_file:
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
                    # set Variable "text" to " "+ rest of values (w.o. 1st one) that consist <= 10% of the numbers

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
                # "text" = text
                # "source_domain" = name of file
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
            coref_dics[topic_folder] = coref_dict

            entity_mentions_local = []
            event_mentions_local = []
            mentions_local = []
            for chain_id, chain_vals in coref_dict.items():
                for m in chain_vals["mentions"]:

                    if ECBPLUS.split(".")[0] not in m['doc_id']:
                        sent_id = int(m["sent_id"])
                    else:
                        df = doc_sent_map[m["doc_id"]]
                        # sent_id = np.sum([df.iloc[:list(df.index).index(int(m["sent_id"])) + 1][IS_TEXT].values])
                        sent_id = int(m["sent_id"])
                    # converts "token_numbers" of m to an array "token_numbers" with int values

                    # create variable "mention_id" out of doc_id+_+chain_id+_+sent_id+_+first value of "token_numbers" of "m"
                    token_numbers = [int(t) for t in m["token_numbers"]]
                    mention_id = m["doc_id"] + "_" + str(chain_id) + "_" + str(m["sent_id"]) + "_" + str(
                        m["token_numbers"][0])

                    #determine missing values--------------------------------------------------------------------------------------------------------------------------------------------------------
                    #code, mention_ner, mention_head_pos, mention_head_lemma, mention_head, coref_link (same as code), mention_context
                    #(mention_hhead_lemma_uniques), (tokens_amount)

                    #for t in tokens:
                    #    if t.head.ent_type_ != "":
                    #        mention_ner = t.head.ent_type_
                    #if mention_ner == "":
                    #    mention_ner = "O"


                    #create the dict. "mention" with all corresponding values
                    #print(m)
                    mention = {"coref_chain": chain_id,
                                "mention_ner": m["mention_ner"],
                                "mention_head_pos": m["mention_head_pos"],
                                "mention_head_lemma": m["mention_head_lemma"],
                                "mention_head": m["mention_head"],
                                "doc_id": m["doc_id"],
                                "is_continuous": True if token_numbers == list(
                                    range(token_numbers[0], token_numbers[-1] + 1))
                                else False,
                                "is_singleton": len(chain_vals["mentions"]) == 1,
                                "mention_id": mention_id,
                                "mention_type": m["type"][:3],
                                "mention_full_type": m["type"],
                                "score": -1.0,
                                "sent_id": sent_id,
                                "mention_context": m["mention_context"],
                                "tokens_number": token_numbers,
                                "tokens_str": m["text"],
                                "topic_id": topic_name,
                                "coref_type": chain_vals["coref_type"],
                                "decription": chain_vals["descr"]
                                }

                    # if the first two entries of chain_id are "ACT" or "NEG", add the "mention" to the array "event_mentions_local"
                    if chain_id[:3] in ["ACT", "NEG"]:
                        event_mentions_local.append(mention)
                    # else add the "mention" to the array "event_mentions_local" and add the following values to the DF "summary_df"
                    else:
                        entity_mentions_local.append(mention)

                    mentions_local.append(mention)

                    summary_df = summary_df.append(pd.DataFrame({
                        "doc_id": m["doc_id"],
                        "coref_chain": chain_id,
                        "decription": chain_vals["descr"],
                        "short_type": chain_id[:3],
                        "full_type": m["type"],
                        "tokens_str": m["text"]
                    }, index=[mention_id]))

                    t_subt = m["topic"]
                    mark_counter = chain_id
                    # todo: once with (number) as 1st entry the placeholder stays as "- ," before the (number).
                    if token_numbers[0] == token_numbers[len(token_numbers) - 1]:
                        if str(conll_df.loc[(conll_df['sent_id'] == sent_id) & (
                                conll_df['token_id'] == token_numbers[0]) & (conll_df[
                                                                                       'topic/subtopic_name'] == t_subt), 'coref'].values[
                                   0]) == "-":

                            conll_df.loc[(conll_df['sent_id'] == sent_id) & (
                                    conll_df['token_id'] == token_numbers[0]) & (conll_df[
                                                                                           'topic/subtopic_name'] == t_subt), 'coref'] = '(' + str(
                                mark_counter) + ')'

                        else:
                            conll_df.loc[(conll_df['sent_id'] == sent_id) & (
                                    conll_df['token_id'] == token_numbers[0]) & (conll_df[
                                                                                           'topic/subtopic_name'] == t_subt), 'coref'] = str(
                                conll_df.loc[(conll_df['sent_id'] == sent_id) & (
                                        conll_df['token_id'] == token_numbers[0] & (
                                            conll_df['topic/subtopic_name'] == t_subt)), 'coref'].values[
                                    0]) + '| (' + str(mark_counter) + ')'

                    else:
                        # if there stands "-" at the location with the df with the right token and sentence and topic-file, hence first entry, then for the 1st token of the arrays:
                        if str(conll_df.loc[(conll_df['sent_id'] == sent_id) & (
                                conll_df['token_id'] == token_numbers[0]) & (conll_df[
                                                                                       'topic/subtopic_name'] == t_subt), 'coref'].values[
                                   0]) == "-":
                            conll_df.loc[(conll_df['sent_id'] == sent_id) & (
                                    conll_df['token_id'] == token_numbers[0]) & (conll_df[
                                                                                           'topic/subtopic_name'] == t_subt), 'coref'] = '(' + str(
                                mark_counter)

                        else:
                            conll_df.loc[(conll_df['sent_id'] == sent_id) & (
                                    conll_df['token_id'] == token_numbers[0]) & (conll_df[
                                                                                           'topic/subtopic_name'] == t_subt), 'coref'] = str(
                                conll_df.loc[(conll_df['sent_id'] == sent_id) & (
                                        conll_df['token_id'] == token_numbers[0]) & (conll_df[
                                                                                               'topic/subtopic_name'] == t_subt), 'coref'].values[
                                    0]) + '| (' + str(mark_counter)

                        # if there stands "-" at the location with the df with the right token and sentence and topic-file, hence first entry, then for the last token of the arrays:
                        if str(conll_df.loc[(conll_df['sent_id'] == sent_id) & (
                                conll_df['token_id'] == token_numbers[
                            len(token_numbers) - 1]) & (conll_df[
                                                            'topic/subtopic_name'] == t_subt), 'coref'].values[
                                   0]) == "-":
                            conll_df.loc[(conll_df['sent_id'] == sent_id) & (
                                    conll_df['token_id'] == token_numbers[
                                len(token_numbers) - 1]) & (conll_df[
                                                                'topic/subtopic_name'] == t_subt), 'coref'] = str(
                                mark_counter) + ')'
                        else:
                            conll_df.loc[(conll_df['sent_id'] == sent_id) & (
                                        conll_df['token_id'] == token_numbers[len(token_numbers) - 1]) &
                                               (conll_df['topic/subtopic_name'] == t_subt), 'coref'] = str(
                                conll_df.loc[(conll_df['sent_id'] == sent_id)
                                                   & (conll_df['token_id'] == token_numbers[
                                    len(token_numbers) - 1]) &
                                                   (conll_df['topic/subtopic_name'] == t_subt), 'coref'].values[
                                    0]) + '| ' + str(mark_counter) + ')'

                    a = 0

            # create annot_path and file-structure (if not already) for the output of the annotations
            annot_path = os.path.join(result_path, topic_name, "annotation", "original")
            # ->root/data/ECBplus-prep/test_parsing/topicName/annotation/original
            if topic_name not in os.listdir(os.path.join(result_path)):
                os.mkdir(os.path.join(result_path, topic_name))

            if "annotation" not in os.listdir(os.path.join(result_path, topic_name)):
                os.mkdir(os.path.join(result_path, topic_name, "annotation"))
                os.mkdir(annot_path)
            # create the entity-mentions and event-mentions - .json files out of the arrays
            with open(os.path.join(annot_path, "entity_mentions_" + topic_name + ".json"), "w") as file:
                json.dump(entity_mentions_local, file)
            entity_mentions.extend(entity_mentions_local)

            with open(os.path.join(annot_path, "event_mentions_" + topic_name + ".json"), "w") as file:
                json.dump(event_mentions_local, file)
            event_mentions.extend(event_mentions_local)

            conll_topic_df = conll_df[conll_df["topic/subtopic_name"].str.contains(f'{topic_name}/')]

            outputdoc_str = ""
            for (topic_local), topic_df in conll_topic_df.groupby(by=["topic/subtopic_name"]):
                outputdoc_str += f'#begin document ({topic_local}); part 000\n'

                for (sent_id_local), sent_df in topic_df.groupby(by=["sent_id"], sort=["sent_id"]):
                    np.savetxt(os.path.join(TMP_PATH, "tmp.txt"), sent_df.values, fmt='%s', delimiter="\t",
                               encoding="utf-8")
                    with open(os.path.join(TMP_PATH, "tmp.txt"), "r", encoding="utf8") as file:
                        saved_lines = file.read()
                    outputdoc_str += saved_lines + "\n"

                outputdoc_str += "#end document\n"
            final_output_str += outputdoc_str

            with open(os.path.join(annot_path, f'{topic_name}.conll'), "w", encoding='utf-8') as file:
                file.write(outputdoc_str)

            #Determine the lexical diversity and the average unique head lemmas

            #Average number of unique head lemmas within a cluster
            #(excluding singletons for fair comparison)
            for m in mentions_local:
                head_lemmas = [m["mention_head_lemma"]]
                uniques = 1
                for m2 in mentions_local:
                    if m2["mention_head_lemma"] not in head_lemmas and m2["tokens_str"] == m["tokens_str"] and m2["topic_id"] == m["topic_id"]:
                        head_lemmas.append(m["mention_head_lemma"])
                        uniques = uniques+1
                m["mention_head_lemma_uniques"] = uniques

            summary_conversion_df = summary_conversion_df.append(pd.DataFrame({
                "files": len(annot_folders),
                "tokens": len(conll_topic_df),
                "chains": len(coref_dict),
                "event_mentions_local": len(event_mentions_local),
                "entity_mentions_local": len(entity_mentions_local),
                "singletons": sum([v["is_singleton"] for v in event_mentions_local]) + sum([v["is_singleton"] for v in entity_mentions_local]),
                "avg_unique_head_lemmas": mean([v["mention_head_lemma_uniques"] for v in mentions_local])
            }, index=[topic_name]))

            a = 1

            #break for testing
            break

    now = datetime.now()

    with open(os.path.join(result_path, now.strftime("%Y-%m-%d_%H-%M") + "_" + 'ecbplus.conll'), "w", encoding='utf-8') as file:
        file.write(final_output_str)

    with open(os.path.join(result_path, now.strftime("%Y-%m-%d_%H-%M") + "_" + "entity_mentions_" + ".json"), "w", encoding='utf-8') as file:
        json.dump(entity_mentions, file)

    with open(os.path.join(result_path, now.strftime("%Y-%m-%d_%H-%M") + "_" + "event_mentions_" + ".json"), "w", encoding='utf-8') as file:
        json.dump(event_mentions, file)

    # create a csv. file out of the summary_df ->root/data/ECBplus-prep/test_parsing with the dateTime in the name
    summary_df.to_csv(os.path.join(result_path, now.strftime("%Y-%m-%d_%H-%M") + "_" + "all_mentions.csv"))
    summary_conversion_df.to_csv(os.path.join(result_path, now.strftime("%Y-%m-%d_%H-%M") + "_" + "dataset_summary.csv"))


# main function for the input which topics of the ecb corpus are to be converted
if __name__ == '__main__':
    topic_num = 45
    convert_files(topic_num)
    print("\nConversion of {0} topics from xml to newsplease format and to annotations in a json file is "
          "done. \n\nFiles are saved to {1}. \n.".format(str(topic_num), result_path))
