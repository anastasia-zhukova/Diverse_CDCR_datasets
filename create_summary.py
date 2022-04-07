DIRECTORIES_TO_SUMMARIZE = ["ECBplus-prep/", "NewsWCL50-prep/"]
OUTPUT_FOLDER_NAME = "output_data"
CONLL_JSON = "conll_as_json.json"

import pandas as pd
import spacy
from tqdm import tqdm
import glob
import os
import json
from pd_metric import phrasing_complexity_calc

summary_df = pd.DataFrame()

for dir in DIRECTORIES_TO_SUMMARIZE:
    #if "ECB" in str(dir):
    #    print("Skipping")
    #    continue
    df = pd.DataFrame()
    events_df = pd.DataFrame()
    entities_df = pd.DataFrame()
    dataset_mentions_dicts = []

    print("Reading in files from folder: " + dir + OUTPUT_FOLDER_NAME)
    for full_filename in glob.glob(os.path.join(dir + OUTPUT_FOLDER_NAME, "*.json")):    #iterate through every file
        if "conll" in full_filename:
            #do not process the jsonified conll (yet):
            continue

        print("Executing code for ", str(full_filename))
        with open(full_filename, encoding='utf-8', mode='r') as currentFile:
            df_tmp = pd.read_json(full_filename)
            df_tmp["dataset"] = [dir for i in range(len(df_tmp.index))]
            df = pd.concat([df, df_tmp], ignore_index=True, axis = 0)
            df.is_singleton = df.is_singleton.astype(bool)
            df.is_continuous = df.is_continuous.astype(bool)

            #append to event or entity list
            if "event" in full_filename:
                events_df = df_tmp
            else:
                entities_df = df_tmp

    #get the data from the jsonified conll 
    with open(dir + OUTPUT_FOLDER_NAME + "/" + CONLL_JSON) as json_file:
        print("Opening conll json...")
        jo = json.loads(json_file.read())
        conll_df = pd.DataFrame(jo)
        print(conll_df)
    

    #get the list of topics
    topics = sorted(list(set(df["topic_id"].tolist())))

    print(df)

    #make list of tokens_str to make it suitable for the PD function
    tokens_texts = []
    for i, row in df.iterrows():
        tokens_texts.append(row["tokens_str"].split(" "))
    df["tokens_text"] = tokens_texts

    for topic in tqdm(topics):
        #print("Summarizing topic " + str(topic) + "...")
        #make a list of dicts
        dicts = []
        coref_chains = []
        pd_c_dicts = []
    
        for i, row in df[~df.is_singleton].iterrows():
            if(row["topic_id"] == topic):
                coref_chains.append(row["coref_chain"])
                mention = {
                        "coref_chain": row["coref_chain"],
                        "sentence": row["sent_id"], 
                        "id": row["topic_id"], 
                        "tokens": row["tokens_number"],
                        "tokens_text": row["tokens_text"],
                        "text": row["tokens_str"], 
                        "head_token_index": row["mention_head_id"], 
                        "mention_head": row["mention_head"]
                    }
                dicts.append( mention )
                dataset_mentions_dicts.append( mention )
        
        coref_chains = list(set(coref_chains))  #make unique

        #calculate pd_c per chain
        for c in coref_chains:
            d_c = [d for d in dicts if d["coref_chain"] == c]
            pd_c = phrasing_complexity_calc(d_c)
            m_c = len(d_c)
            pd_c_dicts.append({
                "m_c": m_c,
                "pd_c": pd_c
            })

        #summarize the pd_cs per chain into a pd per topic (wheighted average)
        pd_total_1 = 0
        pd_total_2 = 0
        for p in pd_c_dicts:
            pd_total_1 = pd_total_1 + p["m_c"] * p["pd_c"]
            pd_total_2 = pd_total_2 + p["m_c"]
           
        print(pd_c_dicts)

        result_pd = pd_total_1 / pd_total_2
        #print("Result PD for the topic: " +  str(result_pd))
        
        #print(dicts)
        #add the values for the summary per topic
        summary_df = summary_df.append(pd.DataFrame({
                "dataset": str(dir).split("-")[0],
                "topic_name": str(topic),
                "articles": len(list(set(df[df.topic_id == topic]["doc_id_full"].tolist()))),
                "tokens": len(conll_df[conll_df["topic/subtopic_name"].str.startswith(str(topic))]),
                "chains": len(list(set(df[df.topic_id == topic]["coref_chain"].tolist()))),
                "event_mentions": len(events_df[events_df.topic_id == topic]),
                "entity_mentions": len(entities_df[entities_df.topic_id == topic]),
                "singletons": df[df.topic_id == topic]["is_singleton"].values.sum(),
                "lexical_diversity": result_pd 
            },
            index = [str(dir).split("-")[0] + "_" + str(topic)]
        ))
       
    #calculate the overall statistics for the whole dataset (not just per topic)
    pd_c_dicts = []
    print(len(dataset_mentions_dicts))
    for c in list(set(df[~df.is_singleton]["coref_chain"])):     #iterate over all unique chains in the dataset (excluding singletons)
        d_c = [d for d in dataset_mentions_dicts if d["coref_chain"] == c]
        pd_c = phrasing_complexity_calc(d_c)
        m_c = len(d_c)
        pd_c_dicts.append({
            "m_c": m_c,
            "pd_c": pd_c
        })
        #print("PD_c for chain: " + str(pd_c))
        #print("m_c: " + str(m_c))
    
    pd_total_1 = 0
    pd_total_2 = 0
    for p in pd_c_dicts:
        pd_total_1 = pd_total_1 + p["m_c"] * p["pd_c"]
        pd_total_2 = pd_total_2 + p["m_c"]
    result_total_pd_dataset = pd_total_1 / pd_total_2
        
    #create total row in summary df

    summary_df = summary_df.append(pd.DataFrame({
        "dataset": str(dir).split("-")[0],
        "topic_name": "total",
        "articles": len(list(set(df["doc_id_full"].tolist()))),
        "tokens": len(conll_df),
        "chains": len(list(set(df["coref_chain"].tolist()))),
        "event_mentions": len(events_df),
        "entity_mentions": len(entities_df),
        "singletons": df["is_singleton"].values.sum(),
        "lexical_diversity": result_total_pd_dataset
        },
        index = [str(dir).split("-")[0] + "_" + str(topic)]
    ))

#Output final result
print(summary_df)

summary_df.to_csv(path_or_buf="dataset_summary.csv", sep=",", na_rep="")
