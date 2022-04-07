CONTEXT_RANGE = 250
PATH = 'test_parsing_no_annotations'  
PATH_ANNOTATIONS = '2019_annot/'   
AGGR_FILENAME = 'aggr_m_conceptcategorization.csv'     
OUT_DIR = 'output_data/'    #the directory to output data to

import spacy
import glob
import os
import json
import pandas as pd
import shortuuid
import ujson
from tqdm import tqdm

def get_context(doc, fitting_tokens_docID):
    context_min_id = min(fitting_tokens_docID) - CONTEXT_RANGE
    context_max_id = max(fitting_tokens_docID) + CONTEXT_RANGE
    if context_min_id < 0:
        context_min_id = 0
    if context_max_id > len(doc):
        context_max_id = len(doc)-1

    mention_context = doc[context_min_id:context_max_id]
    mention_context_str = []
    for c in mention_context:
        mention_context_str.append(c.text)
    return mention_context_str

def checkContinuous(token_numbers):
   return token_numbers == list(range(token_numbers[0], token_numbers[-1] + 1))

print("Starting routine... Retrieving data from specified directories.")

df = pd.DataFrame(columns=["authors", "date_download", "date_modify", "date_publish", "description", "filename", "image_url", "language", "localpath", "source_domain", "text", "title", "title_page", "title_rss", "url"])

for filename in os.listdir(PATH)[1:]:
    frames = []
    print("Reading in files from folder: " + filename)
    for full_filename in glob.glob(os.path.join(PATH+"/"+filename, "*.json")):    #iterate through every file
        print("Executing code for ", str(full_filename))
        with open(full_filename, encoding='utf-8', mode='r') as currentFile:
            jo = json.loads(currentFile.read())
            df_tmp = pd.DataFrame({"authors": [jo["authors"]], "date_download": jo["date_download"], "date_modify": jo["date_modify"], "date_publish": jo["date_publish"], "description": jo["description"], "filename": filename, "image_url": jo["image_url"], "language": jo["language"], "localpath": jo["localpath"], "source_domain": jo["source_domain"], "text": jo["text"], "title": jo["title"], "title_page": jo["title_page"], "title_rss": jo["title_rss"], "url": jo["url"]})
            df = pd.concat([df, df_tmp], ignore_index=True, axis = 0)

nlp = spacy.load('en_core_web_sm')
docs = []

for element in tqdm(df["text"]):
    doc = nlp(element)
    docs.append(doc)

df["doc"] = docs

#read other csvs for annotations
df_annotations = pd.DataFrame()
topics = []

for full_filename in glob.glob(os.path.join(PATH_ANNOTATIONS, "*.csv")):    #iterate through every file
    print("Executing code for ", str(full_filename))
    with open(full_filename, encoding='utf-8', mode='r') as currentFile:
        df_tmp = pd.read_csv(currentFile)
        df_tmp["topic_id"] = int(full_filename.split("\\")[1].split("_")[0])
        topics.append(int(full_filename.split("\\")[1].split("_")[0]))
        df_annotations = pd.concat([df_annotations, df_tmp], ignore_index=True, axis = 0)

with open(AGGR_FILENAME, encoding='utf-8', mode='r') as currentFile:
    concept_df = pd.read_csv(currentFile)
    concept_df.topic_id = concept_df.topic_id.astype(int)

df_annotations = pd.merge(df_annotations, concept_df, how = "left", on = ["topic_id", "Code"])
df_annotations = df_annotations.rename({"comments": "coref_chain"}, axis = 1)
df_annotations = df_annotations[df_annotations.type.notna()]
df_annotations.reset_index(drop=True,inplace=True)

coref_chains = {}
chains_list = []
for i, row in df_annotations.iterrows():
    if row["Code"] + "_" + str(row["topic_id"]) not in coref_chains:
        mention_full_type = str(row["type"])
        mention_type = mention_full_type.replace("-","")
        coref_chain = mention_type + "_" + str(shortuuid.random(8))
        #print(coref_chain)
        coref_chains[row["Code"] + "_" + str(row["topic_id"])] = {"mention_full_type": mention_full_type, "mention_type": mention_full_type.replace("-",""), "coref_chain": coref_chain}    
    chains_list.append(coref_chains[row["Code"] + "_" + str(row["topic_id"])]["coref_chain"])

df_annotations["coref_chain"] = chains_list

print(df_annotations)

df_preparation_list = []
tokens_amount = 0
prev_doc = 0

print("Going to process " + str(len(df)) + " articles...")
for i, row in tqdm(df.iterrows()):
    doc = row["doc"]
    filename = row["filename"]
    description = row["description"]
    doc_id = row["source_domain"].split("_")[0]
    pol_direction = row["source_domain"].split("_")[1]
    doc_id_full = row["source_domain"]
    doc_unique_id = doc_id_full+"_"+str(i)

    #count the amount of tokens for each general topic (only the max value in the dataframe represents the correct tokens_number at the end)
    if prev_doc == doc_id:
        prev_doc = doc_id
        tokens_amount = tokens_amount + len(doc)
    else:
        tokens_amount = len(doc)
        prev_doc = doc_id
    
    #compare whole segment with tokens
    #iterate over tokens and segment tokens seperately
    #if they are close enough (+-1 in position) accept it (maybe print warning)
    #take the accepted tokens and determine the head and other attributes

    for j, sentence in enumerate(doc.sents):
        for anno_i, row_anno in df_annotations.iterrows():
            if doc_id_full == row_anno["Document name"]:
                #if row_anno["Segment"] in sentence.text:

                fitting_tokens = []
                fitting_tokens_docID = []
                fitting_token_str = []
                seg_token = 0
                iterations_without_token_match = 0
            
                #tokenize the segment
                segment_doc = nlp(row_anno["Segment"])
                segment_tokenized = []
                for t in segment_doc:
                    segment_tokenized.append(t)

                for token in sentence:

                    if len(segment_tokenized) <= seg_token:
                        #add to the list
                        if seg_token >= 1:
                            sent_id = j
                            text = row["text"]
                            code = row_anno["Code"].replace("\\", " ")
                            segment = row_anno["Segment"]
                            score = -1  #not row_anno["Weight score"], not that important (legacy)
                            is_continuous = checkContinuous(tokens_number[:])
                            is_singleton = False    #placeholder
                            mention_id = doc_unique_id+"_"+str(sent_id)+"_"+str(tokens_number[0])

                            #determine the head
                            for t_id in fitting_tokens_docID:
                                ancestors_in_mention = 0
                                for a in doc[t_id].ancestors:
                                    if a.i in fitting_tokens_docID:
                                        ancestors_in_mention = ancestors_in_mention + 1
                                        break   #one is enough to make the token unviable as a head
                                if ancestors_in_mention == 0:
                                    #head within the mention
                                    mention_head = doc[t_id]

                            head_pos = mention_head.pos_
                            head_lemma = mention_head.lemma_
                            head_str = mention_head.text
                            coref_type = "STRICT"

                            mention_context_str = get_context(doc, fitting_tokens_docID)
                            
                            mention_ner = mention_head.ent_type_
                            if mention_ner == "":
                                mention_ner = "O"

                            coref_chain = row_anno["coref_chain"]
                            mention_type = row_anno["type"]
                            
                            if iterations_without_token_match >= 1:
                                print("Warning: the segment does not match 100%: " + segment)
                                print("Sentence: " + str(sentence))

                            df_preparation_list.append([coref_chain, code, segment, tokens_amount, mention_ner, head_pos, head_lemma, mention_head.text, code, doc_unique_id, doc_id, pol_direction, is_continuous, is_singleton, text, sentence.text, mention_id, mention_type, mention_type, score, sent_id, mention_context_str, fitting_tokens, fitting_token_str, row_anno["Segment"], filename.split("_")[0], coref_type, code, head_str, mention_head.i])
                            break   #only one mention of the same kind per sentence is valid

                        fitting_tokens = []
                        fitting_tokens_docID = []
                        fitting_token_str = []
                        seg_token = 0

                    if token.text == segment_tokenized[seg_token].text:
                        if seg_token == 0:
                            iterations_without_token_match = 0
                        seg_token = seg_token + 1
                        fitting_tokens.append(token.i - sentence.start)     #the id of the token within a sentence
                        fitting_tokens_docID.append(token.i)
                        fitting_token_str.append(token.text)

                        tokens = []
                        tokens_str = ""
                        tokens_number = []
                        for t in token.head.subtree:
                            tokens.append(t)
                            tokens_str = tokens_str + " " + t.text
                            tokens_number.append(t.i)
                    else:
                        iterations_without_token_match = iterations_without_token_match + 1
                        if iterations_without_token_match > 1:
                            #reset detection if the tokens are spread to far (not exact matching +-1)
                            tokens = []
                            tokens_str = ""
                            tokens_number = []
                            fitting_tokens = []
                            fitting_tokens_docID = []
                            fitting_token_str = []
                            iterations_without_token_match = 0
                            seg_token = 0
    #if i > 3:
    #    break
                
mentions_df = pd.DataFrame(df_preparation_list, columns=["coref_chain", "code", "segment", "tokens_amount", "mention_ner", "mention_head_pos", "mention_head_lemma", "mention_head", "coref_link", "doc_id_full","doc_id", "pol_direction", "is_continuous", "is_singleton", "text", "sentence", "mention_id", "mention_type", "mention_full_type", "score", "sent_id", "mention_context", "tokens_number", "fitting_tokens", "tokens_str", "topic_id", "coref_type", "description", "head_str", "mention_head_id"])

#determine average number of unique head lemmas within a cluster
#(excluding singletons for fair comparison)
print(len(mentions_df))

mentions_df = mentions_df.join(mentions_df.groupby(['topic_id', 'segment'])["mention_head_lemma"].nunique(), on=['topic_id', 'segment'], rsuffix='_uniques')
#remove rows from dataframe with duplicate code possibilities (just take the first one)
print(len(mentions_df))
mentions_df = mentions_df.drop_duplicates(subset=['doc_id_full', 'sent_id', 'coref_chain'], keep='first')   #mention_head_id
#print(len(mentions_df))

#determine singletons
for i, row in tqdm(mentions_df.iterrows()):
    amountOfMentions = len( mentions_df[ mentions_df.coref_chain == row["coref_chain"] ] )
    #print(amountOfMentions)
    if amountOfMentions == 1:
        mentions_df.at[i, "is_singleton"] = True

print("Singletons: " + str(len(mentions_df[ mentions_df.is_singleton == True ])))
print("Amount of mentions: " + str(len(mentions_df))) 

#drop unwanted columns for the output
mentions_df = mentions_df.drop(["segment", "text", "sentence", "fitting_tokens", "head_str"], axis = 1)
mentions_df.reset_index(drop=True,inplace=True)

with open(f'./mentions_df.json', 'w', encoding='utf-8') as f:
    f.write(ujson.dumps(mentions_df.to_dict('index'), indent=4, ensure_ascii=False, escape_forward_slashes=False))
print(len(mentions_df))

#define entities and event mentions to generate two different json files later
events = ["ACTION", "EVENT", "MISC"]
df_entities = pd.DataFrame()
df_events = pd.DataFrame()

for i, row in mentions_df.iterrows():
    if row["mention_type"] not in events:
        df_entities = df_entities.append(row)
    else:
        df_events = df_events.append(row)

df_entities.drop(columns= ["tokens_amount", "mention_head_lemma_uniques"], inplace = True)
df_events.drop(columns= ["tokens_amount", "mention_head_lemma_uniques"], inplace = True)

with open(OUT_DIR +'entity_mentions.json', 'w', encoding='utf-8') as f:
    f.write(ujson.dumps(df_entities.to_dict('records'), indent=4, ensure_ascii=False, escape_forward_slashes=False))

with open(OUT_DIR +'event_mentions.json', 'w', encoding='utf-8') as f:
    f.write(ujson.dumps(df_events.to_dict('records'), indent=4, ensure_ascii=False, escape_forward_slashes=False))

#also make a all_mentions csv file
df_all_mentions = mentions_df.drop(columns=["is_continuous", "is_singleton", "score", "sent_id", "tokens_amount", "tokens_number", "topic_id", "coref_type", "mention_context", "mention_ner", "mention_head_pos", "mention_head_lemma", "coref_link", "tokens_amount", "code", "mention_head", "mention_head_lemma_uniques"]).rename(index = mentions_df.mention_id)
df_all_mentions = df_all_mentions.drop(columns = ["mention_id"])
df_all_mentions.to_csv(path_or_buf="all_mentions.csv", sep=",", na_rep="")

#generate conll file
print("Generating conll...")
df_conll = pd.DataFrame(columns = {"topic_id", "doc_identifier", "sent_id", "token_id", "token", "reference"})

for i, row in tqdm(df.iterrows()):
    for sentence_id, sentence in enumerate(row["doc"].sents):
        for token_id, token in enumerate(sentence):
            if token.text != "\n":
                doc_id_full = row["source_domain"]
                doc_unique_id = doc_id_full+"_"+str(i)
                topic_subtopic = doc_id_full.split("_")[0] + "_" + doc_id_full.split("_")[1]

                df_conll = df_conll.append( {"topic/subtopic_name": topic_subtopic, "doc_identifier": doc_unique_id, "sent_id": sentence_id, "token_id": token_id, "token": token.text, "reference": ""}, ignore_index=True)
    #if i > 3:
    #    break
added_corefs = []

print("Processing " + str(len(df_conll)) + " df_conll rows...")

for i, row_conll in tqdm(df_conll.iterrows()):
    for j, row_final in mentions_df.iterrows():
        tokens_number = row_final["tokens_number"]
        token_id = row_conll["token_id"]
        sent_id = row_conll["sent_id"]
        
        if row_conll["doc_identifier"] == row_final["doc_id_full"] and row_final["sent_id"] == sent_id and token_id in tokens_number:
            coref_chain = row_final["coref_chain"] + "_" + str(row_final["doc_id_full"]) + "_" + str(sent_id) + "_" + str(''.join(str(x) for x in tokens_number))
            added_corefs.append(coref_chain)
            #determine the amount of already added corefs with the id (to get the brackets correct)
            coref_count = added_corefs.count(coref_chain)
            #add a string based on how many or if all corefs have been added
            if coref_count == 1 and len(tokens_number) == 1:    #first and only coref token
                df_conll.at[i, "reference"] = row_conll["reference"] + "| (" + row_final["coref_chain"] + ")"
            elif coref_count == 1:    #first of multiple coref token
                df_conll.at[i, "reference"] = row_conll["reference"] + "| (" + row_final["coref_chain"]
            elif coref_count == len(tokens_number):   #last coref token
                df_conll.at[i, "reference"] = row_conll["reference"] + "| " + row_final["coref_chain"] + ")"
    
    if row_conll["reference"] != "":
        if row_conll["reference"][0] == "|":    #remove first | from string
            df_conll.at[i, "reference"] = row_conll["reference"][2:]
    #if i > 500:
    #    break

df_conll.reset_index(drop=True,inplace=True)

with open(f'./conll_test.json', 'w', encoding='utf-8') as f:
    f.write(ujson.dumps(df_conll.to_dict('index'), indent=4, ensure_ascii=False, escape_forward_slashes=False))

with open(os.path.join(OUT_DIR, "conll_as_json" + ".json"), "w", encoding='utf-8') as file:
    json.dump(df_conll.to_dict('records'), file)

#Make the conll file from the dataframe
dfAsString = ""
previous_sentence = df_conll["sent_id"][0]
previous_doc =  df_conll["doc_identifier"][0]
print("Creating a conll string...")
with open(f'./newswcl50.conll', 'w', encoding='utf-8') as f:
    for i, row in tqdm(df_conll.iterrows()):
        #line breaks at new sentences and #header and #end
        if i == 0:
            dfAsString = dfAsString + "#begin document " + row["doc_identifier"] + "; part 000" + "\n"
        if row["sent_id"] != previous_sentence:
            dfAsString = dfAsString + "\n"
        if row["doc_identifier"] != previous_doc:
            dfAsString = dfAsString + "#end document" + "\n"
            dfAsString = dfAsString + "#begin document " + row["doc_identifier"] + "; part 000" + "\n"

        if row['reference'] != "":
            dfAsString = dfAsString + row['doc_identifier'] + '\t' + str(row['sent_id']) + '\t' + str(row['token_id']) + '\t' + row['token'] + '\t' + row['reference'] + "\n"
        else:
            dfAsString = dfAsString + row['doc_identifier'] + '\t' + str(row['sent_id']) + '\t' + str(row['token_id']) + '\t' + row['token'] + '\t' + "-" + "\n"
        previous_sentence = row["sent_id"]
        previous_doc = row["doc_identifier"]
    f.write(dfAsString)

#check conll by counting brackets (should be equal)
print("Checking equal brackets in conll:")
print("(: " + str(dfAsString.count("(")))
print("): " + str(dfAsString.count(")")))

#generate a dataset summary
df_summary = mentions_df.groupby(by = ["doc_id"], as_index=False).agg(
    {
        'doc_id_full': ["nunique"],
        'coref_chain': ["nunique"],
        'tokens_amount': ["max"],   
        'mention_full_type': [lambda x: x[x.str.contains('EVENT|MISC|ACTION')].count(), lambda x: x[~x.str.contains('EVENT|MISC|ACTION')].count()],    #event_mentions and entity_mentions
        'is_singleton': ["sum"],
        'mention_head_lemma_uniques': ["mean"]
    }
)

df_summary.columns = df_summary.columns.droplevel(0)
df_summary.columns = ["doc_id", "files", "chains", "tokens", "event_mentions", "entity_mentions", "is_singleton", "avg_unique_head_lemmas"]

df_summary.reset_index(drop=True,inplace=True)
df_summary.rename(index = df_summary["doc_id"], inplace = True)
df_summary = df_summary.drop(columns= ["doc_id"] )

df_summary.to_csv(path_or_buf="dataset_summary.csv", sep=",", na_rep="")

print("Done.")