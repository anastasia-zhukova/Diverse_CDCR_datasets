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

#used to get the context within the doc for a specific range of tokens 
#fitting_tokens_doc_id is provided as list of ints
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

#checks wether a list of token ints is continuous
def checkContinuous(token_numbers):
   return token_numbers == list(range(token_numbers[0], token_numbers[-1] + 1))

#START OF PROGRAM--------------
print("Starting routine... Retrieving data from specified directories.")
df = pd.DataFrame(columns=["authors", "date_download", "date_modify", "date_publish", "description", "filename", "image_url", "language", "localpath", "source_domain", "text", "title", "title_page", "title_rss", "url"])

#read dataset files
for filename in os.listdir(PATH)[1:]:
    frames = []
    print("Reading in files from folder: " + filename)
    for full_filename in glob.glob(os.path.join(PATH+"/"+filename, "*.json")):    #iterate through every file
        print("Executing code for ", str(full_filename))
        with open(full_filename, encoding='utf-8', mode='r') as currentFile:
            jo = json.loads(currentFile.read())
            df_tmp = pd.DataFrame({"authors": [jo["authors"]], "date_download": jo["date_download"], "date_modify": jo["date_modify"], "date_publish": jo["date_publish"], "description": jo["description"], "filename": filename, "image_url": jo["image_url"], "language": jo["language"], "localpath": jo["localpath"], "source_domain": jo["source_domain"], "text": jo["text"], "title": jo["title"], "title_page": jo["title_page"], "title_rss": jo["title_rss"], "url": jo["url"]})
            df = pd.concat([df, df_tmp], ignore_index=True, axis = 0)

#loading spacy NLP
nlp = spacy.load('en_core_web_sm')
docs = []

#appending all articles as spacy docs to the dataframe
for element in tqdm(df["text"]):
    doc = nlp(element)
    docs.append(doc)
df["doc"] = docs

#read other csvs for the annotations
df_annotations = pd.DataFrame()
topics = []
for full_filename in glob.glob(os.path.join(PATH_ANNOTATIONS, "*.csv")):    #iterate through every file
    print("Executing code for ", str(full_filename))
    with open(full_filename, encoding='utf-8', mode='r') as currentFile:
        df_tmp = pd.read_csv(currentFile)
        df_tmp["topic_id"] = int(full_filename.split("\\")[1].split("_")[0])
        topics.append(int(full_filename.split("\\")[1].split("_")[0]))
        df_annotations = pd.concat([df_annotations, df_tmp], ignore_index=True, axis = 0)

#Open the aggregated file to get the entity types per code
with open(AGGR_FILENAME, encoding='utf-8', mode='r') as currentFile:
    concept_df = pd.read_csv(currentFile)
    concept_df.topic_id = concept_df.topic_id.astype(int)

#assign every segment that gets mentioned to a specific code and entity type 
df_annotations = pd.merge(df_annotations, concept_df, how = "left", on = ["topic_id", "Code"])
df_annotations = df_annotations.rename({"comments": "coref_chain"}, axis = 1)
#make sure no NA values are present after merging
df_annotations = df_annotations[df_annotations.type.notna()]
df_annotations.reset_index(drop=True,inplace=True)

#create coref_chain ids which show the connection/corellation of many segment mentions within the same topic
coref_chains = {}
chains_list = []
for i, row in df_annotations.iterrows():
    if row["Code"] + "_" + str(row["topic_id"]) not in coref_chains:
        mention_full_type = str(row["type"])
        mention_type = mention_full_type.replace("-","")
        coref_chain = mention_type + "_" + str(shortuuid.random(8))
        coref_chains[row["Code"] + "_" + str(row["topic_id"])] = {"mention_full_type": mention_full_type, "mention_type": mention_full_type.replace("-",""), "coref_chain": coref_chain}    
    chains_list.append(coref_chains[row["Code"] + "_" + str(row["topic_id"])]["coref_chain"])
#assign a unique coref_chain id to every code 
df_annotations["coref_chain"] = chains_list

print(df_annotations)

#process each article doc one by one for the reannotation
df_preparation_list = []
tokens_amount = 0
prev_doc = 0
print("Going to process " + str(len(df)) + " articles...")
for i, row in tqdm(df.iterrows()):
    #read the given data from the dataset 
    doc = row["doc"]
    filename = row["filename"]
    description = row["description"]
    doc_id = row["source_domain"].split("_")[0]
    pol_direction = row["source_domain"].split("_")[1]
    doc_id_full = row["source_domain"]
    doc_unique_id = doc_id_full+"_"+str(i)

    #count the amount of tokens for each general topic (later used for unique head lemmas, maybe legacy)
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
                fitting_tokens = []
                fitting_tokens_docID = []
                fitting_token_str = []
                seg_token = 0   #a counter to show with which token in the tokenized segment to match
                iterations_without_token_match = 0      #a counter to account for +-1 differences in token match
            
                #tokenize the segment
                segment_doc = nlp(row_anno["Segment"])
                segment_tokenized = []
                for t in segment_doc:
                    segment_tokenized.append(t)

                #iterate over every token of the sentence
                for token in sentence:
                    #while iterating check for a +-1 similarity of segment and sentence tokens

                    if len(segment_tokenized) <= seg_token:     #if the segment tokenized has been fully discovered within the sentence
                        if seg_token >= 1:
                            sent_id = j
                            text = row["text"]
                            code = row_anno["Code"].replace("\\", " ")
                            segment = row_anno["Segment"]
                            score = -1  #not row_anno["Weight score"], not that important (legacy)
                            is_continuous = checkContinuous(tokens_number[:])
                            is_singleton = False    #placeholder

                            #determine the head of the mention tokens
                            for t_id in fitting_tokens_docID:
                                ancestors_in_mention = 0
                                for a in doc[t_id].ancestors:
                                    if a.i in fitting_tokens_docID:
                                        ancestors_in_mention = ancestors_in_mention + 1
                                        break   #one is enough to make the token unviable as a head
                                if ancestors_in_mention == 0:   
                                    #head within the mention if the token itself has no ancestors
                                    mention_head = doc[t_id]

                            #get attributes based on the found token head
                            head_pos = mention_head.pos_
                            head_lemma = mention_head.lemma_
                            head_str = mention_head.text

                            #generate an unique id for each mention
                            mention_id = doc_unique_id+"_"+str(sent_id)+"_"+str(mention_head.i)

                            #just set STRICT
                            coref_type = "STRICT"

                            #get the mentions context (range of tokens around the mentions)
                            mention_context_str = get_context(doc, fitting_tokens_docID)
                            
                            #get the NER type for the head
                            mention_ner = mention_head.ent_type_
                            if mention_ner == "":
                                mention_ner = "O"

                            coref_chain = row_anno["coref_chain"]
                            mention_type = row_anno["type"]
                            
                            if iterations_without_token_match >= 1:     #print a warning if the found mention does not matchh +-0, but +-1 
                                print("Warning: the segment does not match 100%: " + segment)
                                print("Sentence: " + str(sentence))

                            #append the mention to the dataframe
                            df_preparation_list.append([coref_chain, code, segment, tokens_amount, mention_ner, head_pos, head_lemma, mention_head.text, code, doc_unique_id, doc_id, pol_direction, is_continuous, is_singleton, text, sentence.text, mention_id, mention_type, mention_type, score, sent_id, mention_context_str, fitting_tokens, fitting_token_str, row_anno["Segment"], filename.split("_")[0], coref_type, code, head_str, mention_head.i])
                            
                            #break   #only one mention of the same kind per sentence is valid (can be commented out or in based on what is desired)

                        #reset counters and so on
                        fitting_tokens = []
                        fitting_tokens_docID = []
                        fitting_token_str = []
                        seg_token = 0

                    if token.text == segment_tokenized[seg_token].text:    #if there has been discovered a token match for a specific token ID
                        if seg_token == 0:
                            iterations_without_token_match = 0
                        seg_token = seg_token + 1
                        fitting_tokens.append(token.i - sentence.start)     #the id of the token within a sentence (not in doc)
                        fitting_tokens_docID.append(token.i)    #the id of the token within a whole doc
                        fitting_token_str.append(token.text)    #the string for the token

                        tokens = []
                        tokens_str = ""
                        tokens_number = []
                        for t in token.head.subtree:    #append the head tokens subtree as tokens_str for the mention
                            tokens.append(t)
                            tokens_str = tokens_str + " " + t.text
                            tokens_number.append(t.i)
                    
                    else:   #the token that is iterated over does not match with the segment token we are looking for  
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
                            #continue looking within the sentence for the next iterations/tokens. But compare sentence tokens with the first segment token again
    #if i > 3:
    #    break

#append the created mention list to the mentions dataframe
mentions_df = pd.DataFrame(df_preparation_list, columns=["coref_chain", "code", "segment", "tokens_amount", "mention_ner", "mention_head_pos", "mention_head_lemma", "mention_head", "coref_link", "doc_id_full","doc_id", "pol_direction", "is_continuous", "is_singleton", "text", "sentence", "mention_id", "mention_type", "mention_full_type", "score", "sent_id", "mention_context", "tokens_number", "fitting_tokens", "tokens_str", "topic_id", "coref_type", "description", "head_str", "mention_head_id"])

#determine average number of unique head lemmas within a cluster (may not work, legacy?)
#(excluding singletons for fair comparison)
mentions_df = mentions_df.join(mentions_df.groupby(['topic_id', 'segment'])["mention_head_lemma"].nunique(), on=['topic_id', 'segment'], rsuffix='_uniques')
#remove rows from dataframe that are duplicates
mentions_df = mentions_df.drop_duplicates(subset=['doc_id_full', 'sent_id', 'mention_head_id', 'coref_chain'], keep='first')  

#determine singletons
for i, row in tqdm(mentions_df.iterrows()):
    amountOfMentions = len( mentions_df[ mentions_df.coref_chain == row["coref_chain"] ] )
    if amountOfMentions == 1:
        mentions_df.at[i, "is_singleton"] = True

print("Singletons: " + str(len(mentions_df[ mentions_df.is_singleton == True ])))
print("Amount of mentions: " + str(len(mentions_df))) 

#drop unwanted columns for the output
mentions_df = mentions_df.drop(["segment", "text", "sentence", "fitting_tokens", "head_str"], axis = 1)
mentions_df.reset_index(drop=True,inplace=True)

#output all mentions into json
with open(OUT_DIR +'mentions_df.json', 'w', encoding='utf-8') as f:
    f.write(ujson.dumps(mentions_df.to_dict('index'), indent=4, ensure_ascii=False, escape_forward_slashes=False))

#define entities and event mentions to generate two different json files
events = ["ACTION", "EVENT", "MISC"]
df_entities = pd.DataFrame()
df_events = pd.DataFrame()
for i, row in mentions_df.iterrows():
    if row["mention_type"] not in events:
        df_entities = df_entities.append(row)
    else:
        df_events = df_events.append(row)

#drop columns for output
df_entities.drop(columns= ["tokens_amount", "mention_head_lemma_uniques"], inplace = True)
df_events.drop(columns= ["tokens_amount", "mention_head_lemma_uniques"], inplace = True)

#output entity and event mentions
with open(OUT_DIR +'entity_mentions.json', 'w', encoding='utf-8') as f:
    f.write(ujson.dumps(df_entities.to_dict('records'), indent=4, ensure_ascii=False, escape_forward_slashes=False))
with open(OUT_DIR +'event_mentions.json', 'w', encoding='utf-8') as f:
    f.write(ujson.dumps(df_events.to_dict('records'), indent=4, ensure_ascii=False, escape_forward_slashes=False))

#also make a all_mentions csv file
df_all_mentions = mentions_df.drop(columns=["is_continuous", "is_singleton", "score", "sent_id", "tokens_amount", "tokens_number", "topic_id", "coref_type", "mention_context", "mention_ner", "mention_head_pos", "mention_head_lemma", "coref_link", "tokens_amount", "code", "mention_head", "mention_head_lemma_uniques"]).rename(index = mentions_df.mention_id)
df_all_mentions = df_all_mentions.drop(columns = ["mention_id"])
df_all_mentions.to_csv(path_or_buf=OUT_DIR +"all_mentions.csv", sep=",", na_rep="")

#generate conll file
print("Generating conll...")
df_conll = pd.DataFrame(columns = {"topic/subtopic_name", "doc_identifier", "sent_id", "token_id", "token", "reference"})

#first create a conll dataframe that gets a new row for each token
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

#then annotate each token (i.e. row) in the conll df with the coref_chain id
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


#output conll data as json file for the dataset_summary to handle more easily
df_conll.reset_index(drop=True,inplace=True)
with open(os.path.join(OUT_DIR, "conll_as_json" + ".json"), "w", encoding='utf-8') as file:
    json.dump(df_conll.to_dict('records'), file)

#Make the conll file string from the dataframe
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
print("Checking equal brackets in conll (if unequal, the result may be incorrect):")
print("(: " + str(dfAsString.count("(")))
print("): " + str(dfAsString.count(")")))


print("Done.")