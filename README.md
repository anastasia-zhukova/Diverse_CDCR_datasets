# Cross-document coreference resolution (CDCR) datasets with diverse annotation schemes

The repository contains the code used to report the results in the [LREC 2022 paper Zhukova A., Hamborg F., Gipp B. "Towards Evaluation of Cross-document Coreference Resolution Models Using Datasets with Diverse Annotation Schemes"](https://aclanthology.org/2022.lrec-1.522/).  
Please use this .bib to cite the paper:
```
@inproceedings{Zhukova22a,
    title = "Towards Evaluation of Cross-document Coreference Resolution Models Using Datasets with Diverse Annotation Schemes",
    author = "Zhukova, Anastasia  and
      Hamborg, Felix  and
      Gipp, Bela",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.522",
    pages = "4884--4893",
    abstract = "Established cross-document coreference resolution (CDCR) datasets contain event-centric coreference chains of events and entities with identity relations. These datasets establish strict definitions of the coreference relations across related tests but typically ignore anaphora with more vague context-dependent loose coreference relations. In this paper, we qualitatively and quantitatively compare the annotation schemes of ECB+, a CDCR dataset with identity coreference relations, and NewsWCL50, a CDCR dataset with a mix of loose context-dependent and strict coreference relations. We propose a phrasing diversity metric (PD) that encounters for the diversity of full phrases unlike the previously proposed metrics and allows to evaluate lexical diversity of the CDCR datasets in a higher precision. The analysis shows that coreference chains of NewsWCL50 are more lexically diverse than those of ECB+ but annotating of NewsWCL50 leads to the lower inter-coder reliability. We discuss the different tasks that both CDCR datasets create for the CDCR models, i.e., lexical disambiguation and lexical diversity. Finally, to ensure generalizability of the CDCR models, we propose a direction for CDCR evaluation that combines CDCR datasets with multiple annotation schemes that focus of various properties of the coreference chains.",
}
```

The repository contains a code that parses original formats of CDCR datasets into the same format (conll format for coreference resolution and a separate list of mentions) and calculates summary values that enable comparison of the datasets.

Parsing scripts per dataset are contained in each separate folder, whereas the summary script is located in the root folder. The parsed datasets are available in this repository in the folders listed below. 

## Installation 

1) __Python 3.8 required__

2) !!! Recommended to create a venv.
3) Install libraries: ```pip install -r requirements.txt```
4) Download the datasets and required libraries from spacy: ```python setup.py```
5) Download and install [Perl](https://strawberryperl.com/). Add perl to PATH, restart your computer, and check that perl has been correctly installed.

## Dataset information 

The parsing scripts and output folders are located  here:

| Dataset   | Parsing script                          | Output files                     |
|:----------|:----------------------------------------|:---------------------------------|
| ECB+      | ```ECBplus-prep/parse_ecbplus.py```     | ```ECBplus-prep/output_data```   |
| NewsWCL50 | ```NewsWCL50-prep/parse_newswcl50.py``` | ```NewsWCL50-prep/output_data``` |
| MEANTIME  | ```MEANTIME-prep/parse_meantime.py```   | ```MEANTIME-prep/output_data```  |
| NP4E      | ```NP4E-prep/parse_np4e.py```           | ```NP4E-prep/output_data```      |

Each dataset contains three output files suitable for a CDCR model: 

Main files:
1) ```dataset.conll```, i.e., a CoNLL format of the full text corpus with the beginning and end tags, with the newline delimiters between the articles.
2) ```entity_mentions.json```, i.e., a list of entity mentions. 
3) ```event_mentions.json```, i.e., a list of event mentions.

Same data in the csv format (used to compute statistics of the datasets):
1) ```conll.csv```, i.e., a CoNLL format in a tabular format without tags and newline delimiters.
2) ```all_mentions.csv```, i.e., a csv file with all mentions combined.

If a dataset has **a predefined split into train/dev/test parts**, the ```output_data``` folder will additionally contain corresponding subfolders with a set of files defined above. 


## CDCR structure
The articles are organized in the following structure: 
```
- topic
    - subtopic
        - document
```
 **Topic** contains text documents about the same topic, e.g., presidential elections.
**Subtopic** further organized the documents into _event-specific_ more narrowly related folders, e.g., (rather broad event) presidential elections in the U.S. in 2018. 
**Document** is a specific text, e.g., a news article. 

The composion of these attributes as ```topic_id/subtopic_id/doc_id``` will be used as a unique document key within a dataset. 
To make a document unique across the datasets, modify the key into ```dataset/topic/subtopic/document```.

If a dataset contains only subtopics, but they are all related to one topic, e.g., football, then they are organized under one topic. 
If a dataset contains multiple subtopics but they do not share same topics, then for each subtopic separate topics are artificially created.  

## CoNLL format (simplified)

CoNLL format is a standard input format for within-document [coreference resolution](https://paperswithcode.com/dataset/conll-2012-1). The original format contains multiple columns that contain information per each token, e.g., POS tags, NER labels. We use a simplified format (based on the format of input filed used by [Barhom et al. 2019](https://github.com/shanybar/event_entity_coref_ecb_plus/tree/master/data/interim/cybulska_setup)) that contains tokens, their identifiers in the text (e.g., doc_id, sent_id), and labels of coref chains: 

| Column ID |  Type  | Description                               |
|:----------|:------:|:------------------------------------------|
| 0         | string | Composed document id: topic/subtopic/doc  |
| 1         |  int   | Sentence ID                               |
| 2         |  int   | Token ID                                  |
| 3         | string | Token                                     |
| 4         | string | Coreference chain                         |

Each document is accompanied with a beginning and end tags, sentences are separated with news lines (warning: some new line delimiters can be tokens themselves (e.g., in NewsWCL50)). 

Example: 
```
#begin document 0/0/0_LL; part 000
0/0/0_LL 0 0 This -
0/0/0_LL 0 1 is -
0/0/0_LL 0 2 Jim (1)
0/0/0_LL 0 3 . -

0/0/0_LL 1 0 He (1)
0/0/0_LL 1 1 likes - 
0/0/0_LL 1 2 sports - 
0/0/0_LL 1 3 . -

#end document
#begin document 1/1ecb/12; part 000
1/1ecb/12 0 0 This -
1/1ecb/12 0 1 is -
1/1ecb/12 0 2 Anna (2)
1/1ecb/12 0 3 . -

1/1ecb/12 1 0 She (2)
1/1ecb/12 1 1 likes - 
1/1ecb/12 1 2 singing - 
1/1ecb/12 1 3 . -

#end document
```


## Mentions.json
The format is adapted and extended from [WEC-Eng](https://huggingface.co/datasets/Intel/WEC-Eng) and from the mention format used by [Barhom et al. 2019](https://github.com/shanybar/event_entity_coref_ecb_plus/tree/master/data/interim/cybulska_setup). 

| Field              | Type            | Description                                                                                         |
|--------------------|-----------------|-----------------------------------------------------------------------------------------------------|
| coref_chain        | string          | Unique identifier of a coreference chain to which this mention belongs to.                          |
| description        | string          | Description of a coreference chain.                                                                 |
| coref_type         | string          | Type of a coreference link, e.g., strict indentity.                                                 |
| mention_id         | string          | Mention ID.                                                                                         |
| mention_type       | string          | Short form of a mention type, e.g., HUM                                                             |
| mention_full_type  | string          | Long form of a mention type, e.g., HUMAN_PART_PER                                                   |
| tokens_str         | string          | A full mention string, i.e., all consequitive chars of the mention as found in the text.            |
| tokens_text        | list of strings | A mention split into a list of tokens, text of tokens                                               |
| tokens_numbers     | list of int     | A mention split into a list of tokens, token id of these tokens (as occurred in a sentence).        |
| mention_head       | string          | A head of mention's phrase, e.g., Barack *Obama*                                                    |
| mention_head_id    | int             | Token id of the head of mention's phrase                                                            |
| mention_head_pos   | string          | Token's POS tag of the head of mention's phrase                                                     |
| mention_head_lemma | string          | Token's lemma of the head of mention's phrase                                                       |
| sent_id            | int             | Sentence ID                                                                                         |
| topic_id           | string          | Topic ID                                                                                            |
| topic              | string          | Topic ID with its description (if any)                                                              |
| subtopic_id        | string          | Subtopic id (optionally with short name)                                                            |
| subtopic           | string          | Subtopic ID with its description (if any)                                                           |
| doc_id             | string          | Document ID                                                                                         |
| doc                | string          | Document ID with its description (if any)                                                           |
| is_continuous      | bool            | If all tokens in the annotated mention continuously occur in the text                               |
| is_singleton       | bool            | If a coreference chain consists of only one mention.                                                |
| mention_context    | list of strings | -N and +N tokens before and after the mention (N=100).                                              |
| language           | string          | Optional. A language of the mention. If not provided, the default value will be considered english. |
| conll_doc_key      | string          | a compositional key for one-to-one mapping documents between .conll and .json files.                |

Example: 
```
[{
    "coref_chain": "0_Denuclearization_MISC", 
    "tokens_number": [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49], 
    "score": -1, 
    "sent_id": 21, 
    "mention_type": "MISC", 
    "mention_full_type": "MISC", 
    "mention_id": "0_L_21_49_VZrL", 
    "topic_id": "0", 
    "topic": "0", 
    "subtopic_id": "0",
    "subtopic": "0_CIADirectorMikePompeoMeetingNorthKorea", 
    "doc_id": "0_L", 
    "doc": "0_L", 
    "description": "Denuclearization", 
    "coref_type": "IDENTITY", 
    "mention_ner": "O", 
    "mention_head_pos": "PUNCT", 
    "mention_head_lemma": "\"", 
    "mention_head": "\"", 
    "mention_head_id": 49, 
    "is_continuous": true, 
    "is_singleton": false, 
    "mention_context": ["newspaper", ",", "Munhwa", "Ilbo", ",", "reported", "that", "the", "two", "countries", "were", "negotiating", "an", "announcement", "\"", "to", "ease", "military", "tensions", "and", "end", "a", "military", "confrontation", ",", "\"", "as", "part", "of", "the", "summit", "meeting", "planned", "between", "Mr.", "Kim", "and", "President", "Moon", "Jae", "-", "in", "of", "South", "Korea", ".", "\n", "That", "could", "involve", "pulling", "troops", "out", "of", "the", "Demilitarized", "Zone", ",", "making", "it", "a", "genuinely", "\"", "Demilitarized", "Zone", ".", "\"", "A", "South", "Korean", "government", "official", "later", "played", "down", "the", "report", ",", "saying", "it", "was", "too", "soon", "to", "tell", "what", "a", "joint", "statement", "by", "Mr.", "Moon", "and", "Mr.", "Kim", "would", "contain", ",", "other", "than", "broad", "and", "\"", "abstract", "\"", "statements", "about", "the", "need", "for", "North", "Korea", "to", "\"", "denuclearize", ".", "\"", "\n", "But", "analysts", "said", "South", "Korea", "was", "aiming", "for", "a", "comprehensive", "deal", ",", "in", "which", "the", "North", "agreed", "to", "give", "up", "its", "weapons", "in", "return", "for", "a", "security", "guarantee", ",", "including", "a", "peace", "treaty", ".", "Mr.", "Trump", "'s", "comments", "suggested", "he", "backed", "that", "effort", ".", "\n", "\"", "They", "do", "have", "my", "blessing", "to", "discuss", "the", "end", "of", "the", "war", ",", "\"", "he", "said", ".", "\"", "People", "do", "n't", "realize", "that", "the", "Korean", "War", "has", "not", "ended", ".", "It", "'s", "going", "on", "right", "now", ".", "And", "they", "are", "discussing", "an", "end", "to", "war", ".", "Subject", "to", "a", "deal", ",", "they"], 
    "tokens_str": "broad and \"abstract\" statements about the need for North Korea to \"denuclearize.\" ", 
    "tokens_text": ["broad", "and", "\"", "abstract", "\"", "statements", "about", "the", "need", "for", "North", "Korea", "to", "\"", "denuclearize", ".", "\""], 
    "language": "english", 
    "conll_doc_key": "0/0/0_L"
}]
```


## Dataset summary metrics

The following values enable comparison of the CDCR datasets on dataset and topic levels.    

| Field                                | Type     | Description                                                                                                                                                                               |
| :---                                 | :----:   |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dataset                              | string   | Name of the dataset                                                                                                                                                                       |
| topic                                | string   | Topic name (or empty for the line that contains stats for a full dataset)                                                                                                                 |
| articles                             | int      | Number of articles in a dataset/topic                                                                                                                                                     |
| tokens                               | int      | Number of tokens in a dataset/topic                                                                                                                                                       |
| coref_chain                          | int      | Number of coref chains in a dataset/topic                                                                                                                                                 |
| mentions                             | int      | Number of all mentions in a dataset/topic                                                                                                                                                 |
| event_mentions                       | int      | Number of event mentions in a dataset/topic                                                                                                                                               |
| entity_mentions                      | int      | Number of entity mentions in a dataset/topic                                                                                                                                              |
| singletons                           | int      | Number of singleton coref chains in a dataset/topic                                                                                                                                       |
| average_size                         | float    | Average number of mentions in a coref chain, i.e., chain size                                                                                                                             |
| unique_lemmas_all                    | float    | Lexical diversity measurement: a number of unique mention lemmas in a chain. Calculated on all coref chains.                                                                              |
| unique_lemmas_wo_singl               | float    | -//- Calculated on non-singleton chains.                                                                                                                                                  |
| phrasing_diversity_weighted_all      | float    | Lexical diversity measurement: phrasing diversity (see LREC paper). Measures diversity of the mentions given variation and frequency of the chains' mentions. Calculated on all mentions. |
| phrasing_diversity_weighted_wo_singl | float    | -//- Calculated on non-singleton chains.                                                                                                                                                  |
| F1_CONLL_all                         | float    | F1 CoNLL (average of B3, MUC, and CEAF_e) calculated on the simple same-lemma baseline. Calculated on all coref chains.                                                                   |
| F1_CONLL_wo_singl                    | float    | -//- Calculated on non-singleton chains.                                                                                                                                                  |

The results of dataset comparison is available in ```/summary``` folder.
