# CDCR Benchmark: Cross-document coreference resolution datasets with diverse annotation schemes

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
2) Recommended to create a venv.
3) Install libraries: ```pip install -r requirements.txt```
4) Download the datasets and required libraries from spacy: ```python setup.py```
5) Download and install [Perl](https://strawberryperl.com/). Add perl to PATH, restart your computer, and check that perl has been correctly installed.

## Dataset information 

The parsing scripts and output folders are located  here:
[Bugert et al. 2021](https://direct.mit.edu/coli/article/47/3/575/102774/Generalizing-Cross-Document-Event-Coreference)

| Dataset                                    | Coreference target      | Parsing script                          | Available versions  | Train/val/test splits           | Previous benchmark                                                                                                      |
|:-------------------------------------------|-------------------------|:----------------------------------------|---------------------|---------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| [ECB+](ECBplus-prep/README.md)             | (mainly) event + entity | ```ECBplus-prep/parse_ecbplus.py```     | public & protected  | previously introduced & reused  | [Bugert et al. 2021](https://direct.mit.edu/coli/article/47/3/575/102774/Generalizing-Cross-Document-Event-Coreference) |
| [ECB+ unvalidated](ECBplus-prep/README.md) | (mainly) event + entity | ```ECBplus-prep/parse_ecbplus.py```     | public & protected  | previously introduced & reused  | -                                                                                                                       |
| [FCC](FCC-prep/README.md)                  | event                   | ```FCC-prep/parse_fcc.py```             | protected           | original & reused               | [Bugert et al. 2021](https://direct.mit.edu/coli/article/47/3/575/102774/Generalizing-Cross-Document-Event-Coreference) |
| [FCC-T](FCC-prep/README.md)                | event                   | ```FCC-prep/parse_fcc.py```             | protected           | original & reused               | [Bugert et al. 2021](https://direct.mit.edu/coli/article/47/3/575/102774/Generalizing-Cross-Document-Event-Coreference) |
| [GVC](GVC-prep/README.md)                  | event                   | ```GVC-prep/parse_gvc.py```             | public & protected  | previously introduced & reused  | [Bugert et al. 2021](https://direct.mit.edu/coli/article/47/3/575/102774/Generalizing-Cross-Document-Event-Coreference) |
| [WEC-Eng](WECEng-prep/README.md)           | event                   | ```WECEng-prep/parse_weceng.py```       | public & protected  | original & reused               | -                                                                                                                       |
| [CD2CR](CD2CR-prep/README.md)              | entity                  | ```CD2CR-prep/parse_cd2cr.py```         | public & protected  | original & reused               | -                                                                                                                       |
| [NewsWCL50](NewsWCL50-prep/README.md)      | event + entity          | ```NewsWCL50-prep/parse_newswcl50.py``` | protected           | new (didn't exist before)       | -                                                                                                                       |
| [MEANTIME](MEANTIME-prep/README.md)        | event + entity          | ```MEANTIME-prep/parse_meantime.py```   | public & protected  | new (didn't exist before)       | -                                                                                                                       |
| [NP4E](NP4E-prep/README.md)                | entity                  | ```NP4E-prep/parse_np4e.py```           | public & protected  | new (didn't exist before)       | -                                                                                                                       |
| [NiDENT](NiDENT-prep/README.md)            | entity                  | ```NiDENT-prep/parse_nident.py```       | protected           | new (didn't exist before)       | -                                                                                                                       |

Each dataset contains three **main** output files suitable for a CDCR model: 
1) ```dataset.conll```, i.e., a CoNLL format of the full text corpus with the beginning and end tags, with the newline delimiters between the articles.
2) ```entity_mentions.json```, i.e., a list of entity mentions with assigned coreference chain IDs. 
3) ```event_mentions.json```, i.e., a list of event mentions with assigned coreference chain IDs.

Same data in the csv format (used for data analysis, e.g., to compute statistics of the datasets or have an overview of the mentions):
1) ```conll.csv```, i.e., a CoNLL format in a tabular format without tags and newline delimiters.
2) ```all_mentions.csv```, i.e., a csv file with all mentions combined.

More details about the formats of these formats see below. 

### Train/val/test subsets
If each dataset contains **train/ val/ test** splits, which are either reused from the previous research or newly introduced, if not existed before. 
The ```output_data``` folder contain corresponding subfolders with a set of files defined above.

Each subfolder contains one ```entity_mentions.json``` and one ```event_mentions.json``` file, but ```dataset.conll``` can 
be split into multiple parts, e.g., ```dataset_1.conll``` and ```dataset_2.conll```. 

### CDCR topic structure
The articles are organized in the following structure: 
```
- topic
    - subtopic
        - document
```
 **Topic** contains text documents about the same topic, e.g., presidential elections.
**Subtopic** further organized the documents into _event-specific_ more narrowly related events, e.g., presidential elections in the U.S. in 2018. 
**Document** is a specific text, e.g., a news article. 

The composition of these attributes as ```topic_id/subtopic_id/doc_id``` will be used as a unique identifier within a dataset. 
To make the identifier unique across the datasets, e.g., to distinguish between topics with topic_id = 0, 
modify the key into ```dataset/topic_id/subtopic_id/doc_id```.

If a dataset contains only subtopics, but they are all related to one topic, e.g., football, then they are organized under one topic. 
If a dataset contains multiple subtopics but they do not share same topics, then, for each subtopic, separate topics are artificially created.  


### Simple use case: binary classification model
To train a simple binary classification mentions, one requires only ```entity_mentions.json``` and ```event_mentions.json``` files. 
Each file contains a list of mentions. To encode a mention, you need to use the following attributes: 
1) ```mention_context``` with a list of tokens within which a mention occurs
2) ```tokens_number_context``` with a list of indexed where a mention occurs in the ```mention_context```, which are needed to position the mention 
3) ```coref_chain``` that indicates if two mentions are coreferencial if the value is identical between two mentions

Similar to [Cattan et al. 2021](https://aclanthology.org/2021.findings-acl.453/) or [Caciularu et al. 2021](https://aclanthology.org/2021.findings-emnlp.225/), 
a pair of mentions can be encoded within their contexts and a coroference chain sets a training objective. 

For more information about the format and attributes, see below. 

## Input formats
### 1) (simplified) CoNLL format: Full document texts & annotations

CoNLL format is a standard input format for within-document [coreference resolution](https://paperswithcode.com/dataset/conll-2012-1). 
The original format contains multiple columns that contain information per each token, e.g., POS tags, NER labels. 
We use a simplified format (based on the format of input filed used by [Barhom et al. 2019](https://github.com/shanybar/event_entity_coref_ecb_plus/tree/master/data/interim/cybulska_setup)) 
that contains tokens, their identifiers in the text (e.g., doc_id, sent_id), and labels of coref chains: 

| Column ID |  Type  | Description                                       |
|:----------|:------:|:--------------------------------------------------|
| 0         | string | Composed document id: topic_id/subtopic_id/doc_id |
| 1         |  int   | Sentence ID                                       |
| 2         |  int   | Token ID                                          |
| 3         | string | Token                                             |
| 4         | string | Reference labels, i.e., coreference chain         |

Each document is accompanied with a beginning and end tags, sentences are separated with news lines (warning: some new line delimiters can be tokens themselves (e.g., in NewsWCL50)). 

Example: 
```
#begin document 0/0/0_LL; part 000
0/0/0_LL 0 0 This -
0/0/0_LL 0 1 is -
0/0/0_LL 0 2 Jim (1
0/0/0_LL 0 3 Jones -
0/0/0_LL 0 4 , -
0/0/0_LL 0 5 a -
0/0/0_LL 0 6 police (2)
0/0/0_LL 0 7 officer 1)
0/0/0_LL 0 8 . -

0/0/0_LL 1 0 He (1)
0/0/0_LL 1 1 likes - 
0/0/0_LL 1 2 sports - 
0/0/0_LL 1 3 . -

#end document
#begin document 1/1ecb/12; part 000
1/1ecb/12 0 0 This -
1/1ecb/12 0 1 is -
1/1ecb/12 0 2 Anna (3
1/1ecb/12 0 3 Maria -
1/1ecb/12 0 4 Stevens 3)
1/1ecb/12 0 5 . -

1/1ecb/12 1 0 She (3)
1/1ecb/12 1 1 likes - 
1/1ecb/12 1 2 singing - 
1/1ecb/12 1 3 . -

#end document
```


### 2) ***_mentions.json: Annotations and mention context
The format is adapted and extended from [WEC-Eng](https://huggingface.co/datasets/Intel/WEC-Eng) and from the mention format used by [Barhom et al. 2019](https://github.com/shanybar/event_entity_coref_ecb_plus/tree/master/data/interim/cybulska_setup). 

To extract some mentions' attributes, we parse document sentences by spaCy. To extract a mention head, we align each mention 
to the corresponding sentences in the documents and extract the head of mention as highest node in the dependency subtree.

| Field                | Type            | Description                                                                                         |
|----------------------|-----------------|-----------------------------------------------------------------------------------------------------|
| coref_chain          | string          | Unique identifier of a coreference chain to which this mention belongs to.                          |
| description          | string          | Description of a coreference chain.                                                                 |
| coref_type           | string          | Type of a coreference link, e.g., strict indentity.                                                 |
| mention_id           | string          | Mention ID.                                                                                         |
| mention_type         | string          | Short form of a mention type, e.g., HUM                                                             |
| mention_full_type    | string          | Long form of a mention type, e.g., HUMAN_PART_PER                                                   |
| tokens_str           | string          | A full mention string, i.e., all consequitive chars of the mention as found in the text.            |
| tokens_text          | list of strings | A mention split into a list of tokens, text of tokens                                               |
| tokens_number        | list of int     | A mention split into a list of tokens, token id of these tokens (as occurred in a sentence).        |
| mention_head         | string          | A head of mention's phrase, e.g., Barack *Obama*                                                    |
| mention_head_id      | int             | Token id of the head of mention's phrase                                                            |
| mention_head_pos     | string          | Token's POS tag of the head of mention's phrase                                                     |
| mention_head_lemma   | string          | Token's lemma of the head of mention's phrase                                                       |
| sent_id              | int             | Sentence ID                                                                                         |
| topic_id             | string          | Topic ID                                                                                            |
| topic                | string          | Topic ID with its description (if any)                                                              |
| subtopic_id          | string          | Subtopic id (optionally with short name)                                                            |
| subtopic             | string          | Subtopic ID with its description (if any)                                                           |
| doc_id               | string          | Document ID                                                                                         |
| doc                  | string          | Document ID with its description (if any)                                                           |
| is_continuous        | bool            | If all tokens in the annotated mention continuously occur in the text                               |
| is_singleton         | bool            | If a coreference chain consists of only one mention.                                                |
| mention_context      | list of strings | -N and +N tokens within one document before and after the mention (N=100).                          |
| tokens_number_context | list of int     | Positioning of the mention's tokens within the context.                                             |
| language             | string          | Optional. A language of the mention. If not provided, the default value will be considered english. |
| conll_doc_key        | string          | a compositional key for one-to-one mapping documents between .conll and .json files.                |

Example: 
```json
[
  {
    "coref_chain": "2293469", 
    "mention_ner": "O", 
    "mention_head_pos": "PROPN", 
    "mention_head_lemma": "Tour", 
    "mention_head": "Tour", 
    "mention_head_id": 193, 
    "doc_id": "Wd36WuWE3hRzmH2hRTpdgy", 
    "doc": "Ice Cube", 
    "is_continuous": true, 
    "is_singleton": false, 
    "mention_id": "108173", 
    "mention_type": "EVE",
    "mention_full_type": "EVENT", 
    "score": -1.0, 
    "sent_id": 0, 
    "mention_context": ["for", "the", "''", "Murder", "Was", "The", "Case", "''", "soundtrack", ",", "and", "also", "contributed", "to", "the", "''", "Office", "Space", "''", "soundtrack", ".", "He", "also", "featured", "on", "Kool", "G", "Rap", "'s", "song", "\"", "Two", "To", "The", "Head", "\"", "from", "the", "Kool", "G", "Rap", "&", "DJ", "Polo", "album", "\"", "Live", "And", "Let", "Die", "\"", ".", "He", "also", "collaborated", "with", "David", "Bowie", "and", "Trent", "Reznor", "from", "Nine", "Inch", "Nails", "for", "a", "remix", "of", "Bowie", "'s", "\"", "I", "'m", "Afraid", "of", "Americans", "\"", ".", "Ice", "Cube", "appeared", "on", "the", "song", "\"", "Children", "of", "the", "Korn", "\"", "by", "the", "band", "Korn", ",", "joining", "them", "on", "the", "Family", "Values", "Tour", "1998", ",", "and", "they", "also", "collaborated", "on", "'", "Fuck", "Dying", "'", "from", "Cube", "'s", "fifth", "album", ".", "He", "also", "lent", "his", "voice", "to", "British", "DJ", "Paul", "Oakenfold", "'s", "solo", "debut", "album", ",", "''", "Bunkka", "''", ",", "on", "the", "track", "\"", "Get", "Em", "Up", "\"", ".", "Ice", "Cube", "appeared", "in", "several", "songs", "in", "WC", "Guilty", "by", "Affiliation", "like", "\"", "Keep", "it", "100", "\"", ",", "\"", "80", "'s", "babies", "\"", "and", "\"", "Jack", "and", "the", "bean", "stalk", "\"", ".", "Ice", "Cube", "also", "appeared", "in", "D.A.Z.", "in", "the", "song", "\"", "Iz", "You", "Ready", "to", "die", "\"", "and", "in", "DJ", "Quik"], 
    "tokens_number_context": [100, 101, 102, 103], 
    "tokens_number": [191, 192, 193, 194], 
    "tokens_str": "Family Values Tour 1998", 
    "tokens_text": ["Family", "Values", "Tour", "1998"], 
    "topic_id": "4", 
    "topic": "Concert", 
    "subtopic_id": "4A4Hs78SQWhihhpMAZt65s", 
    "subtopic": "Family Values Tour 1998", 
    "coref_type": "IDENTITY", 
    "description": "Family Values Tour 1998", 
    "conll_doc_key": "4/4A4Hs78SQWhihhpMAZt65s/Wd36WuWE3hRzmH2hRTpdgy"
  }
]
```


## Dataset comparison

The following values enable comparison of the CDCR datasets on dataset, topic+subtopic, and language (optional) levels.    

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
