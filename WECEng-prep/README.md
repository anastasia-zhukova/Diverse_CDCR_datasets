# Parsing WEC-Eng

Original repository: https://huggingface.co/datasets/Intel/WEC-Eng. 

### Paper 
https://aclanthology.org/2021.naacl-main.198/

```
@inproceedings{eirew-etal-2021-wec,
    title = "{WEC}: Deriving a Large-scale Cross-document Event Coreference dataset from {W}ikipedia",
    author = "Eirew, Alon  and
      Cattan, Arie  and
      Dagan, Ido",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.198",
    doi = "10.18653/v1/2021.naacl-main.198",
    pages = "2498--2510"
}
```

### To parse WEC-Eng 
1) make sure that you downloaded the dataset by running ```python setup.py``` 
2) execute ```python weceng.py```

Take topic names from Table 7 of the WEC paper and reconstruct the assignment of the subtopics (Wikipedia titles of the 
coreference chains which acts as seed/seminal events) to the topics. We create a heuristic of parsing Wikipedia categories 
of the subtopic Wiki pages and overlapping them with the topics. Afterwards, we performed a manual validation. The assigned 
topics to the subtopics are located in the ```WEC-Eng/topics``` folder. 

Since one documents could contain mentions referring to multiple chains, i.e., belong to multiple seminal events, we need 
ensure that a document has a unique assignment to subtopics. Each document is uniquely assigned to the first subtopic 
that occurs in each topic file in ```WEC-Eng/topics``` folder. Such a setup makes possible cross-subtopic coreferences, 
which is a similar setup to MEANTIME and FCC. 

To create CoNLL files, we reconstructed the Wikipedia articles by merging articles' paragraphs from mentions' context. 
Since the sentence split was not provided, we assign one sentence_id per paragraph, i.e., mention's context. 

To match the logic of the other parsed dataset, we modify mention's context into a window of -+ N (N=100) words before and after 
mention's first token.

The size of the CoNLL file of the train split was quite big, so we split it into multiple parts. But event mentions remained in 
one file per split. A file with entity mentions is empty because the dataset focused on event CDCR. 


### Topic organization
News articles in the dataset are organized as following:
```
-> topic 
    -> subtopic (coref chain name which represents seminal/seed event)
        -> documents (wikipedia articles)
   ``` 

### Download WEC-Eng 
The re-parsed WEC-Eng can be downloaded from [this link](https://drive.google.com/drive/folders/1W8yilLEJo0_HDbIfJ4oWcl70t39tSnWG?usp=drive_link).

____________________________

# WEC-Eng
A large-scale dataset for cross-document event coreference extracted from English Wikipedia. </br>

- **Repository (Code for generating WEC):** https://github.com/AlonEirew/extract-wec
- **Paper:** https://aclanthology.org/2021.naacl-main.198/

### Languages

English

## Load Dataset
You can read in WEC-Eng files as follows (using the **huggingface_hub** library):

```json
from huggingface_hub import hf_hub_url, cached_download
import json
REPO_ID = "datasets/Intel/WEC-Eng"
splits_files = ["Dev_Event_gold_mentions_validated.json",
                "Test_Event_gold_mentions_validated.json",
                "Train_Event_gold_mentions.json"]
wec_eng = list()
for split_file in splits_files:
    wec_eng.append(json.load(open(cached_download(
        hf_hub_url(REPO_ID, split_file)), "r")))
```

## Dataset Structure

### Data Splits
- **Final version of the English CD event coreference dataset**<br>
    - Train - Train_Event_gold_mentions.json 
    - Dev - Dev_Event_gold_mentions_validated.json
    - Test - Test_Event_gold_mentions_validated.json

|                             | Train   | Valid | Test |
| -----                       | ------ | ----- | ----  |
| Clusters                    | 7,042  |  233  | 322   |
| Event Mentions              | 40,529 |  1250 | 1,893 |

- **The non (within clusters) controlled version of the dataset (lexical diversity)**<br>
    - All (experimental) - All_Event_gold_mentions_unfiltered.json

### Data Instances

```json
{
        "coref_chain": 2293469,
        "coref_link": "Family Values Tour 1998",
        "doc_id": "House of Pain",
        "mention_context": [
            "From",
            "then",
            "on",
            ",",
            "the",
            "members",
            "continued",
            "their"
  ],
  "mention_head": "Tour",
  "mention_head_lemma": "Tour",
  "mention_head_pos": "PROPN",
  "mention_id": "108172",
  "mention_index": 1,
  "mention_ner": "UNK",
  "mention_type": 8,
  "predicted_coref_chain": null,
  "sent_id": 2,
  "tokens_number": [
    50,
    51,
    52,
    53
  ],
  "tokens_str": "Family Values Tour 1998",
  "topic_id": -1
}
```

### Data Fields

|Field|Value Type|Value|
|---|:---:|---|
|coref_chain|Numeric|Coreference chain/cluster ID|
|coref_link|String|Coreference link wikipeida page/article title|
|doc_id|String|Mention page/article title|
|mention_context|List[String]|Tokenized mention paragraph (including mention)|
|mention_head|String|Mention span head token|
|mention_head_lemma|String|Mention span head token lemma|
|mention_head_pos|String|Mention span head token POS|
|mention_id|String|Mention id|
|mention_index|Numeric|Mention index in json file|
|mention_ner|String|Mention NER|
|tokens_number|List[Numeric]|Mentions tokens ids within the context|
|tokens_str|String|Mention span text|
|topic_id|Ignore|Ignore|
|mention_type|Ignore|Ignore|
|predicted_coref_chain|Ignore|Ignore|
|sent_id|Ignore|Ignore|


## License
We provide the following data sets under a <a href="https://creativecommons.org/licenses/by-sa/3.0/deed.en_US">Creative Commons Attribution-ShareAlike 3.0 Unported License</a>. It is based on content extracted from Wikipedia that is licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported License

## Contact
If you have any questions please create a Github issue at https://github.com/AlonEirew/extract-wec.