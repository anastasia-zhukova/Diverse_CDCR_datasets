# Parsing CD2CR
The original repository of CD2CR: https://github.com/ravenscroftj/cdcrtool

### Paper 
```
@inproceedings{ravenscroft-etal-2021-cd,
    title = "{CD}{\^{}}2{CR}: Co-reference resolution across documents and domains",
    author = "Ravenscroft, James  and
      Clare, Amanda  and
      Cattan, Arie  and
      Dagan, Ido  and
      Liakata, Maria",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.eacl-main.21",
    doi = "10.18653/v1/2021.eacl-main.21",
    pages = "270--280"
}
```

### To parse CD2CR 
1) make sure that you downloaded the dataset by running ```python setup.py``` 
2) execute ```python parse_cd2cr.py```

The dataset was collected for the topics about science and technology, so we moved the original topics to the subtopics. 
Such a change also follows the [definitions of topic and subtopic](README.md). 
The original datasets didn't have unique topics, i.e., now subtopics, across the subsets (train, validation, and test). 
To enable correct calculation of the comparison metrics of the datasets, 
we assigned unique subtopic IDs to each subset. We also removed the former subtopic labels for news and scientific articles because 
a) such a differentiation didn't follow a definition of a subtopic, 2) each former subtopic had only one article, i.e., either news or science, 
which doesn't correspond to the CDCR task. So, the current version has one general topic and two articles per each subtopic. 


### Topic organization
News articles in the dataset are organized as following:
```
-> topic (one topic about science and technology)
    -> subtopic (scientific "event")
        -> documents (one scientific and one news article)
   ```
The dataset contains _one topic_ about technology and science. Subtopics report about different events within this topic.  

______________________________________________________

# CD2CR 
_From the original repository_

## Dataset Access

The $CD^2CR$ dataset is provided in CONLL markup format. 

You can download the partitioned dataset here:

 * [train set](CDCR_Corpus/train.conll)
 * [test set](CDCR_Corpus/test.conll)
 * [dev set](CDCR_Corpus/dev.conll)

Document IDs in the corpus follow the convention "{topic_id}_{document_type}_{document_id}" where topic_id uniquely identifies a pair of related news article and scientific paper documents, document_type indicates which of the pair the document is (either news or science) and document_id uniquely identifies the document in our annotation system's RDBMS tables.

A JSON mapping of news article IDs "{something}_news_{id}" to the original URL of the news article can be found here:

 * [news urls](CDCR_Corpus/news_urls.json)

A JSON Mapping of scientific paper IDs "{something}_science_{id}" to the DOI of the scientific paper can be found here:

 * [scientific paper DOIs](CDCR_Corpus/sci_papers.json)

## Checklist tasks

In table 4 of our paper we describe a series of 'challenging' co-reference tasks that we use as test cases/unit tests in the style of [Ribeiro et al](https://www.aclweb.org/anthology/2020.acl-main.442/). 

The specific list of these challenging tasks can be found [here](CDCR_Corpus/checklist_conll_testset.csv). This file contains a list of manually annotated 'is co-referent?' yes/no checks that were hand picked by the authors as particularly difficult as described in [our paper](https://arxiv.org/abs/2101.12637).

Each row gives the document IDs containing the two mentions (corresponding to the document ID in the third column of the test.conll file without the topic prefix), the token indices of the mention separated by semi colons (corresponding to the 5th column in the CoNLL file), is_coreferent indicates whether or not the pair co-refer or not (there are challenging cases of both yes and no that we wanted to explore) and finally the type of task or reason that we picked it.

The below diagram provides visual explanation for how the information fits together

![diagram showing how the tables in the checklist csv correspond to the CoNLL file](https://github.com/ravenscroftj/cdcrtool/blob/master/assets/checklist_table.png?raw=true)

## Trained Model

You can find model and config files for our CA-V model [here](https://papro.org.uk/downloads/cd2cr/models/cd2cr_ca_v.zip). The model is compatible with [Arie Cattan's coref repository](https://github.com/ariecattan/coref).