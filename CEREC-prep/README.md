# Parse CEREC 
The original repository of CEREC: https://github.com/paragdakle/emailcoref/

# Paper 
https://aclanthology.org/2020.coling-main.30/

```
@inproceedings{dakle-moldovan-2020-cerec,
    title = "{CEREC}: A Corpus for Entity Resolution in Email Conversations",
    author = "Dakle, Parag Pravin  and
      Moldovan, Dan",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2020.coling-main.30",
    doi = "10.18653/v1/2020.coling-main.30",
    pages = "339--349"
}
```

### To parse CEREC 
1) make sure that you downloaded the dataset by running ```python setup.py``` 
2) execute ```python parse_cerec.py```

We assign the two original .conll files into the train/val/test splits as following: 

| Split | File                            | Number of email threads |
|-------|---------------------------------|-------------------------|
| train | ```cerec.conll```               | randomly selected 100   |
| val   | ```cerec.validation.20.conll``` | 20                      |
| test  | ```cerec.validation.14.conll``` | 14                      |

We form document_id as a composition of a message identified (see below), a speaker, and a document index, which is unique 
across all splits. We exclude tokens that belong to a speaker "SYSTEM". 

A coreference chain consists of the subtopic name, original coreference_id within a subtopic, and a shortened entity type, 
which is created based on the extracted and aggregated NER of the mentions' heads. 


### Topic organization
Documents in the dataset are organized as following: 

```
-> topic (one topic about emails)
    -> subtopic (an email thread)
        -> documents (emails)
   ```

The dataset contains _one topic_ about emails. Subtopics report about different email chains within this topic.  
_______________________

# CEREC: Corpus for Entity Resolution in Email Conversations

Corpus containing 6001 Enron email threads weakly-annotated for entity coreference resolution task. The actual emails can be downloaded from [here](https://www.cs.cmu.edu/~./enron/).


## Corpus Description

CEREC contains 6001 email threads from the Enron Email Corpus containing 36,448 email messages and 38,996 entity coreference chains.

> <strong>For using just the Seed corpus, follow the instructions provided [here](https://github.com/paragdakle/emailcoref/blob/master/data/LREC/).</strong>

An email thread annotation is saved in the CoNLL format with the following naming convention: ```username_directory_email_no.conll```
where: ```username``` - Name of the user directory in the Enron Email Corpus,
```directory``` - Name of the directory for the specific user.
```email_no``` - Filename in the specific directory.
Each annotation file is an eight column double tab separated file, and contains mention and coreference annotations. 

Detailed column information in the order found is as follows:

| Column | Type        | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|--------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1      | Token       | The actual token as found in the email thread                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 2      | MI          | The value of message identifier feature for the token                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 3      | SI          | The value of section identifier feature for the token                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 4      | Speaker     | The speaker of the token                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 5      | Entity Type | The type of the entity this token represents. This column also contains two additional annotations - coreference chain informaion for the entity type encoded in a parenthesis structure and if the entity is the antecedent given by "*". E.g. In "(P0*", ( implies the token is starting a mention span, P implies the token is of PER entity type, 0 implies the token belongs to the coreference chain with id 0 for PER entity type, and * implies it is part of the antecedent of the coreference chain. |
| 6      | Mention     | Mention information encoded in a parenthesis structure.                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| 7      | Coreference | Coreference chain information encoded in a parenthesis structure.                                                                                                                                                                                                                                                                                                                                                                                                                                              |


The corpus can be found in zip file in the following directory ```data/COLING/```.

The zip file contains the following files:
1. cerec.conll - The CEREC corpus containing 6001 email threads, and their mention and coreference annotations.
2. cerec.validation.XX.conll - Email threads used in the validation set for CEREC experiments. These email threads were also used as validation and test sets for feature evaluation.
3. mention.corrected.XX.conll - Email threads that were manually corrected for mention annotations and then used to train a model for annotation quality evaluation.
4. seed.conll - The email threads from the seed corpus containing feature annotations and separated mention annotations.

<strong>Note: Annotations for columns 2-5 are provided only for files mentioned in points 2 and 4 above. For other files, either the columns are blank or have system generated values.</strong>

## Experiments

The code used to generate the results can be found [here](https://github.com/mandarjoshi90/coref). Evalution scripts for all metrics can be found [here](https://github.com/conll/reference-coreference-scorers).