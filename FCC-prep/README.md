# Parsing FCC and FCC-T 

Original repository of FCC/FCC-T dataset: https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2305

### Papers
* For the original FCC, see Bugert et al. 2020 "Breaking the Subtopic Barrier in Cross-Document Event Coreference 
Resolution", http://ceur-ws.org/Vol-2593/paper3.pdf
  ```
  @inproceedings{bugert2020breaking,
    title={{Breaking the Subtopic Barrier in Cross-Document Event Coreference Resolution}},
    author={Bugert, Michael and Reimers, Nils and Barhom, Shany and Dagan, Ido and Gurevych, Iryna},
    booktitle={Text2Story@ ECIR},
    pages={23--29},
    year={2020}
  }
  ```
* For the token-level reannotation FCC-T, see Bugert et al. 2021 "Generalizing Cross-Document Event Coreference 
Resolution Across Multiple Corpora", https://direct.mit.edu/coli/article/47/3/575/102774/Generalizing-Cross-Document-Event-Coreference
  ```
  @article{10.1162/coli_a_00407,
        author = {Bugert, Michael and Reimers, Nils and Gurevych, Iryna},
        title = "{Generalizing Cross-Document Event Coreference Resolution Across Multiple Corpora}",
        journal = {Computational Linguistics},
        volume = {47},
        number = {3},
        pages = {575-614},
        year = {2021},
        month = {11},
        issn = {0891-2017},
        doi = {10.1162/coli_a_00407},
        url = {https://doi.org/10.1162/coli\_a\_00407},
        eprint = {https://direct.mit.edu/coli/article-pdf/47/3/575/1971857/coli\_a\_00407.pdf},
  }
  ```

### To parse FCC + FCC-T
1) make sure that you obtained the dataset by following the guidelines from the original repository below. All dataset 
folders need to be placed into ```FCC-prep/FCC``` folder.
2) execute ```python parse_fcc.py```

### Output format

The dataset articles is organized as following: 

```
-> topic (one topic about football matches)
    -> subtopic (seminal event)
        -> documents (news articles)
   ```

The dataset contains _one topic_ about football. Subtopics report about different football events within this topic. 

The script produced two folders: 
1. ```FCC-prep/output_folder_FCC``` with the original event annotation on the sentence level
2. ```FCC-prep/output_folder_FCC-T``` with the original event annotation on the token level. Since the entities are not 
annotated into the entity clusters (i.e., no coreferences), we save them separately into a ```entity_mentions_attr.json``` file.
The regular ```entity_mentions.json``` is hence empty. 

### Event coreference
If there is an event with a label "other_event", we create compositional ID per collection to make it a less general event, e.g., 
```other_event-uefa_euro_2016```. 

We parsed a version of FCC-T with stacked actions.

### Entity (non)coreference
Since there is no coreferences among the entities, all of them are created as singletons but saved into an additional file 
called ```entity_mentions_attr.json``` . 

The ```chain_id``` is a compositional key and consists of the event(s) it belongs, i.e., event type + a unique ID based on its name, 
a semantic role label that this entity mention has, and a unique ID, if the same entity mention had multiple roles. 
For example, ```chain_id: "OCCaaalll111_participants_aa11"```. To preserve semantic roles of the entities in the existing 
mention's format, we save the labels in the ```coref_type``` attributes, e.g., ```coref_type: "participants"```. 

### Subtopics
Each split might contain documents that are not assigned to any seminal event. Similar to [Bugert et al. 2021]
(https://direct.mit.edu/coli/article/47/3/575/102774/Generalizing-Cross-Document-Event-Coreference), we create a new 
seminal event for such unassigned documents. 



----------------------------------------------------------------
# Football Coreference Corpus

[This original script](https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2305/FCC_FCC-T_ECBp_dates_2022-05-25.zip?sequence=4&isAllowed=y) generates:
1. the original sentence-level Football Coreference Corpus (*FCC*),
2. a version of the sentence-level FCC which was cleaned and updated after manual review,
3. *FCC-T*, the extended version of the Football Coreference Corpus with reannotated token-level spans,
4. and publication date annotations for the *ECB+* corpus [1].

The script downloads the original documents from archive.org's WaybackMachine, cleans and processes them locally on your
machine and combines the result with our annotations. See README.md for instructions.


## Creating the datasets
1. Install the [docker engine](https://docs.docker.com/engine/install/) on your machine.
2. Choose a path to which the datasets will be written to on the host system (this example uses `~/fcc`). Open a terminal and run:
   ```bash
   docker run -v ~/fcc:/fcc/datasets -it mbugert/fcc
   ```
3. In the docker shell that opens, run `make` to generate the dataset. The whole process takes roughly 30 minutes.

The archive.org servers can be a bit finicky. If the script terminates prematurely, try running `make clean; make`. If that didn't help, please try again from scratch with `make cleanall; make`.
Once the script finishes, `exit` the shell. The datasets will be located at the host path specified earlier.


If this doesn't work, try the following: 
1. Run docker run ```-v ~/fcc:/fcc/datasets -it mbugert/fcc```
2. In the container, run ```apt update && apt install -y nano```
3. Run ```nano Makefile```, then in the line with pip install (line 34), replace ```pip install --upgrade pip==22.0.4``` with ```pip install --upgrade pip>=22.0.4 setuptools wheel```
4. Save the changes (Ctrl+o, y, Ctrl+x)
5. Run make to create the dataset


## Dataset contents
### `2020-10-05_FCC_cleaned` and `2020-03-18_FCC`
```
.
├── train
│   ├── documents.csv                                   # contains for each document: the collection (i.e. football tournament), publication date and seminal event
|   ├── tokens.csv                                      # tokenized document contents
│   ├── mentions_cross_subtopic.csv                     # contains sentence-level event mention and linking annotation of events different from a document's seminal event    
│   ├── mentions_seminal_other.csv                      # contains sentence-level annotation for mentions of a document's seminal event, plus sentence-level event mention annotations of football events *outside* of the football tournament that each document was primarily written about 
│   └── hard_mentions_same_type_out_of_ontology.csv     # mentions of events outside of the current football tournament (see `mentions_seminal_other.csv`) that we removed during cleaning because they were too difficult to be linked to a knowledge base of football event, with reason why                               
├── dev
|   ...
└── test
    ...
```

### `2020-10-05_FCC-T`
These annotations are meant to be used with the `tokens.csv` from `2020-10-05_FCC_cleaned` or `2020-03-18_FCC` (the `tokens.csv` files in each of these folders are identical).
```
.
├── train
│   ├── with_stacked_actions                            # Version of FCC-T containing event mentions which have identical spans but refer to multiple different events. This may be useful for training.
│   │   ├── cross_subtopic_mentions_action.csv          # Contains token-level span and linking annotations for event actions. The annotations cover the evemt mentions from `mentions_cross_subtopic.csv` and the `mentions-same-type-out-of-ontology` column from `mentions_seminal_other` of the cleaned sentence-level FCC.
│   │   ├── cross_subtopic_mentions_location.csv        # contains token-level location spans and their type
│   │   ├── cross_subtopic_mentions_participants.csv    # contains token-level participant spans and their type
│   │   ├── cross_subtopic_mentions_time.csv            # contains token-level temporal expression spans and their type
│   │   └── cross_subtopic_semantic_roles.csv           # contains pseudo-SRL annotations which link event action mentions (first two columns) to corresponding participants/time/location mentions in the same sentence (last two columns) 
│   └── without_stacked_actions                         # Version of FCC-T in which each event action span only refers to a single event. This is meant to be used for testing, since current evaluation metrics cannot handle these cases. 
|       ...
├── dev
|   ...
└── test
    ...
```

### `2020-09-11_ECBplus_publication_dates`
```
.
└── 2020-09-11_ecbp_publication_dates_timex.csv         # contains for each document where it is available the TIMEX3 type and value (see [2]) and whether the TIMEX value is grounded, i.e. without free variables. 
```

## Support
Please get in touch on Github: https://github.com/UKPLab/cdcr-beyond-corpus-tailored/issues

## References
* [1] http://www.newsreader-project.eu/results/data/the-ecb-corpus/
* [2] http://www.timeml.org/tempeval2/tempeval2-trial/guidelines/timex3guidelines-072009.pdf