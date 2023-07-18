# Parsing NiDENT 

Original repository of English NiDENT dataset: http://clic.ub.edu/corpus/en/

### Papers
* Concept of near-identity https://aclanthology.org/L10-1103/ 
```
@inproceedings{recasens-etal-2010-typology,
    title = "A Typology of Near-Identity Relations for Coreference ({NIDENT})",
    author = "Recasens, Marta  and
      Hovy, Eduard  and
      Mart{\'\i}, M. Ant{\`o}nia",
    booktitle = "Proceedings of the Seventh International Conference on Language Resources and Evaluation ({LREC}'10)",
    month = may,
    year = "2010",
    address = "Valletta, Malta",
    publisher = "European Language Resources Association (ELRA)",
    url = "http://www.lrec-conf.org/proceedings/lrec2010/pdf/160_Paper.pdf"
}
  ```
* NiDENT dataset https://aclanthology.org/L12-1391/
```
@inproceedings{recasens-etal-2012-annotating,
    title = "Annotating Near-Identity from Coreference Disagreements",
    author = "Recasens, Marta  and
      Mart{\'\i}, M. Ant{\`o}nia  and
      Orasan, Constantin",
    booktitle = "Proceedings of the Eighth International Conference on Language Resources and Evaluation ({LREC}'12)",
    month = may,
    year = "2012",
    address = "Istanbul, Turkey",
    publisher = "European Language Resources Association (ELRA)",
    url = "http://www.lrec-conf.org/proceedings/lrec2012/pdf/674_Paper.pdf",
    pages = "165--172"
}
```

### To Parse NiDENT
1. request access to the original files on http://clic.ub.edu/corpus/en/. Download the dataset and make sure that the XML files are located in the following repository: 
```NiDENT-prep/NiDENT/english-corpus```.
2. download NP4E via ```setup.py```. NP4E is required to maintain the subtopic structure (NiDENT is reannotated NP4E). 
3. execute ```python parse_nident.py``` 

NiDENT annotated only entities, so ```event_mentions.json``` are saved as empty list. MMAX format didn't provide an extra tag to 
link coreference chains from the event-related documents into cross-document clusters, so we applied a simple yet reliable heuristic 
to restore CDCR clusters. If at least two non-pronoun mentions or their heads are identical, we merge the chains into clusters. 

We propose a train-val-test split for NiDENT in the ```train_val_test_split.json``` file. The split is on the subtopic level
and assigns three subtopics for training and one per validation and test. 

A mapping of the subtopic IDs to topic names is the following: 0) bukavu, 1) china, 2) israel, 3) peru, 4) tajikistan.

### Topic organization
News articles in the dataset are organized as following: 

```
-> topic (one topic about bomb, explosion, and kidnap)
    -> subtopic (event)
        -> documents (news articles)
   ```

The dataset contains _one topic_ about bomb, explosion, and kidnap. Subtopics report about different events within this topic.  
