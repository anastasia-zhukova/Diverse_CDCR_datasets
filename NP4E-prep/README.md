# Parsing NP4E

Original repository of NP4E dataset: http://clg.wlv.ac.uk/projects/NP4E/

### Paper
https://aclanthology.org/L06-1325/ 
```
@inproceedings{hasler-etal-2006-nps,
    title = "{NP}s for Events: Experiments in Coreference Annotation",
    author = "Hasler, Laura  and
      Orasan, Constantin  and
      Naumann, Karin",
    booktitle = "Proceedings of the Fifth International Conference on Language Resources and Evaluation ({LREC}{'}06)",
    month = may,
    year = "2006",
    address = "Genoa, Italy",
    publisher = "European Language Resources Association (ELRA)",
    url = "http://www.lrec-conf.org/proceedings/lrec2006/pdf/539_pdf.pdf",
}
```

### To parse NP4E
1) make sure that you downloaded the dataset by running ```python setup.py``` 
2) execute ```python parse_np4e.py``` 


### Topic organization
News articles in the dataset are organized as following:
```
-> topic (one topic about bomb, explosion, and kidnap)
    -> subtopic (event)
        -> documents (news articles)
   ```
The dataset contains _one topic_ about bomb, explosion, and kidnap. Subtopics report about different events within this topic.  


### Entity coreference 
Since annotation of events was limited to five specific events and the authors didn't annotate event clusters for all 
topics, we parse only entity clusters and save ```event_mentions.json``` as empty list. 
Unlike most CDCR datasets, the events described as a noun phrase, e.g., an attack, is annotated as entity. 

Since the MMAX format didn't provide an extra tag to link coreference chains from the event-releted documents into 
cross-document clusters, so we applied a simple yet reliable heuristic to restore CDCR clusters. 
If at least two non-pronoun mentions or their heads are identical, we merge the chains into clusters. We make an exception 
to the chains where there is one ovelap with a proper noun, and merge such cases as well. 