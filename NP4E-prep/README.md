# Parsing NP4E

Original repository of NP4E dataset: http://clg.wlv.ac.uk/projects/NP4E/

Paper: https://aclanthology.org/L06-1325/ 
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

Since annotation of events was limited to five specific events and the authors didn't annotate event clusters for all 
topics, we parse only entity clusters. Unlike most CDCR datasets, the events described as a noun phrase, e.g., an attack, 
is annotated as entity. 

To parse NP4E:
1) make sure that you downloaded the dataset by running ```python setup.py``` 
2) execute ```python parse_np4e.py``` 

The dataset is organized as following: 

```
-> topic
    -> subtopic (event)
        -> documents (news articles)
   ```

The dataset contains _one topic_ about bomb, explosion, and kidnap. Subtopics report about different events within this topic.  