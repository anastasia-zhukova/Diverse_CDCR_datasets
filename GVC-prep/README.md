# Parsing GVC

Original repository of GVC corpus: https://github.com/cltl/GunViolenceCorpus 

### Paper
https://aclanthology.org/L18-1480/

```
@inproceedings{vossen-etal-2018-dont,
    title = "Don{'}t Annotate, but Validate: a Data-to-Text Method for Capturing Event Data",
    author = "Vossen, Piek  and
      Ilievski, Filip  and
      Postma, Marten  and
      Segers, Roxane",
    booktitle = "Proceedings of the Eleventh International Conference on Language Resources and Evaluation ({LREC} 2018)",
    month = may,
    year = "2018",
    address = "Miyazaki, Japan",
    publisher = "European Language Resources Association (ELRA)",
    url = "https://aclanthology.org/L18-1480",
}
```

### To parse GVC
1. make sure that you downloaded the dataset by running ```python setup.py```
2. execute ```python parse_gvc.py```

The parsing script reuses a train/val/test split from [Burgert et al. (2021)](https://direct.mit.edu/coli/article/47/3/575/102774/Generalizing-Cross-Document-Event-Coreference) 
to create folders with the train/dev/test splits located in the ```\output_data```. 

Both the assignment of documents to subtopics and the assignment the subtopics to the train/val/test split are reused 
from the [GitHub repository](https://github.com/UKPLab/cdcr-beyond-corpus-tailored/tree/master/resources/data/gun_violence) 
of Burgert et al. (2021). 

If a mention has a cluster_ID = 0, then we make such mentions singleton clusters with unique cluster IDs.  


### Output format
The dataset articles is organized as following: 

```
-> topic (one topic about gun violence)
    -> subtopic (subtopic events)
        -> documents (news articles)
   ```

The dataset contains _one topic_ about gun violence. Subtopics report about different accidents reporting within this topic. 

