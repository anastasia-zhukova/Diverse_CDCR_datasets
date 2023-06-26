# Parsing NewsWCL50

Original repository of NewsWCL50: https://github.com/fhamborg/NewsWCL50. 

Paper: https://ieeexplore.ieee.org/document/8791197 

```
@INPROCEEDINGS{Hamborg2019a,
  author={Hamborg, Felix and Zhukova, Anastasia and Gipp, Bela},
  booktitle={2019 ACM/IEEE Joint Conference on Digital Libraries (JCDL)}, 
  title={Automated Identification of Media Bias by Word Choice and Labeling in News Articles}, 
  year={2019},
  volume={},
  number={},
  pages={196-205},
  doi={10.1109/JCDL.2019.00036}}
```

The dataset contains 10 topics with no subtopic level. But according to [the definition](README.md) subtopics contain 
event-related articles whereas topic aggregate subtopics. To ensure that the dataset fits into the structure of ```topic/subtopic/document```,
we turn each original topic into a subtopic and place the subtopics under the topic of the identical to subtopic ID. 

The dataset is organized as following: 

```
-> topic (same as subtopic)
    -> subtopic (original topic)
        -> documents (news articles)
   ```

Since the dataset contained newline delimiters in the original texts to separate paragraphs, we kept them in the datasets as original tokens. 
To ensure that these symbols do cause troubles in parsing CoNLL format, we saved them as ```\\n```. So if you decide to 
convert the dataset into the original text, remember to replace these symbold with the correct ```\n```. 

To parse NewsWCL50:
1) make sure that you downloaded the dataset by running ```python setup.py``` 
2) execute ```python parse_newswcl50.py``` 