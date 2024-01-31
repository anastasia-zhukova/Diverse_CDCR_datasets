# Parsing HyperCoref 

Original repository of HyperCoref dataset: https://github.com/UKPLab/emnlp2021-hypercoref-cdcr/

### Papers
https://aclanthology.org/2021.emnlp-main.38/
```
@inproceedings{bugert2021event,
    title = {{Event Coreference Data (Almost) for Free: Mining Hyperlinks from Online News}},
    author = "Bugert, Michael and Gurevych, Iryna",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = {11},
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.38",
    doi = "10.18653/v1/2021.emnlp-main.38",
    pages = "471--491",
}
  ```


### To Parse HyperCoref
1. follow the guideline on this page: https://github.com/UKPLab/emnlp2021-hypercoref-cdcr/tree/master/hypercoref 
2. after downloading a list of urls of the news articles, make sure that you use only the urls for ABCNews and BBC outlets. 
3. after the execution of the data recollection and preprocessing is over, copy the folders of each outlet (i.e., ```abcnews.go.com```
and ```bbc.com```) into the folder ```HyperCoref-prep/HyperCoref/```. Make sure that each outlet contains a subfolder ```6_CreateSplitsStage_create_splits```. 
You might need to rename a folder from ```7_CreateSplitsStage_create_splits``` to ```6_CreateSplitsStage_create_splits```.
4. execute ```python parse_hypercoref.py``` 

HyperCoref annotated only events, so ```entity_mentions.json``` are saved as empty list. We reuse the train/val/test subset split 
provided in the dataset. We parse the dataset to keep as close as possible to the experimental setup described in the paper, i.e., 
1. we ignore documents if they only contain a singleton-mention
2. we downsample the train sets to 25k mentions for each outlet
3. we downsample the val splits to 1.7k mentions for ABC and 2.4k for BBC 
4. when downsampling, we follow a strategy of keeping the larger clusters first and then moving on to smaller sizes. We randomly 
select clusters when we need to make a specified. This strategy minimized the number of singletons in the dataset.
5. we keep the test sets as-is 
6. we keep the original maximum-span annotation style of HyperCoref

To fit HyperCoref to the topic/subtopic/doc structure of the benchmark, we create subtopics from the original document urls: 
```abcnews.go.com/{subtopic}/{article_id}```. 

### Topic organization
News articles in the dataset are organized as following: 

```
-> topic (i.e., outlet)
    -> subtopic (i.e., event defined from news articles' urls)
        -> documents (news articles)
   ```

