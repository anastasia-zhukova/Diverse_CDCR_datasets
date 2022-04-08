import nltk
from nltk.corpus import stopwords
import string
import numpy as np
from typing import Dict, List

nltk.download('stopwords')


def phrasing_complexity_calc(mentions: List[Dict]) -> float:
    """
    Calculates a metric that represents phrasing complexity of an entity, i.e., how various is the wording of the
    phrases referring to this entity. For more details see XCoref paper
    Returns:
        Internal headword-based structure, value of the phrasing complexity

    a format of members from the json for vis:
    [
        {
            "sentence": 0,
            "id: ll0009,
            "doc_id: 0,
            "tokens": [6, 7, 8, 9, 10, 11, 12, 13, 14],
            "tokens_text": ["Trump", "in", "broad", "quest", "on", "Russia", "ties", "and", "obstruction"],
            "text": "Trump in broad quest on Russia ties and obstruction",
            "head_token_index": 6,
            "head_token_word": "Trump"
        }
    ]
    """
    headwords_phrase_tree = {}
    for mention in mentions:
        mentions_wo_stopwords = [w for w in mention["tokens_text"] if w not in string.punctuation and w not in stopwords.words("english")]

        if not len(mentions_wo_stopwords):
            continue

        mentios_wordset = frozenset(mentions_wo_stopwords)

        if mention["mention_head"] not in headwords_phrase_tree:
            headwords_phrase_tree[mention["mention_head"]] = {"set": {mentios_wordset},
                                                                 "list": [mentions_wo_stopwords]}
        else:
            # set ensures unique phrases
            headwords_phrase_tree[mention["mention_head"]]["set"] = \
                                        headwords_phrase_tree[mention["mention_head"]]["set"].union(mentios_wordset)
            # list keeps actual number of phrase occurence
            headwords_phrase_tree[mention["mention_head"]]["list"].append(mentions_wo_stopwords)

    sets = []
    fractions = []
    for head, head_properties in headwords_phrase_tree.items():
        fractions.append(len(head_properties["set"]) / (len(head_properties["list"])))
        sets.append(len(head_properties["set"]))
        #print(len(head_properties["set"]))
        #print(len(head_properties["list"]))
        #print("----")

    score = np.sum(np.array(fractions)) * np.sum(np.array(sets)) / len(mentions) if len(mentions) > 1 else 1
    #print(np.sum(np.array(fractions)) * np.sum(np.array(sets)))
    #print(len(mentions))
    #print("-------------------")
    return float(format(score, '.3f'))


if __name__ == '__main__':
    import json
    with open("2021-10-27_18_53_13_8_MuellerQuestionsTrump_entity_data.json", "r") as file:
        data_dict =json.load(file)
    ent = data_dict["entities"][0]
    pc = phrasing_complexity_calc(ent["mentions"])
    print(f'Phrasing complexity is {pc}')