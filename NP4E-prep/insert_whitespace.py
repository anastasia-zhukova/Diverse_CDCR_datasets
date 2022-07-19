from typing import Tuple
import re


def append_text(text, word) -> Tuple[str, str, bool]:
    """
    Decides which whitespace to add when addiding a token to the already concatenated tokens.
    :param text: Preceding part to the word of the sentence
    :param word: a word to concatenate
    :return: A sentences with the concatenated word, modifications to this word, and a whitespace delimiter.
    """
    word = "\"" if word == "``" else word

    if not len(text):
        return word, word, True

    space = "" if word in ".,?!)]`\"\'" or word == "'s" else " "
    space = " " if word[0].isupper() and not len(text) else space
    space = " " if word in ["-", "(","\""] else space

    if len(text):
        space = "" if text[-1] in ["(", "``", "\"", "["] else space
        space = " " if text[-1] in [".,?!)]`\"\'"] else space
        space = "" if text[-1] in ["\""] and word.istitle() else space
        space = "" if word in ["com", "org"] else space

    if len(text) > 1:
        space = "" if word.isupper() and text[-1] == "." and text[-2].isupper() else space
        space = " " if not len(re.sub(r'\W+', "", text[-2:])) and \
                       len(text[-2:]) == len(text[-2:].replace(" ", "")) else space
        space = "" if text[-1] == "." and text[-2] in "0123456789" and len(
            set(word).intersection(set("0123456789"))) > 0 else space

    return text + space + word, word, space == ""
