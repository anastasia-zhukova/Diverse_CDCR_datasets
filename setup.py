import spacy
import gdown

spacy.cli.download('en_core_web_sm')
gdown.download("https://github.com/cltl/ecbPlus", "ECBplus-prep\\ECB+", quiet=False)
