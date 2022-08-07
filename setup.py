# PARAMS
import os

CONTEXT_RANGE = 100

# FOLDERS
NEWSWCL50_FOLDER_NAME = "2019_annot"
ECBPLUS_FOLDER_NAME = "ECB+"
NIDENT_FOLDER_NAME = "NiDENT"
NP4E_FOLDER_NAME = "NP4E"
MEANTIME_FOLDER_NAME_ENGLISH = "meantime_newsreader_english_oct15"
MEANTIME_FOLDER_NAME_DUTCH = "meantime_newsreader_dutch_dec15"
MEANTIME_FOLDER_NAME_ITALIAN = "meantime_newsreader_italian_dec15"
MEANTIME_FOLDER_NAME_SPANISH = "meantime_newsreader_spanish_nov15"
OUTPUT_FOLDER_NAME = "output_data"
SUMMARY_FOLDER = "summary"
TMP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
DATA_PATH = "C:\\Users\\snake\\Documents\\GitHub\\Diverse_CDCR_datasets\\"

# DATASETS
NEWSWCL50 = "NewsWCL50-prep"
ECB_PLUS = "ECBplus-prep"
MEANTIME = "MEANTIME-prep"
NIDENT = "NiDENT-prep"
NP4E = "NP4E-prep"

# FILES
SAMPLE_DOC_JSON = "_sample_doc.json"
SAMPLE_MENTION_JSON = "_sample_mention.json"
MENTIONS_ALL_CSV = "all_mentions.csv"
MENTIONS_EVENTS_JSON = "event_mentions.json"
MENTIONS_ENTITIES_JSON = "entities_mentions.json"
CONLL_CSV = "conll.csv"
SUMMARY_CHAINS_CSV = "summary_chains.csv"
SUMMARY_TOPICS_CSV = "summary_dataset_topics.csv"
MANUAL_REVIEW_FILE = "manual_review_needed.json"

# coref types
STRICT = "STRICT"
NEAR_IDENTITY = "NEAR_IDENTITY"

# doc.json fields (from news-please)
TITLE = "title"
DESCRIPTION = "description"
TEXT = "text"
SOURCE_DOMAIN = "source_domain"

# NewsWCL50 original column names in annotated mentions)
CODE = "Code"
SEGMENT = "Segment"
DOCUMENT_NAME = "Document name"
BEGINNING = "Beginning"
TYPE = "type"

# mentions.json fields
TOPIC_ID = "topic_id"
TOPIC = "topic"
COREF_CHAIN = "coref_chain"
MENTION_FULL_TYPE = "mention_full_type"
MENTION_TYPE = "mention_type"
MENTION_NER = "mention_ner"
MENTION_HEAD_POS = "mention_head_pos"
MENTION_HEAD_LEMMA = "mention_head_lemma"
MENTION_HEAD = "mention_head"
MENTION_HEAD_ID = "mention_head_id"
DOC_ID_FULL = "doc_id_full"
DOC_ID = "doc_id"
IS_CONTINIOUS = "is_continuous"
IS_SINGLETON = "is_singleton"
SENTENCE = "sentence"
MENTION_ID = "mention_id"
SCORE = "score"
SENT_ID = "sent_id"
MENTION_CONTEXT = "mention_context"
TOKENS_NUMBER = "tokens_number"
TOKENS_TEXT = "tokens_text"
TOKENS_STR = "tokens_str"
TOKEN_ID = "token_id"
COREF_TYPE = "coref_type"
SUBTOPIC = "subtopic"
CONLL_DOC_KEY = "conll_doc_key"

# conll fields
REFERENCE = "reference"
DOC_IDENTIFIER = "doc_identifier"
TOKEN = "token"
TOPIC_SUBTOPIC = "topic/subtopic_name"

# summary fields
DATASET_NAME = "dataset"
TOPICS = "topics"
ENTITY = "entity"
EVENT = "event"
MENTIONS = "mentions"
PHRASING_DIVERSITY = "phrasing_diversity"
UNIQUE_LEMMAS = "unique_lemmas"
WEIGHTED = "_weighted"
MEAN = "_mean"
ALL = "_all"
WO_SINGL = "_wo_singl"
ARTICLES = "articles"
TOKENS = "tokens"
SINGLETONS = "singletons"
AVERAGE_SIZE = "average_size"

# support fields
CONCAT_TEXT = "concat_text"
FIRST_TOKEN = "first_token"
LAST_TOKEN = "last_token"

# ECB+ orig annotated files
T_ID = "t_id"
ID = "id"
SENT = "sent"
M_ID = "m_id"
NUM = "number"

# MEANTIME languages
EN = "en"
NL = "nl"
IT = "it"
ES = "es"

if __name__ == '__main__':
    import spacy
    import gdown
    import zipfile
    from logger import LOGGER

    FOLDER = "folder"
    ZIP = "zip"
    LINK = "link"

    while True:
        try:
            b = input("Would you like to download the spacy languages? (y/n) : ")
            b = b.lower()
            assert b == "y" or b == "n"
            break
        except (ValueError, AssertionError) as e:
            print("Oops! That input was not correct (y/n). Please retry.")

    if b == "y":
        print("Downloading spacy languages...")
        spacy.cli.download('en_core_web_sm')
        spacy.cli.download('es_core_news_sm')
        spacy.cli.download('nl_core_news_sm')
        spacy.cli.download('it_core_news_sm')
    else:
        print("Skipping the download of languages.")

    datasets = {ECB_PLUS: {LINK: "https://github.com/cltl/ecbPlus/raw/master/ECB%2B_LREC2014/ECB%2B.zip",
                           ZIP: os.path.join(os.getcwd(), ECB_PLUS, ECBPLUS_FOLDER_NAME + ".zip"),
                           FOLDER: os.path.join(os.getcwd(), ECB_PLUS)},
                MEANTIME: {LINK: "https://drive.google.com/u/0/uc?id=1K0hcWHOomyrFaKigwzrwImHugdb1pjAX&export=download;https://drive.google.com/u/0/uc?id=1qhKFhO-EszieMz_B7rOJhvbWcIeEg1F5&export=download;https://drive.google.com/u/0/uc?id=1-i3DoyenEYV8_jY6bYaNJ4lsmtb4-4Tw&export=download;https://drive.google.com/u/0/uc?id=1NB6Vw_W7KYii7L7OLMnW2qfq1KWPZ4de&export=download",
                           ZIP: os.path.join(os.getcwd(), MEANTIME, "MEANTIME_tmp" + ".zip"),
                           FOLDER: os.path.join(os.getcwd(), MEANTIME)},
                NEWSWCL50: {LINK: "https://drive.google.com/u/1/uc?id=1ZcTnDeY85iIeUX0nvg3cypnRq87tVSVo&export=download",
                            ZIP: os.path.join(os.getcwd(), NEWSWCL50, NEWSWCL50_FOLDER_NAME + ".zip"),
                            FOLDER: os.path.join(os.getcwd(), NEWSWCL50)}},


    prompt_str = "The following datasets are available for download: \n\n"
    for i, dataset in enumerate(datasets.keys()):
        prompt_str = prompt_str + str(i) + ": " + dataset + "\n"
    prompt_str = prompt_str + str(len(datasets)) + ": all datasets \n"

    print(prompt_str)
    while True:
        try:
            input_number = int(input("Please enter a number to download the dataset: "))
            assert 0 <= input_number <= len(datasets)
            break
        except (ValueError, AssertionError) as e:
            print("Oops! Seems like the number you entered is not a number or not valid. Please retry. ")

    # All datasets download
    if input_number == len(datasets):
        for dataset, values in datasets.items():
            LOGGER.info(f"Getting: {dataset}")
            if dataset == MEANTIME:
                # download all languages
                links = values[LINK].split(";")
                LOGGER.info(f"Downloading datasets for {str(len(links))} languages.")
                for link in links:
                    gdown.download(link, values[ZIP], quiet=False)
                    with zipfile.ZipFile(values[ZIP], 'r') as zip_ref:
                        zip_ref.extractall(values[FOLDER])
            else:
                gdown.download(values[LINK], values[ZIP], quiet=False)
                with zipfile.ZipFile(values[ZIP], 'r') as zip_ref:
                    zip_ref.extractall(values[FOLDER])

            if dataset == ECB_PLUS:
                gdown.download("https://raw.githubusercontent.com/cltl/ecbPlus/master/ECB%2B_LREC2014/ECBplus_coreference_sentences.csv",
                               os.path.join(os.getcwd(), ECB_PLUS, ECBPLUS_FOLDER_NAME, "ECBplus_coreference_sentences.csv"), quiet=False)

    # Download selected dataset
    elif 0 <= input_number < len(datasets):
        for i, (dataset, values) in enumerate(datasets.items()):
            if i != input_number:   # skip other datasets
                continue
            LOGGER.info(f"Getting: {dataset}")
            if dataset == MEANTIME:
                # download all languages
                links = values[LINK].split(";")
                LOGGER.info(f"Downloading datasets for {str(len(links))} languages.")
                for link in links:
                    gdown.download(link, values[ZIP], quiet=False)
                    with zipfile.ZipFile(values[ZIP], 'r') as zip_ref:
                        zip_ref.extractall(values[FOLDER])
            else:
                gdown.download(values[LINK], values[ZIP], quiet=False)
                with zipfile.ZipFile(values[ZIP], 'r') as zip_ref:
                    zip_ref.extractall(values[FOLDER])

            if dataset == ECB_PLUS:
                gdown.download(
                    "https://raw.githubusercontent.com/cltl/ecbPlus/master/ECB%2B_LREC2014/ECBplus_coreference_sentences.csv",
                    os.path.join(os.getcwd(), ECB_PLUS, ECBPLUS_FOLDER_NAME, "ECBplus_coreference_sentences.csv"),
                    quiet=False)
    print("Setup successful.")
