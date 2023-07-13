# PARAMS
import os
import json
import spacy
import gdown
import zipfile
import requests
from logger import LOGGER
from huggingface_hub import hf_hub_url, hf_hub_download

CONTEXT_RANGE = 100

# FOLDERS
NEWSWCL50_FOLDER_NAME = "2019_annot"
ECBPLUS_FOLDER_NAME = "ECB+"
NIDENT_FOLDER_NAME = "NiDENT"
NP4E_FOLDER_NAME = "NP4E"
WECENG_FOLDER_NAME = "WEC-Eng"
GVC_FOLDER_NAME = "GVC"
FCC_FOLDER_NAME = "FCC"
MEANTIME_FOLDER_NAME = "meantime_newsreader"
MEANTIME_FOLDER_NAME_ENGLISH = "meantime_newsreader_english_oct15"
MEANTIME_FOLDER_NAME_DUTCH = "meantime_newsreader_dutch_dec15"
MEANTIME_FOLDER_NAME_ITALIAN = "meantime_newsreader_italian_dec15"
MEANTIME_FOLDER_NAME_SPANISH = "meantime_newsreader_spanish_nov15"
OUTPUT_FOLDER_NAME = "output_data"
SUMMARY_FOLDER = "summary"
TMP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# DATASETS
NEWSWCL50 = "NewsWCL50-prep"
ECB_PLUS = "ECBplus-prep"
MEANTIME = "MEANTIME-prep"
NIDENT = "NiDENT-prep"
NP4E = "NP4E-prep"
WEC_ENG = "WECEng-prep"
GVC = "GVC-prep"
FCC = "FCC-prep"

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
IDENTITY = "IDENTITY"
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
DOC = "doc"
DOC_ID = "doc_id"
IS_CONTINIOUS = "is_continuous"
IS_SINGLETON = "is_singleton"
SENTENCE = "sentence"
MENTION_ID = "mention_id"
SCORE = "score"
SENT_ID = "sent_id"
MENTION_CONTEXT = "mention_context"
TOKENS_NUMBER = "tokens_number"
TOKENS_NUMBER_CONTEXT = "tokens_number_context"
TOKENS_TEXT = "tokens_text"
TOKENS_STR = "tokens_str"
TOKEN_ID = "token_id"
COREF_TYPE = "coref_type"
SUBTOPIC_ID = "subtopic_id"
SUBTOPIC = "subtopic"
CONLL_DOC_KEY = "conll_doc_key"
LANGUAGE = "language"

# conll fields
REFERENCE = "reference"
DOC_IDENTIFIER = "doc_identifier"
TOKEN = "token"
TOPIC_SUBTOPIC = "topic/subtopic_name"
TOPIC_SUBTOPIC_DOC = "topic/subtopic_name/doc"

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

# spacy packages
SPACY_EN = "en_core_web_sm"
SPACY_ES = "es_core_news_sm"
SPACY_NL = "nl_core_news_sm"
SPACY_IT = "it_core_news_sm"

FOLDER = "folder"
ZIP = "zip"
LINK = "link"


if __name__ == '__main__':
    datasets = {
        ECB_PLUS: {
            LINK: "https://github.com/cltl/ecbPlus/raw/master/ECB%2B_LREC2014/ECB%2B.zip",
            ZIP: os.path.join(TMP_PATH, ECBPLUS_FOLDER_NAME + ".zip"),
            FOLDER: os.path.join(os.getcwd(), ECB_PLUS)
        },
        NEWSWCL50: {
            LINK: "https://drive.google.com/uc?export=download&confirm=pbef&id=1ZcTnDeY85iIeUX0nvg3cypnRq87tVSVo",
            ZIP: os.path.join(TMP_PATH, NEWSWCL50_FOLDER_NAME + ".zip"),
            FOLDER: os.path.join(os.getcwd(), NEWSWCL50)
        },
        MEANTIME: {
            LINK: "https://drive.google.com/uc?export=download&confirm=pbef&id=1K0hcWHOomyrFaKigwzrwImHugdb1pjAX;https://drive.google.com/uc?export=download&confirm=pbef&id=1qhKFhO-EszieMz_B7rOJhvbWcIeEg1F5;https://drive.google.com/uc?export=download&confirm=pbef&id=1-i3DoyenEYV8_jY6bYaNJ4lsmtb4-4Tw;https://drive.google.com/uc?export=download&confirm=pbef&id=1NB6Vw_W7KYii7L7OLMnW2qfq1KWPZ4de",
            ZIP: os.path.join(TMP_PATH, MEANTIME_FOLDER_NAME + ".zip"),
            FOLDER: os.path.join(os.getcwd(), MEANTIME, MEANTIME_FOLDER_NAME)
        },
        WEC_ENG: {
            LINK: "Intel/WEC-Eng",
            ZIP: "",
            FOLDER: os.path.join(os.getcwd(), WEC_ENG, WECENG_FOLDER_NAME)
        },
        NP4E: {
           LINK: "http://clg.wlv.ac.uk/projects/NP4E/mmax/np4e_mmax2.zip",
           ZIP: os.path.join(TMP_PATH, NP4E_FOLDER_NAME + ".zip"),
           FOLDER: os.path.join(os.getcwd(), NP4E)
        },
        NIDENT: {
            LINK: "https://drive.google.com/uc?export=download&confirm=pbef&id=1BtjKwRGW0dWm4AqdkYRtwS7IGys1jnQx",
            ZIP: os.path.join(TMP_PATH, NIDENT_FOLDER_NAME + ".zip"),
            FOLDER: os.path.join(os.getcwd(), NIDENT)
        },
        FCC: {
            LINK: "https://drive.google.com/uc?export=download&confirm=pbef&id=1ZBe0JZAI-hJ-QzXcunDOzpfcl5s-1eTF",
            ZIP: os.path.join(TMP_PATH, FCC_FOLDER_NAME + ".zip"),
            FOLDER: os.path.join(os.getcwd(), FCC)
        },
        GVC: {
            LINK: "https://raw.githubusercontent.com/cltl/GunViolenceCorpus/master/;https://raw.githubusercontent.com/UKPLab/cdcr-beyond-corpus-tailored/master/resources/data/gun_violence/",
            ZIP: os.path.join(TMP_PATH, GVC_FOLDER_NAME + ".zip"),
            FOLDER: os.path.join(os.getcwd(), GVC, GVC_FOLDER_NAME)
        }
    }

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

    selected_datasets = {}

    # All datasets download
    if input_number == len(datasets):
        selected_datasets = datasets.copy()

    # Download selected dataset
    else:
        key, val = list(datasets.items())[input_number]
        selected_datasets = {key: val}

    for dataset, dataset_params in selected_datasets.items():
        LOGGER.info(f"Getting: {dataset}")

        if dataset == MEANTIME:
            # download all languages
            links = dataset_params[LINK].split(";")
            LOGGER.info(f"Downloading datasets for {str(len(links))} languages.")
            for link in links:
                gdown.download(link, dataset_params[ZIP], quiet=False)
                with zipfile.ZipFile(dataset_params[ZIP], 'r') as zip_ref:
                    zip_ref.extractall(dataset_params[FOLDER])

            # download required spacy packages
            for spacy_package in [SPACY_EN, SPACY_ES, SPACY_NL, SPACY_IT]:
                if not spacy.util.is_package(spacy_package):
                    spacy.cli.download(spacy_package)

        elif dataset == WEC_ENG:
            splits_files = ["Dev_Event_gold_mentions_validated.json",
                            "Test_Event_gold_mentions_validated.json",
                            "Train_Event_gold_mentions.json"]

            if not os.path.exists(dataset_params[FOLDER]):
                os.mkdir(dataset_params[FOLDER])

            for split_filename in splits_files:
                with open(hf_hub_download(dataset_params[LINK], filename=split_filename, repo_type="dataset"), encoding='utf-8') as cd:
                    local_file = json.load(cd)
                    with open(os.path.join(dataset_params[FOLDER], split_filename), "w") as file:
                        json.dump(local_file, file)

                    LOGGER.info(f'Downloaded {split_filename}')

            # download required spacy packages
            if not spacy.util.is_package(SPACY_EN):
                spacy.cli.download(SPACY_EN)

        elif dataset == ECB_PLUS:
            gdown.download(dataset_params[LINK], dataset_params[ZIP], quiet=False)
            with zipfile.ZipFile(dataset_params[ZIP], 'r') as zip_ref:
                zip_ref.extractall(dataset_params[FOLDER])

            gdown.download(
                "https://raw.githubusercontent.com/cltl/ecbPlus/master/ECB%2B_LREC2014/ECBplus_coreference_sentences.csv",
                os.path.join(os.getcwd(), ECB_PLUS, ECBPLUS_FOLDER_NAME, "ECBplus_coreference_sentences.csv"),
                quiet=False)

            # download required spacy packages
            if not spacy.util.is_package(SPACY_EN):
                spacy.cli.download(SPACY_EN)

        elif dataset == NEWSWCL50:
            gdown.download(dataset_params[LINK], dataset_params[ZIP], quiet=False)
            with zipfile.ZipFile(dataset_params[ZIP], 'r') as zip_ref:
                zip_ref.extractall(dataset_params[FOLDER])

            # download required spacy packages
            if not spacy.util.is_package(SPACY_EN):
                spacy.cli.download(SPACY_EN)

        elif dataset == NP4E:
            gdown.download(dataset_params[LINK], dataset_params[ZIP], quiet=False)
            with zipfile.ZipFile(dataset_params[ZIP], 'r') as zip_ref:
                zip_ref.extractall(dataset_params[FOLDER])
            os.rename(os.path.join(dataset_params[FOLDER], "mmax2"), os.path.join(dataset_params[FOLDER], NP4E_FOLDER_NAME))

            # download required spacy packages
            if not spacy.util.is_package(SPACY_EN):
                spacy.cli.download(SPACY_EN)

        elif dataset == GVC:
            for url, file_names in zip(dataset_params[LINK].split(";"),
                                       [["gold.conll", "system_input.conll", "verbose.conll"],
                                        ["dev.csv", "gvc_doc_to_event.csv", "test.csv", "train.csv"]]):
                for file_name in file_names:
                    gdown.download(f'{url}{file_name}',
                                   os.path.join(dataset_params[FOLDER],  file_name), quiet=False)

            # download required spacy packages
            if not spacy.util.is_package(SPACY_EN):
                spacy.cli.download(SPACY_EN)
        else:
            NotImplementedError(f'There is no data download script implemented for the dataset {dataset}. Please make sure '
                                f'that you have manully downloaded the raw data before parsing it. ')

    LOGGER.info("Setup was successful.")
