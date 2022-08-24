import xml.etree.ElementTree as ET
import os
import json
import sys
import string
import spacy
import copy
import re
import pandas as pd
import numpy as np
from insert_whitespace import append_text
from nltk import Tree
from tqdm import tqdm
import warnings
from setup import *
from logger import LOGGER

# opens and loads the newsplease-format out of the json file: _sample_doc.json
with open(path_sample, "r") as file:
    newsplease_format = json.load(file)

WECENG_PARSING_FOLDER = os.path.join(os.getcwd())
OUT_PATH = os.path.join(WECENG_PARSING_FOLDER, OUTPUT_FOLDER_NAME)
source_path = os.path.join(WECENG_PARSING_FOLDER, WECENG_FOLDER_NAME)
result_path = os.path.join(OUT_PATH, 'test_parsing')

def conv_files(path):


if __name__ == '__main__':
    LOGGER.info(f"Processing WEC-Eng {source_path[-34:].split('_')[2]}.")
    conv_files(source_path)
