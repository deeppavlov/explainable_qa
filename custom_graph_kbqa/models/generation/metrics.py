# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
from collections import Counter
from typing import List

import nltk
import pymorphy2
from nltk.corpus import stopwords


nltk.download('stopwords')

morph = pymorphy2.MorphAnalyzer()
stopwords = set(stopwords.words("english"))

def normal_form(word):
    morph_parse_tok = morph.parse(word)[0]
    if morph_parse_tok:
        normal_form = morph_parse_tok.normal_form
    else:
        normal_form = word
    return normal_form


def squad_exact_match(y_true: List[List[str]], y_predicted: List[str]) -> float:
    """ Calculates Exact Match score between y_true and y_predicted
        EM score uses the best matching y_true answer:
            if y_pred equal at least to one answer in y_true then EM = 1, else EM = 0
        Skips examples without an answer.
    Args:
        y_true: list of correct answers (correct answers are represented by list of strings)
        y_predicted: list of predicted answers
    Returns:
        exact match score : float
    """
    EM_total = 0
    count = 0
    for ground_truth, prediction in zip(y_true, y_predicted):
        if len(ground_truth[0]) == 0:
            # skip empty answers
            continue
        count += 1
        EMs = [int(normalize_answer(gt) == normalize_answer(prediction)) for gt in ground_truth]
        EM_total += max(EMs)
    return 100 * EM_total / count if count > 0 else 0


def squad_f1(y_true: List[List[str]], y_predicted: List[str]) -> float:
    """ Calculates F-1 score between y_true and y_predicted
        F-1 score uses the best matching y_true answer

        Skips examples without an answer.
    Args:
        y_true: list of correct answers (correct answers are represented by list of strings)
        y_predicted: list of predicted answers
    Returns:
        F-1 score : float
    """
    f1_total = 0.0
    count = 0
    for ground_truth, prediction in zip(y_true, y_predicted):
        if len(ground_truth[0]) == 0:
            # skip empty answers
            continue
        count += 1
        prediction_tokens = normalize_answer(prediction)
        f1s = []
        for gt in ground_truth:
            gt_tokens = normalize_answer(gt)
            common = Counter(prediction_tokens) & Counter(gt_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                f1s.append(0.0)
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(gt_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            f1s.append(f1)
        f1_total += max(f1s)
    return 100 * f1_total / count if count > 0 else 0


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    tokens = white_space_fix(remove_articles(remove_punc(lower(s)))).split()
    tokens = [tok for tok in tokens if tok not in stopwords]
    tokens = [normal_form(tok) for tok in tokens]
    return tokens
