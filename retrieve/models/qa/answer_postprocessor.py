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
from collections import Counter
from logging import getLogger
from operator import itemgetter
from string import punctuation
from typing import List, Union, Tuple, Optional

import nltk
import pymorphy2
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component
from nltk.corpus import stopwords


logger = getLogger(__name__)
nltk.download("stopwords")


@register("answer_postprocessor")
class AnswerPostprocessor(Component):
    def __init__(self, top_n, **kwargs):
        self.top_n = top_n
        self.re_tokenizer = re.compile(r"[\w']+|[^\w ]")
        self.stopwords = set(stopwords.words("russian"))
        self.morph = pymorphy2.MorphAnalyzer()

    def __call__(self, answers_batch, scores_batch, logits_batch, doc_ids_batch, doc_pages_batch, places_batch,
                       sentences_batch):
        f_answers_batch, f_scores_batch, f_logits_batch, f_places_batch, f_sentences_batch = [], [], [], [], []
        for answers, scores, logits, doc_ids, doc_pages, places, sentences in \
                zip(answers_batch, scores_batch, logits_batch, doc_ids_batch, doc_pages_batch, places_batch,
                    sentences_batch):
            answers_lemm = [self.sanitize(answer) for answer in answers]
            answer_cnts_dict = {}
            for answer_lemm, score in zip(answers_lemm, scores):
                if answer_lemm not in answer_cnts_dict:
                    answer_cnts_dict[answer_lemm] = 1
                elif score > 0.95:
                    answer_cnts_dict[answer_lemm] += 1
            answer_cnts = [answer_cnts_dict.get(ans, 0) for ans in answers_lemm]
            answers_info = list(zip(answers, answers_lemm, answer_cnts, scores, logits, doc_ids, doc_pages, places,
                                    sentences))
            answers_info = sorted(answers_info, key=lambda x: (x[2], x[3], x[4]), reverse=True)
            answers_info = answers_info[:self.top_n]
            f_answers_batch.append([elem[0] for elem in answers_info])
            f_scores_batch.append([elem[3] for elem in answers_info])
            f_logits_batch.append([elem[4] for elem in answers_info])
            f_places_batch.append([elem[7] for elem in answers_info])
            f_sentences_batch.append([elem[8] for elem in answers_info])
        return f_answers_batch, f_scores_batch, f_logits_batch, f_places_batch, f_sentences_batch

    def normal_form(self, word):
        morph_parse_tok = self.morph.parse(word)[0]
        if morph_parse_tok:
            normal_form = morph_parse_tok.normal_form
        else:
            normal_form = word
        return normal_form

    def sanitize(self, substr):
        substr = substr.replace("а́", "а")
        words = re.findall(self.re_tokenizer, substr)
        words = [tok for tok in words if (len(tok) > 0 and tok not in punctuation and tok not in self.stopwords)]
        if any([word[0].isupper() for word in words]):
            words = [word for word in words if word[0].isupper()]
        words = [self.normal_form(word) for word in words]
        return " ".join(words).lower()
