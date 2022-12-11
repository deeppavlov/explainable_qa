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

    def __call__(self, questions_batch, answers_batch, scores_batch, logits_batch, doc_ids_batch, doc_pages_batch,
                       places_batch, sentences_batch):
        f_answers_batch, f_scores_batch, f_logits_batch, f_places_batch, f_sentences_batch = [], [], [], [], []
        for question, answers, scores, logits, doc_ids, doc_pages, places, sentences in \
                zip(questions_batch, answers_batch, scores_batch, logits_batch, doc_ids_batch, doc_pages_batch,
                    places_batch, sentences_batch):
            answers_lemm = [self.sanitize(answer) for answer in answers]
            answer_cnts_dict = {}
            answer_pages = set()
            for answer_lemm, doc_page, score in zip(answers_lemm, doc_pages, scores):
                if answer_lemm not in answer_cnts_dict and (answer_lemm, doc_page) not in answer_pages:
                    answer_cnts_dict[answer_lemm] = 1
                    answer_pages.add((answer_lemm, doc_page))
                elif score > 0.95 and (answer_lemm, doc_page) not in answer_pages:
                    answer_cnts_dict[answer_lemm] += 1
                    answer_pages.add((answer_lemm, doc_page))

            ngrams_inters = []
            question_ngrams = self.extract_ngrams(question)
            for sentence in sentences:
                sentence_ngrams = self.extract_ngrams(sentence)
                if set(sentence_ngrams).intersection(question_ngrams):
                    ngrams_inters.append(1)
                else:
                    ngrams_inters.append(0)

            answer_cnts = [answer_cnts_dict.get(ans, 0) for ans in answers_lemm]
            answers_info = list(zip(answers, answers_lemm, answer_cnts, ngrams_inters, scores, logits, doc_ids, doc_pages, places,
                                    sentences))
            answers_info = sorted(answers_info, key=lambda x: (x[2], x[3], x[4], x[5]), reverse=True)
            answers_info = answers_info[:self.top_n]
            f_answers_batch.append([elem[0] for elem in answers_info])
            f_scores_batch.append([elem[4] for elem in answers_info])
            f_logits_batch.append([elem[5] for elem in answers_info])
            f_places_batch.append([elem[8] for elem in answers_info])
            f_sentences_batch.append([elem[9] for elem in answers_info])
        return f_answers_batch, f_scores_batch, f_logits_batch, f_places_batch, f_sentences_batch

    def normal_form(self, word):
        morph_parse_tok = self.morph.parse(word)[0]
        if morph_parse_tok:
            normal_form = morph_parse_tok.normal_form
        else:
            normal_form = word
        return normal_form
    
    def extract_ngrams(self, text):
        ngrams = []
        text_tokens = re.findall(self.re_tokenizer, text)
        text_tokens = [tok for tok in text_tokens if tok not in self.stopwords]
        text_tokens = [self.normal_form(tok).lower() for tok in text_tokens]
        chunks = []
        chunk = []
        for tok in text_tokens:
            if tok in punctuation and chunk:
                chunks.append(chunk)
                chunk = []
            if tok not in punctuation:
                chunk.append(tok)
        for chunk in chunks:
            if len(chunk) >= 4:
                for i in range(len(chunk) - 3):
                    ngrams.append(" ".join(chunk[i:i + 4]))
        return ngrams

    def sanitize(self, substr):
        substr = substr.replace("а́", "а")
        words = re.findall(self.re_tokenizer, substr)
        words = [tok for tok in words if (len(tok) > 0 and tok not in punctuation and tok not in self.stopwords)]
        if any([word[0].isupper() for word in words]):
            words = [word for word in words if word[0].isupper()]
        words = [self.normal_form(word) for word in words]
        return " ".join(words).lower()
