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

import bisect
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict, Set
from typing_extensions import Literal

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.data.processors.utils import InputFeatures

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import zero_pad
from deeppavlov.core.models.component import Component
from deeppavlov.models.preprocessors.mask import Mask

log = getLogger(__name__)


@register('torch_squad_transformers_preprocessor')
class TorchSquadTransformersPreprocessor(Component):
    """Tokenize text on subtokens, encode subtokens with their indices, create tokens and segment masks.

    Check details in :func:`bert_dp.preprocessing.convert_examples_to_features` function.

    Args:
        vocab_file: path to vocabulary
        do_lower_case: set True if lowercasing is needed
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        return_tokens: whether to return tuple of input features and tokens, or only input features

    Attributes:
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        return_tokens: whether to return tuple of input features and tokens, or only input features
        tokenizer: instance of Bert FullTokenizer

    """

    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = True,
                 max_seq_length: int = 512,
                 return_tokens: bool = False,
                 add_token_type_ids: bool = False,
                 **kwargs) -> None:
        self.max_seq_length = max_seq_length
        self.return_tokens = return_tokens
        self.add_token_type_ids = add_token_type_ids
        vocab_file = str(expand_path(vocab_file))
        self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)

    def __call__(self, question_batch: List[str], context_batch: Optional[List[str]] = None) -> Union[
        List[InputFeatures],
        Tuple[List[InputFeatures],
              List[List[str]]]]:
        """Tokenize and create masks.

        texts_a_batch and texts_b_batch are separated by [SEP] token

        Args:
            texts_a_batch: list of texts,
            texts_b_batch: list of texts, it could be None, e.g. single sentence classification task

        Returns:
            batch of :class:`transformers.data.processors.utils.InputFeatures` with subtokens, subtoken ids, \
                subtoken mask, segment mask, or tuple of batch of InputFeatures, batch of subtokens and batch of
                split paragraphs
        """

        if context_batch is None:
            context_batch = [None] * len(question_batch)

        input_features_batch, tokens_batch, split_context_batch = [], [], []
        for question, context in zip(question_batch, context_batch):
            question_list, context_list = [], []
            context_subtokens = self.tokenizer.tokenize(context)
            question_subtokens = self.tokenizer.tokenize(question)
            max_chunk_len = self.max_seq_length - len(question_subtokens) - 3
            if 0 < max_chunk_len < len(context_subtokens):
                number_of_chunks = math.ceil(len(context_subtokens) / max_chunk_len)
                sentences = context.split(". ")
                sentences = [f"{sentence}." for sentence in sentences if not sentence.endswith(".")]
                for chunk in np.array_split(sentences, number_of_chunks):
                    context_list += [' '.join(chunk)]
                    question_list += [question]
            else:
                context_list += [context]
                question_list += [question]

            input_features_list, tokens_list = [], []
            for question_elem, context_elem in zip(question_list, context_list):
                encoded_dict = self.tokenizer.encode_plus(
                    text=question_elem, text_pair=context_elem,
                    add_special_tokens=True,
                    max_length=self.max_seq_length,
                    truncation=True,
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='pt')
                if 'token_type_ids' not in encoded_dict:
                    if self.add_token_type_ids:
                        input_ids = encoded_dict['input_ids']
                        seq_len = input_ids.size(1)
                        sep = torch.where(input_ids == self.tokenizer.sep_token_id)[1][0].item()
                        len_a = min(sep + 1, seq_len)
                        len_b = seq_len - len_a
                        encoded_dict['token_type_ids'] = torch.cat((torch.zeros(1, len_a, dtype=int),
                                                                    torch.ones(1, len_b, dtype=int)), dim=1)
                    else:
                        encoded_dict['token_type_ids'] = torch.tensor([0])

                curr_features = InputFeatures(input_ids=encoded_dict['input_ids'],
                                              attention_mask=encoded_dict['attention_mask'],
                                              token_type_ids=encoded_dict['token_type_ids'],
                                              label=None)
                input_features_list.append(curr_features)
                if self.return_tokens:
                    tokens_list.append(self.tokenizer.convert_ids_to_tokens(encoded_dict['input_ids'][0]))

            input_features_batch.append(input_features_list)
            tokens_batch.append(tokens_list)
            split_context_batch.append(context_list)

        if self.return_tokens:
            return input_features_batch, tokens_batch, split_context_batch
        else:
            return input_features_batch, split_context_batch


@register('squad_bert_mapping')
class SquadBertMappingPreprocessor(Component):
    """Create mapping from BERT subtokens to their characters positions and vice versa.
        Args:
            do_lower_case: set True if lowercasing is needed
    """

    def __init__(self, do_lower_case: bool = True, *args, **kwargs):
        self.do_lower_case = do_lower_case

    def __call__(self, contexts_batch, bert_features_batch, subtokens_batch, **kwargs):
        subtok2chars_batch: List[List[Dict[int, int]]] = []
        char2subtoks_batch: List[List[Dict[int, int]]] = []

        for batch_counter, (context_list, features_list, subtokens_list) in \
                enumerate(zip(contexts_batch, bert_features_batch, subtokens_batch)):
            subtok2chars_list, char2subtoks_list = [], []
            for context, features, subtokens in zip(context_list, features_list, subtokens_list):
                if self.do_lower_case:
                    context = context.lower()
                context_start = subtokens.index('[SEP]') + 1
                idx = 0
                subtok2char: Dict[int, int] = {}
                char2subtok: Dict[int, int] = {}
                for i, subtok in list(enumerate(subtokens))[context_start:-1]:
                    subtok = subtok[2:] if subtok.startswith('##') else subtok
                    subtok_pos = context[idx:].find(subtok)
                    if subtok_pos == -1:
                        # it could be UNK
                        idx += 1  # len was at least one
                    else:
                        # print(k, '\t', t, p + idx)
                        idx += subtok_pos
                        subtok2char[i] = idx
                        for j in range(len(subtok)):
                            char2subtok[idx + j] = i
                        idx += len(subtok)
                subtok2chars_list.append(subtok2char)
                char2subtoks_list.append(char2subtok)
            subtok2chars_batch.append(subtok2chars_list)
            char2subtoks_batch.append(char2subtoks_list)
        return subtok2chars_batch, char2subtoks_batch


@register('squad_bert_ans_preprocessor')
class SquadBertAnsPreprocessor(Component):
    """Create answer start and end positions in subtokens.
        Args:
            do_lower_case: set True if lowercasing is needed
    """

    def __init__(self, do_lower_case: bool = True, *args, **kwargs):
        self.do_lower_case = do_lower_case

    def __call__(self, answers_raw, answers_start, char2subtoks, **kwargs):
        answers, starts, ends = [], [], []
        for answers_raw, answers_start, c2sub in zip(answers_raw, answers_start, char2subtoks):
            answers.append([])
            starts.append([])
            ends.append([])
            for ans, ans_st in zip(answers_raw, answers_start):
                if self.do_lower_case:
                    ans = ans.lower()
                try:
                    indices = {c2sub[0][i] for i in range(ans_st, ans_st + len(ans)) if i in c2sub[0]}
                    st = min(indices)
                    end = max(indices)
                except ValueError:
                    # 0 - CLS token
                    st, end = 0, 0
                    ans = ''
                starts[-1] += [st]
                ends[-1] += [end]
                answers[-1] += [ans]
        return answers, starts, ends


@register('squad_bert_ans_postprocessor')
class SquadBertAnsPostprocessor(Component):
    """Extract answer and create answer start and end positions in characters from subtoken positions."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, answers_start_batch, answers_end_batch, contexts_batch,
                 subtok2chars_batch, subtokens_batch, ind_batch, *args, **kwargs):
        answers = []
        starts = []
        ends = []
        for answer_st, answer_end, context_list, sub2c_list, subtokens_list, ind in \
                zip(answers_start_batch, answers_end_batch, contexts_batch, subtok2chars_batch, subtokens_batch,
                    ind_batch):
            sub2c = sub2c_list[ind]
            subtok = subtokens_list[ind][answer_end]
            context = context_list[ind]
            # CLS token is no_answer token
            if answer_st == 0 or answer_end == 0:
                answers += ['']
                starts += [-1]
                ends += [-1]
            else:
                st = self.get_char_position(sub2c, answer_st)
                end = self.get_char_position(sub2c, answer_end)

                subtok = subtok[2:] if subtok.startswith('##') else subtok
                answer = context[st:end + len(subtok)]
                answers += [answer]
                starts += [st]
                ends += [ends]
        return answers, starts, ends

    @staticmethod
    def get_char_position(sub2c, sub_pos):
        keys = list(sub2c.keys())
        found_idx = bisect.bisect(keys, sub_pos)
        if found_idx == 0:
            return sub2c[keys[0]]

        return sub2c[keys[found_idx - 1]]


@register('torch_transformers_generative_qa_preprocessor')
class TorchTransformersGenerativeQAPreprocessor(Component):
    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = True,
                 max_seq_length: int = 512,
                 return_tokens: bool = False,
                 answer_maxlength: int = 20,
                 **kwargs) -> None:
        self.max_seq_length = max_seq_length
        self.return_tokens = return_tokens
        if Path(vocab_file).is_file():
            vocab_file = str(expand_path(vocab_file))
            self.tokenizer = AutoTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)
        self.answer_maxlength = answer_maxlength

    def __call__(self, questions_batch: List[str], contexts_batch: List[List[str]], targets_batch: List[str] = None):
        input_ids_batch, attention_mask_batch, lengths = [], [], []
        for question, contexts in zip(questions_batch, contexts_batch):
            input_ids_list, attention_mask_list = [], []
            if isinstance(contexts, str):
                contexts = [contexts]
            for context in contexts:
                encoded_dict = self.tokenizer.encode_plus(question, context)
                input_ids_list += encoded_dict["input_ids"]
                attention_mask_list += encoded_dict["attention_mask"]
            lengths.append(len(input_ids_list))
            input_ids_batch.append(input_ids_list)
            attention_mask_batch.append(attention_mask_list)
        max_length = min(max(lengths), self.max_seq_length)
        for i in range(len(input_ids_batch)):
            for j in range(max_length - len(input_ids_batch[i])):
                input_ids_batch[i].append(0)
                attention_mask_batch[i].append(0)
        
        target_ids_batch = None
        if targets_batch is not None:
            targets_batch = list(targets_batch)
            target_encoding = self.tokenizer.batch_encode_plus(
                targets_batch,
                max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
                pad_to_max_length=True,
                truncation=True if self.answer_maxlength > 0 else False,
            )
            target_ids_batch = target_encoding["input_ids"]
        
        return input_ids_batch, attention_mask_batch, target_ids_batch
