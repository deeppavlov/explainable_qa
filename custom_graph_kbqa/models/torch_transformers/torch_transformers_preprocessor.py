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

import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict, Set, Any

import nltk
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


@register('torch_transformers_preprocessor')
class TorchTransformersPreprocessor(Component):
    """Tokenize text on subtokens, encode subtokens with their indices, create tokens and segment masks.

    Args:
        vocab_file: A string, the `model id` of a predefined tokenizer hosted inside a model repo on huggingface.co or
            a path to a `directory` containing vocabulary files required by the tokenizer.
        do_lower_case: set True if lowercasing is needed
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens

    Attributes:
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        tokenizer: instance of Bert FullTokenizer

    """

    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = True,
                 max_seq_length: int = 512,
                 **kwargs) -> None:
        self.max_seq_length = max_seq_length
        # vocab_file = str(expand_path(vocab_file))
        self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)

    def __call__(self, texts_a: List[str], texts_b: Optional[List[str]] = None) -> Union[List[InputFeatures],
                                                                                         Tuple[List[InputFeatures],
                                                                                               List[List[str]]]]:
        """Tokenize and create masks.
        texts_a and texts_b are separated by [SEP] token
        Args:
            texts_a: list of texts,
            texts_b: list of texts, it could be None, e.g. single sentence classification task
        Returns:
            batch of :class:`transformers.data.processors.utils.InputFeatures` with subtokens, subtoken ids, \
                subtoken mask, segment mask, or tuple of batch of InputFeatures and Batch of subtokens
        """

        # in case of iterator's strange behaviour
        if isinstance(texts_a, tuple):
            texts_a = list(texts_a)

        input_features = self.tokenizer(text=texts_a,
                                        text_pair=texts_b,
                                        add_special_tokens=True,
                                        max_length=self.max_seq_length,
                                        padding='max_length',
                                        return_attention_mask=True,
                                        truncation=True,
                                        return_tensors='pt')
        return input_features


@register('torch_transformers_entity_ranker_preprocessor')
class TorchTransformersEntityRankerPreprocessor(Component):
    """Class for tokenization of text into subtokens, encoding of subtokens with indices and obtaining positions of
    special [ENT]-tokens
    Args:
        vocab_file: path to vocabulary
        do_lower_case: set True if lowercasing is needed
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        special_tokens: list of special tokens
        special_token_id: id of special token
        return_special_tokens_pos: whether to return positions of found special tokens
    """

    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = False,
                 max_seq_length: int = 512,
                 special_tokens: List[str] = None,
                 special_token_id: int = None,
                 return_special_tokens_pos: bool = False,
                 **kwargs) -> None:
        self.max_seq_length = max_seq_length
        self.do_lower_case = do_lower_case
        if Path(vocab_file).is_file():
            vocab_file = str(expand_path(vocab_file))
            self.tokenizer = AutoTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)
        if special_tokens is not None:
            special_tokens_dict = {'additional_special_tokens': special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
        self.special_token_id = special_token_id
        self.return_special_tokens_pos = return_special_tokens_pos

    def __call__(self, texts_a: List[str]) -> Tuple[Any, List[int]]:
        """Tokenize and find special tokens positions.
        Args:
            texts_a: list of texts,
        Returns:
            batch of :class:`transformers.data.processors.utils.InputFeatures` with subtokens, subtoken ids, \
                subtoken mask, segment mask, or tuple of batch of InputFeatures and Batch of subtokens
            batch of indices of special token ids in input ids sequence
        """
        # in case of iterator's strange behaviour
        if isinstance(texts_a, tuple):
            texts_a = list(texts_a)
        if self.do_lower_case:
            texts_a = [text.lower() for text in texts_a]
        lengths = []
        input_ids_batch = []
        for text_a in texts_a:
            encoding = self.tokenizer.encode_plus(
                text_a, add_special_tokens=True, pad_to_max_length=True, return_attention_mask=True)
            input_ids = encoding["input_ids"]
            input_ids_batch.append(input_ids)
            lengths.append(len(input_ids))

        max_length = min(max(lengths), self.max_seq_length)
        input_features = self.tokenizer(text=texts_a,
                                        add_special_tokens=True,
                                        max_length=max_length,
                                        padding='max_length',
                                        return_attention_mask=True,
                                        truncation=True,
                                        return_tensors='pt')
        special_tokens_pos = []
        for input_ids_list in input_ids_batch:
            found_n = -1
            for n, input_id in enumerate(input_ids_list):
                if input_id == self.special_token_id:
                    found_n = n
                    break
            if found_n == -1:
                found_n = 0
            special_tokens_pos.append(found_n)

        if self.return_special_tokens_pos:
            return input_features, special_tokens_pos
        else:
            return input_features


@register('rel_ranking_preprocessor')
class RelRankingPreprocessor(Component):
    """Class for tokenization of text and relation labels
    Args:
        vocab_file: path to vocabulary
        add_special_tokens: special_tokens_list
        do_lower_case: set True if lowercasing is needed
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
    """

    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = True,
                 max_seq_length: int = 512,
                 **kwargs) -> None:
        self.max_seq_length = max_seq_length
        # vocab_file = str(expand_path(vocab_file))
        self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)

    def __call__(self, questions_batch: List[List[str]], rels_batch: List[List[str]] = None) -> Dict[str, torch.tensor]:
        """Tokenize questions and relations
        texts_a and texts_b are separated by [SEP] token
        Args:
            questions_batch: list of texts,
            rels_batch: list of relations list
        Returns:
            batch of :class:`transformers.data.processors.utils.InputFeatures` with subtokens, subtoken ids, \
                subtoken mask, segment mask, or tuple of batch of InputFeatures and Batch of subtokens
        """
        lengths, proc_rels_batch = [], []
        for question, rels_list in zip(questions_batch, rels_batch):
            if isinstance(rels_list, list):
                rels_str = " ".join(rels_list)
            else:
                rels_str = rels_list
            encoding = self.tokenizer.encode_plus(text=question, text_pair=rels_str,
                                                  return_attention_mask=True, add_special_tokens=True,
                                                  truncation=True)
            lengths.append(len(encoding["input_ids"]))
            proc_rels_batch.append(rels_str)
        max_len = max(lengths)
        input_ids_batch, attention_mask_batch, token_type_ids_batch = [], [], []
        for question, rels_list in zip(questions_batch, proc_rels_batch):
            encoding = self.tokenizer.encode_plus(text=question, text_pair=rels_list,
                                                  truncation=True, max_length=max_len,
                                                  pad_to_max_length=True, return_attention_mask=True)
            input_ids_batch.append(encoding["input_ids"])
            attention_mask_batch.append(encoding["attention_mask"])
            if "token_type_ids" in encoding:
                token_type_ids_batch.append(encoding["token_type_ids"])
            else:
                token_type_ids_batch.append([0])
        input_features = {"input_ids": torch.LongTensor(input_ids_batch),
                          "attention_mask": torch.LongTensor(attention_mask_batch),
                          "token_type_ids": torch.LongTensor(token_type_ids_batch)}
        return input_features


@register('path_ranking_preprocessor')
class PathRankingPreprocessor(Component):
    def __init__(self,
                 vocab_file: str,
                 add_special_tokens: List[str] = None,
                 do_lower_case: bool = True,
                 max_seq_length: int = 67,
                 num_neg_samples: int = 499,
                 **kwargs) -> None:
        self.max_seq_length = max_seq_length
        self.num_neg_samples = num_neg_samples
        vocab_file = str(expand_path(vocab_file))
        self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)
        self.add_special_tokens = add_special_tokens
        if self.add_special_tokens:
            special_tokens_dict = {'additional_special_tokens': add_special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)

    def __call__(self, questions_batch: List[str], rels_batch: List[List[List[str]]]):
        lengths, proc_rels_batch = [], []
        for question, rels_list in zip(questions_batch, rels_batch):
            proc_rels_list = []
            for rels in rels_list:
                if isinstance(rels, str):
                    rels = [rels]
                rels_str = ""
                if len(rels) == 1:
                    if self.add_special_tokens:
                        rels_str = f"<one_rel> {rels[0]} </one_rel>"
                    else:
                        rels_str = rels[0]
                elif len(rels) == 2:
                    if rels[0] == rels[1]:
                        rels_str = f"<double> {rels[0]} </double>"
                    else:
                        rels_str = f"<first_rel> {rels[0]} <mid> {rels[1]} </second_rel>"
                encoding = self.tokenizer.encode_plus(text=question, text_pair=rels_str,
                                                      return_attention_mask=True, add_special_tokens=True,
                                                      truncation=True)
                lengths.append(len(encoding["input_ids"]))
                proc_rels_list.append(rels_str)
            proc_rels_batch.append(proc_rels_list)

        max_len = min(max(lengths), self.max_seq_length)
        input_ids_batch, attention_mask_batch, token_type_ids_batch = [], [], []
        for question, rels_list in zip(questions_batch, proc_rels_batch):
            input_ids_list, attention_mask_list, token_type_ids_list = [], [], []
            for rels_str in rels_list:
                encoding = self.tokenizer.encode_plus(text=question, text_pair=rels_str,
                                                      truncation=True, max_length=max_len, add_special_tokens=True,
                                                      pad_to_max_length=True, return_attention_mask=True)
                input_ids_list.append(encoding["input_ids"])
                attention_mask_list.append(encoding["attention_mask"])
                if "token_type_ids" in encoding:
                    token_type_ids_list.append(encoding["token_type_ids"])
                else:
                    token_type_ids_list.append([0])
            input_ids_batch.append(input_ids_list)
            attention_mask_batch.append(attention_mask_list)
            token_type_ids_batch.append(token_type_ids_list)
        input_features = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch,
                          "token_type_ids": token_type_ids_batch}
        return input_features


@register('torch_transformers_ner_preprocessor')
class TorchTransformersNerPreprocessor(Component):
    """
    Takes tokens and splits them into bert subtokens, encodes subtokens with their indices.
    Creates a mask of subtokens (one for the first subtoken, zero for the others).

    If tags are provided, calculates tags for subtokens.

    Args:
        vocab_file: path to vocabulary
        do_lower_case: set True if lowercasing is needed
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        max_subword_length: replace token to <unk> if it's length is larger than this
            (defaults to None, which is equal to +infinity)
        token_masking_prob: probability of masking token while training
        provide_subword_tags: output tags for subwords or for words
        subword_mask_mode: subword to select inside word tokens, can be "first" or "last"
            (default="first")

    Attributes:
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        max_subword_length: rmax lenght of a bert subtoken
        tokenizer: instance of Bert FullTokenizer
    """

    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = False,
                 max_seq_length: int = 512,
                 max_subword_length: int = None,
                 token_masking_prob: float = 0.0,
                 provide_subword_tags: bool = False,
                 subword_mask_mode: str = "first",
                 **kwargs):
        self._re_tokenizer = re.compile(r"[\w']+|[^\w ]")
        self.provide_subword_tags = provide_subword_tags
        self.mode = kwargs.get('mode')
        self.max_seq_length = max_seq_length
        self.max_subword_length = max_subword_length
        self.subword_mask_mode = subword_mask_mode
        vocab_file = str(expand_path(vocab_file))
        self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)
        self.token_masking_prob = token_masking_prob

    def __call__(self,
                 tokens: Union[List[List[str]], List[str]],
                 tags: List[List[str]] = None,
                 **kwargs):
        tokens_offsets_batch = [[] for _ in tokens]
        if isinstance(tokens[0], str):
            tokens_batch = []
            tokens_offsets_batch = []
            for s in tokens:
                tokens_list = []
                tokens_offsets_list = []
                for elem in re.finditer(self._re_tokenizer, s):
                    tokens_list.append(elem[0])
                    tokens_offsets_list.append((elem.start(), elem.end()))
                tokens_batch.append(tokens_list)
                tokens_offsets_batch.append(tokens_offsets_list)
            tokens = tokens_batch
        subword_tokens, subword_tok_ids, startofword_markers, subword_tags = [], [], [], []
        for i in range(len(tokens)):
            toks = tokens[i]
            ys = ['O'] * len(toks) if tags is None else tags[i]
            assert len(toks) == len(ys), \
                f"toks({len(toks)}) should have the same length as ys({len(ys)})"
            sw_toks, sw_marker, sw_ys = \
                self._ner_bert_tokenize(toks,
                                        ys,
                                        self.tokenizer,
                                        self.max_subword_length,
                                        mode=self.mode,
                                        subword_mask_mode=self.subword_mask_mode,
                                        token_masking_prob=self.token_masking_prob)
            if self.max_seq_length is not None:
                if len(sw_toks) > self.max_seq_length:
                    raise RuntimeError(f"input sequence after bert tokenization"
                                       f" shouldn't exceed {self.max_seq_length} tokens.")
            subword_tokens.append(sw_toks)
            subword_tok_ids.append(self.tokenizer.convert_tokens_to_ids(sw_toks))
            startofword_markers.append(sw_marker)
            subword_tags.append(sw_ys)
            assert len(sw_marker) == len(sw_toks) == len(subword_tok_ids[-1]) == len(sw_ys), \
                f"length of sow_marker({len(sw_marker)}), tokens({len(sw_toks)})," \
                f" token ids({len(subword_tok_ids[-1])}) and ys({len(ys)})" \
                f" for tokens = `{toks}` should match"

        subword_tok_ids = zero_pad(subword_tok_ids, dtype=int, padding=0)
        startofword_markers = zero_pad(startofword_markers, dtype=int, padding=0)
        attention_mask = Mask()(subword_tokens)

        if tags is not None:
            if self.provide_subword_tags:
                return tokens, subword_tokens, subword_tok_ids, \
                       attention_mask, startofword_markers, subword_tags
            else:
                nonmasked_tags = [[t for t in ts if t != 'X'] for ts in tags]
                for swts, swids, swms, ts in zip(subword_tokens,
                                                 subword_tok_ids,
                                                 startofword_markers,
                                                 nonmasked_tags):
                    if (len(swids) != len(swms)) or (len(ts) != sum(swms)):
                        log.warning('Not matching lengths of the tokenization!')
                        log.warning(f'Tokens len: {len(swts)}\n Tokens: {swts}')
                        log.warning(f'Markers len: {len(swms)}, sum: {sum(swms)}')
                        log.warning(f'Masks: {swms}')
                        log.warning(f'Tags len: {len(ts)}\n Tags: {ts}')
                return tokens, subword_tokens, subword_tok_ids, \
                       attention_mask, startofword_markers, nonmasked_tags
        return tokens, subword_tokens, subword_tok_ids, startofword_markers, attention_mask, tokens_offsets_batch

    @staticmethod
    def _ner_bert_tokenize(tokens: List[str],
                           tags: List[str],
                           tokenizer: AutoTokenizer,
                           max_subword_len: int = None,
                           mode: str = None,
                           subword_mask_mode: str = "first",
                           token_masking_prob: float = None) -> Tuple[List[str], List[int], List[str]]:
        do_masking = (mode == 'train') and (token_masking_prob is not None)
        do_cutting = (max_subword_len is not None)
        tokens_subword = ['[CLS]']
        startofword_markers = [0]
        tags_subword = ['X']
        for token, tag in zip(tokens, tags):
            token_marker = int(tag != 'X')
            subwords = tokenizer.tokenize(token)
            if not subwords or (do_cutting and (len(subwords) > max_subword_len)):
                tokens_subword.append('[UNK]')
                startofword_markers.append(token_marker)
                tags_subword.append(tag)
            else:
                if do_masking and (random.random() < token_masking_prob):
                    tokens_subword.extend(['[MASK]'] * len(subwords))
                else:
                    tokens_subword.extend(subwords)
                if subword_mask_mode == "last":
                    startofword_markers.extend([0] * (len(subwords) - 1) + [token_marker])
                else:
                    startofword_markers.extend([token_marker] + [0] * (len(subwords) - 1))
                tags_subword.extend([tag] + ['X'] * (len(subwords) - 1))

        tokens_subword.append('[SEP]')
        startofword_markers.append(0)
        tags_subword.append('X')
        return tokens_subword, startofword_markers, tags_subword
