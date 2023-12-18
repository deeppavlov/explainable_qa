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
        vocab_file = str(expand_path(vocab_file))
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
