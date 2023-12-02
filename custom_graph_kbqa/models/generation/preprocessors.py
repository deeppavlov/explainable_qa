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
        # vocab_file = str(expand_path(vocab_file))
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
            for j in range(len(input_ids_batch[i]) - max_length):
                input_ids_batch[i].pop()
                attention_mask_batch[i].pop()
        
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
