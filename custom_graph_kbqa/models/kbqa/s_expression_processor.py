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

import itertools
import json
from logging import getLogger
from typing import Tuple, List, Dict, Optional, Union, Any, Set
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable

import re

class SExpressionProcessor(Component):
     """Class for obtaining S-expression from raw S-expression (from generative model output)
     Performs replacement of entities and relations labels by their names"""
    def __init__(self, entity_linker, **kwargs):
        self.entity_linker = entity_linker

    def process_one(self, s_expression_raw, relation_ids, relations_labels):
        for ID, label in zip(relation_ids, relations_labels):
            s_expression_raw = s_expression_raw.replace(f'[{label}]', ID)

        extracted_entities = re.findall(r'\[(.*)\]', s_expression_raw)
        tags = ['e'] * len(extracted_entities)
        probas = [0] * len(extracted_entities)
        entities_ids = self.entity_linker([extracted_entities], [tags], [probas])[0][0]
        entities_ids = [e[0] for e in entities_ids]
        
        for ID, label in zip(entities_ids, extracted_entities):
            s_expression_raw = s_expression_raw.replace(f'[{label}]', ID)
        
        return s_expression_raw

    def __call__(self, 
                s_expression_raw_batch, 
                relations_ids_batch, 
                relations_labels_batch):
        return [self.process_one(s_expression_raw, relations_ids, relations_labels)
                for s_expression_raw, relations_ids, relations_labels in 
                zip(s_expression_raw_batch, relations_ids_batch, relations_labels_batch)]
