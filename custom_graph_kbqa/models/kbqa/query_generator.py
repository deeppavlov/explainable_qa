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
from collections import defaultdict
from logging import getLogger
from typing import Tuple, List, Optional, Union, Dict, Any, Set
from models.kbqa.s_expression_to_sparql import lisp_to_sparql
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register

log = getLogger(__name__)

@register('query_generator')
class QueryGenerator(Component):
     """Class for generation SPARQL queries for S-expressions"""
    def __init__(self, **kwargs):
        pass

    def gen_one(self, s_expr):
        return lisp_to_sparql(s_expr)

    def __call__(self, s_expr_batch):
        return [self.gen_one(s_expr) for s_expr in s_expr_batch]
