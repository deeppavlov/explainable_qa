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

from rdflib import Graph
from rdflib_hdt import HDTStore, optimize_sparql
from rdflib import URIRef

log = getLogger(__name__)

def freebase_name_filler(ID):
    PREFIX = 'http://rdf.freebase.com/ns/'
    return """PREFIX : <http://rdf.freebase.com/ns/>
            SELECT (?x0 AS ?name) WHERE { :""" + ID.replace(PREFIX, "") + """ :type.object.name ?x0 }"""

class QueryExecutor(Component):
     """Class for executing SPARQL queries"""
    def __init__(self, path_to_hdt, name_filler=freebase_name_filler, **kwargs):
        """
        Args:
            freq_dict_filename: hdt file containing knowledge graph
            name_filler: function which forms SPARQL-query for getting name from id
            **kwargs:
        """
        self.graph = Graph(store=HDTStore(path_to_hdt))
        self.name_filler = name_filler
        optimize_sparql()

    def get_answer_name(self, ID):
        query = self.name_filler(ID)
        result_iter = self.graph.query(query)
        try:
            result = next(iter(result_iter))
        except StopIteration:
            return "No name"
        return str(result.name)

    def __call__(self, sparql_queries):
        answer_arguments, answer_names = [], []
        for query in sparql_queries:
            answer_arguments.append([])
            answer_names.append([])
            results = self.graph.query(query)
            for result in results:
                answer_arguments[-1].append(str(result.value))
                if isinstance(result.value, URIRef):
                    answer_names[-1].append(self.get_answer_name(str(result.value)))
                else:
                    answer_names[-1].append("")
        return answer_arguments, answer_names
