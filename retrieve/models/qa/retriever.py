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


import os
import time
import logging
from typing import List
import torch
import faiss
import numpy as np
from tqdm import trange
from transformers import AutoTokenizer

from models.qa.biencoder import BiEncoder
from models.qa.index import FaissBinaryIndex, FaissIndex

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Retriever(object):
    def __init__(self, index: FaissIndex, biencoder: BiEncoder):
        self.index = index
        self._biencoder = biencoder
        self._tokenizer = AutoTokenizer.from_pretrained(biencoder.hparams.base_pretrained_model, use_fast=True)

    def encode_queries(self, queries: List[str], batch_size: int = 256) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for start in trange(0, len(queries), batch_size):
                model_inputs = self._tokenizer.batch_encode_plus(
                    queries[start : start + batch_size],
                    return_tensors="pt",
                    max_length=self._biencoder.hparams.max_query_length,
                    padding="max_length",
                    truncation=True,
                )

                model_inputs = {k: v.to(self._biencoder.device) for k, v in model_inputs.items()}
                emb = self._biencoder.query_encoder(**model_inputs).cpu().numpy()
                embeddings.append(emb)

        return np.vstack(embeddings)

    def search(self, query_embeddings: np.ndarray, k: int, **faiss_index_options):
        scores_list, ids_list = self.index.search(query_embeddings, k, **faiss_index_options)
        return scores_list, ids_list


@register("passage_retriever")
class PassageRetriever(Component, Serializable):
    def __init__(
        self,
        pretrained_model,
        load_path,
        index_file,
        query_encoder_file,
        binary: bool=False,
        top_n: int=50,
        nprobe: int=10,
        device: str = "gpu",
        *args,
        **kwargs,
    ):
        super().__init__(save_path=None, load_path=load_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.index_file = index_file
        self.binary = binary
        self.top_n = top_n
        self.nprobe = nprobe
        self.hparams = {
            "base_pretrained_model": pretrained_model,
            "load_path": f"{self.load_path}/{query_encoder_file}",
            "max_query_length": 256,
        }
        self.load()


    def load(self):
        self.biencoder = BiEncoder(self.hparams)
        checkpoint = torch.load(self.hparams["load_path"], map_location=self.device)
        self.biencoder.query_encoder.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.biencoder.eval()
        self.biencoder.freeze()
        if self.binary:
            base_index = faiss.read_index_binary(str(self.load_path / self.index_file))
            base_index.nprobe = self.nprobe
            self.index = FaissBinaryIndex(base_index)
        else:
            base_index = faiss.read_index(str(self.load_path / self.index_file))
            base_index.nprobe = self.nprobe
            self.index = FaissIndex(base_index)
        self.retriever = Retriever(self.index, self.biencoder)

    def save(self) -> None:
        pass

    def __call__(self, queries: List[str]):
        queries = [query.lower() for query in queries]
        query_embeddings = self.retriever.encode_queries(queries)
        tm1 = time.time()
        scores_batch, ids_batch = self.retriever.search(query_embeddings, k=self.top_n)
        tm2 = time.time()
        logger.info(f"time of ranking {tm2 - tm1}")
        ids_batch = ids_batch.tolist()

        return ids_batch