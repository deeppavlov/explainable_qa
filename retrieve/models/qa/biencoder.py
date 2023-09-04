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
from argparse import Namespace
from typing import Dict, List, Optional, Tuple, Union

import torch
from pytorch_lightning.core import LightningModule
from transformers import (
    BertConfig,
    BertModel,
    BertTokenizerFast,
)


class BiEncoderModel(BertModel):
    def __init__(self, config: BertConfig, hparams: Namespace):
        super().__init__(config)

        self.hparams = hparams
        if getattr(hparams, "projection_dim_size", None) is not None:
            self.dense = torch.nn.Linear(config.hidden_size, hparams.projection_dim_size)
            self.layer_norm = torch.nn.LayerNorm(hparams.projection_dim_size, eps=config.layer_norm_eps)

        self.init_weights()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        sequence_output = super().forward(*args, **kwargs)[0]
        cls_output = sequence_output[:, 0, :].contiguous()
        if getattr(self.hparams, "projection_dim_size", None) is not None:
            cls_output = self.layer_norm(self.dense(cls_output))

        return cls_output


class BiEncoder(LightningModule):
    def __init__(self, hparams: Optional[Union[Namespace, dict]]):
        super().__init__()

        if isinstance(hparams, dict):
            if "binary_passage" in hparams and hparams["binary_passage"]:
                hparams["binary"] = True
            hparams = Namespace(**hparams)

        self.hparams.update(vars(hparams))

        self.query_encoder = BiEncoderModel.from_pretrained(self.hparams.base_pretrained_model, hparams=self.hparams)
        self.passage_encoder = BiEncoderModel.from_pretrained(self.hparams.base_pretrained_model, hparams=self.hparams)


    @classmethod
    def _init_worker(cls, worker_id, hparams) -> None:
        cls.tokenizer = BertTokenizerFast.from_pretrained(hparams.base_pretrained_model)


    def forward(
        self, query_input: Dict[str, torch.LongTensor], passage_input: Dict[str, torch.LongTensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_repr = self.query_encoder(**query_input)
        passage_repr = self.passage_encoder(**passage_input)
        return query_repr, passage_repr


    def convert_to_binary_code(self, input_repr: torch.Tensor) -> torch.Tensor:
        if self.training:
            if self.hparams.use_ste:
                hard_input_repr = input_repr.new_ones(input_repr.size()).masked_fill_(input_repr < 0, -1.0)
                input_repr = torch.tanh(input_repr)
                return hard_input_repr + input_repr - input_repr.detach()
            else:
                # https://github.com/thuml/HashNet/blob/55bcaaa0bbaf0c404ca7a071b47d6287dc95e81d/pytorch/src/network.py#L40
                scale = math.pow((1.0 + self.global_step * self.hparams.hashnet_gamma), 0.5)
                return torch.tanh(input_repr * scale)
        else:
            return input_repr.new_ones(input_repr.size()).masked_fill_(input_repr < 0, -1.0)
