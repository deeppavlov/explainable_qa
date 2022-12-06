from transformers import BartTokenizer
from JointGT.data import WebNLGDataset 

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

from logging import getLogger
import torch
log = getLogger(__name__)

@register("triplets_tokenizer")
class TripletsTokenizer(Component):
    
    def __init__(self, tokenizer_path, **kwargs):
        tokenizer_path = str(expand_path(tokenizer_path))
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
        self.webnlg_dataset = WebNLGDataset(tokenizer=self.tokenizer)

    def __call__(self, batch, *args, **kwargs):
        BATCH_LEN = 9
        result_batch = [[] for _ in range(BATCH_LEN)]
        for triplet_set in batch:
            for i, comp in enumerate(self.webnlg_dataset.transform(triplet_set)):
                result_batch[i].append(comp) 
        
        return result_batch

