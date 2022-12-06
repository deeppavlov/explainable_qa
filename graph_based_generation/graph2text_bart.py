from typing import List
import torch
from transformers import BartTokenizer
from JointGT.modeling_bart import MyBartForConditionalGeneration as MyBart

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from utils import sanitize


@register('graph2text_bart')
class GraphToTextBart(Component):
    
    def __init__(self, checkpoint_path, tokenizer_path, **kwargs):
        checkpoint_path = str(expand_path(checkpoint_path))
        tokenizer_path = str(expand_path(tokenizer_path))
        self.model = MyBart.from_pretrained(checkpoint_path)
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
    
    def __call__(self, batch):
        for i in range(len(batch)):
            batch[i] = torch.stack(batch[i]).to(self.device)


        outputs = self.model.generate(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 input_node_ids=batch[4],
                                 input_edge_ids=batch[5],
                                 node_length=batch[6],
                                 edge_length=batch[7],
                                 adj_matrix=batch[8],
                                 num_beams=5,
                                 length_penalty=1,
                                 max_length=128,
                                 early_stopping=True,)
        predictions = []
        for input_, output in zip(batch[0], outputs):
            pred = self.tokenizer.decode(output, skip_special_tokens=True)
            predictions.append(sanitize(pred.strip()))

        return predictions
