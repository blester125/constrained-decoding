import torch
import numpy as np
from eight_mile.pytorch.layers import TaggerGreedyDecoder
from baseline.utils import read_json
from baseline.model import register_model
from baseline.pytorch.tagger.model import RNNTaggerModel


class PremadeTransitionsTagger(TaggerGreedyDecoder):
    def __init__(self, *args, **kwargs):
        transitions = kwargs.pop("transitions")
        super().__init__(*args, **kwargs)
        self.register_buffer("transitions_p", torch.from_numpy(transitions).unsqueeze(0))

    @property
    def transitions(self):
        return self.transitions_p.masked_fill(self.constraint_mask, -1e4)


@register_model(task="tagger", name="premade")
class RemadeTaggerModle(RNNTaggerModel):
    def init_decode(self, **kwargs):

        label_vocab = read_json(kwargs["label_vocab"])
        label_trans = np.load(kwargs["label_trans"])

        trans = np.zeros((len(self.labels), len(self.labels)))

        for src, src_idx in self.labels.items():
            if src not in label_vocab:
                continue
            for tgt, tgt_idx in self.labels.items():
                if tgt not in label_vocab:
                    continue
                trans[src_idx, tgt_idx] = label_trans[label_vocab[src], label_vocab[tgt]]

        name = kwargs.get("decode_name")
        self.constraint_mask = kwargs.get("constraint_mask").unsqueeze(0)
        return PremadeTransitionsTagger(len(self.labels), self.constraint_mask, transitions=trans)
