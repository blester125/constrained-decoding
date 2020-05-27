import numpy as np
import tensorflow as tf
from eight_mile.tf.layers import TaggerGreedyDecoder
from baseline.utils import read_json, revlut
from baseline.model import register_model
from baseline.tf.tagger.model import RNNTaggerModel


class PremadeTransitionsTagger(TaggerGreedyDecoder):
    def __init__(self, *args, **kwargs):
        self.trans = kwargs.pop("transitions")
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.A = self.add_weight(
            "transitions",
            shape=(self.num_tags, self.num_tags),
            dtype=tf.float32,
            initializer=tf.constant_initializer(self.trans),
            trainable=False,
        )
        if self.inv_mask is not None:
            self.inv_mask = self.add_weight(
                "inverse_constraint_mask",
                shape=(self.num_tags, self.num_tags),
                dtype=tf.float32,
                initializer=tf.constant_initializer(self.inv_mask),
                trainable=False,
            )


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
        self.constraint_mask = kwargs.get("constraint_mask")
        return PremadeTransitionsTagger(len(self.labels), self.constraint_mask, name=name, transitions=trans)
