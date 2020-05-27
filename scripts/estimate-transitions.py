import os
import json
import argparse
from collections import defaultdict
import numpy as np
from baseline.utils import read_conll
from .utils import download_dataset, estimate_counts, normalize_transitions


def make_vocab(transitions):
    vocab = defaultdict(lambda: len(vocab))
    for src in transitions:
        for tgt in transitions[src]:
            vocab[tgt]
        vocab[src]
    return dict(vocab.items())


def to_dense(transitions):
    vocab = make_vocab(transitions)
    trans = np.zeros((len(vocab), len(vocab)))
    for src, src_idx in vocab.items():
        for tgt, tgt_idx in vocab.items():
            trans[src_idx, tgt_idx] = transitions[src][tgt]
    return vocab, trans


def main():
    parser = argparse.ArgumentParser(description="Estimate transition probabilities from the training data.")
    parser.add_argument("dataset")
    parser.add_argument("--datasets-index", "--datasets_index", default="configs/datasets.json")
    parser.add_argument("--cache", default="data")
    parser.add_argument("--surface-index", "--surface_index", default=0, type=int)
    parser.add_argument("--entity-index", "--entity_index", default=-1, type=int)
    parser.add_argument("--delim")
    parser.add_argument("--output")
    args = parser.parse_args()

    dataset = download_dataset(args.dataset, args.datasets_index, args.cache)

    _, transitions = estimate_counts(
        list(read_conll(dataset["train_file"], args.delim)), args.surface_index, args.entity_index
    )

    transitions, _ = normalize_transitions(transitions)

    vocab, transitions = to_dense(transitions)

    args.output = args.dataset if args.output is None else args.output

    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, "vocab.json"), "w") as wf:
        json.dump(vocab, wf, indent=2)
    np.save(os.path.join(args.output, "transitions.npy"), transitions)


if __name__ == "__main__":
    main()
