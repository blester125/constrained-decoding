import argparse
from itertools import chain
from typing import List, Tuple, Any
from baseline import read_conll
from iobes import extract_function, TokenFunction
from .utils import download_dataset


def bigrams(seq: List[Any]) -> List[Tuple[Any, Any]]:
    return list(zip(seq, seq[1:]))


def real_transitions(file_name, entity_index, delim):
    real = 0
    total = 0
    for sentence in read_conll(file_name, delim):
        tags = list(zip(*sentence))[entity_index]
        for prev, curr in bigrams(list(chain([TokenFunction.GO], tags, [TokenFunction.EOS]))):
            prev_func = extract_function(prev)
            curr_func = extract_function(curr)
            if prev_func not in ("B", "I"):
                if curr_func not in ("E", "I"):
                    real += 1
            total += 1
    return real, total


def main():
    parser = argparse.ArgumentParser(description="Calculate the number of transitions that aren't covered by the mask.")
    parser.add_argument("dataset")
    parser.add_argument("--datasets-index", "--datasets_index", default="configs/datasets.json")
    parser.add_argument("--cache", default="data")
    parser.add_argument("--entity-index", "--entity_index", default=-1, type=int)
    parser.add_argument("--delim")
    args = parser.parse_args()

    dataset = download_dataset(args.dataset, args.datasets_index, args.cache)

    real, total = real_transitions(dataset['train_file'], args.entity_index, args.delim)

    print(f"{real / total * 100}% of transitions are real.")

if __name__ == "__main__":
    main()
