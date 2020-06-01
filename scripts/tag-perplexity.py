import argparse
from itertools import chain
from typing import Optional, List
from iobes import TokenFunction
from baseline.utils import read_conll
from .utils import download_dataset
from .lm import KneserNeyLM, LaplaceLM, LaplaceUnigramLM, ReducedLaplaceLM, ReducedKneserNeyLM


def extract_tags(file_name: str, index: int, delim: Optional[str] = None) -> List[List[str]]:
    tags = []
    for sentence in read_conll(file_name, delim):
        tags.append(list(chain([TokenFunction.GO], list(zip(*sentence))[index], [TokenFunction.EOS])))
    return tags


def main():
    parser = argparse.ArgumentParser(description="Calculate the number of entities that have easy starts.")
    parser.add_argument("dataset")
    parser.add_argument("--datasets-index", "--datasets_index", default="configs/datasets.json")
    parser.add_argument("--cache", default="data")
    parser.add_argument("--tag-index", "--tag_index", default=-1, type=int)
    parser.add_argument("--delim")
    parser.add_argument(
        "--lm-type",
        "--lm_type",
        default="kneser-ney",
        choices=("kneser-ney", "laplace", "unigram", "reduced-kneser-ney", "reduced-laplace"),
    )
    parser.add_argument("--discount", type=float, default=0.75)
    args = parser.parse_args()

    dataset = download_dataset(args.dataset, args.datasets_index, args.cache)

    training = extract_tags(dataset["train_file"], args.tag_index, args.delim)
    testing = extract_tags(dataset["test_file"], args.tag_index, args.delim)

    if args.lm_type == "kneser-ney":
        lm = KneserNeyLM(args.discount)
    elif args.lm_type == "laplace":
        lm = LaplaceLM()
    elif args.lm_type == "reduced-kneser-ney":
        lm = ReducedKneserNeyLM(args.discount)
    elif args.lm_type == "reduced-laplace":
        lm = ReducedLaplaceLM()
    else:
        lm = LaplaceUnigramLM()

    lm.train(training)

    print(f"Perplexity: {lm.perplexity(testing)}")


if __name__ == "__main__":
    main()
