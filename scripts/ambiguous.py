import argparse
from baseline.utils import read_conll
from iobes import extract_type
from .utils import download_dataset, estimate_counts


def main():
    parser = argparse.ArgumentParser(description="Calculate the number of entities that have easy starts.")
    parser.add_argument("dataset")
    parser.add_argument("--datasets-index", "--datasets_index", default="configs/datasets.json")
    parser.add_argument("--cache", default="data")
    parser.add_argument("--surface-index", "--surface_index", default=0, type=int)
    parser.add_argument("--entity-index", "--entity_index", default=-1, type=int)
    parser.add_argument("--types", action="store_true")
    parser.add_argument("--delim")
    args = parser.parse_args()

    dataset = download_dataset(args.dataset, args.datasets_index, args.cache)

    emissions, _ = estimate_counts(
        list(read_conll(dataset["train_file"], args.delim)), args.surface_index, args.entity_index
    )

    ambiguous = 0
    for token, emission in emissions.items():
        if args.types:
            if len(set(extract_type(t) for t in emission)) > 1:
                ambiguous += 1
        else:
            if len(emission) > 1:
                ambiguous += 1

    print(f"{ambiguous / len(emissions) * 100}% of types are ambiguous.")


if __name__ == "__main__":
    main()
