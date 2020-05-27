import argparse
from itertools import chain
from iobes import SpanEncoding
from .utils import download_dataset, read_entities


def main():
    parser = argparse.ArgumentParser(description="Calculate the number of distinct types of entities")
    parser.add_argument("dataset")
    parser.add_argument("--datasets-index", "--datasets_index", default="configs/datasets.json")
    parser.add_argument("--cache", default="data")
    parser.add_argument("--entity-index", "--entity_index", default=-1, type=int)
    parser.add_argument(
        "--span-type", "--span_type", default=SpanEncoding.IOBES, type=SpanEncoding.from_string,
    )
    parser.add_argument("--delim")
    args = parser.parse_args()

    dataset = download_dataset(args.dataset, args.datasets_index, args.cache)

    train_entities = read_entities(dataset["train_file"], args.entity_index, args.span_type, args.delim)
    dev_entities = read_entities(dataset["valid_file"], args.entity_index, args.span_type, args.delim)
    test_entities = read_entities(dataset["test_file"], args.entity_index, args.span_type, args.delim)

    types = set()
    types = set(s.type for s in chain(train_entities, dev_entities, test_entities))

    print(f"There are {len(types)} in {args.dataset}")
    for t in types:
        print(t)


if __name__ == "__main__":
    main()
