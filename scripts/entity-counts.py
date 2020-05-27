import argparse
from iobes import SpanEncoding
from .utils import download_dataset, read_entities


def main():
    parser = argparse.ArgumentParser(description="Count the number of entities in a dataset.")
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

    print(f"There are {len(train_entities)} in the training data.")
    print(f"There are {len(dev_entities)} in the development data.")
    print(f"There are {len(test_entities)} in the testing data.")
    print(f"There are {len(train_entities) + len(dev_entities) + len(test_entities)} in the whole dataset.")


if __name__ == "__main__":
    main()
