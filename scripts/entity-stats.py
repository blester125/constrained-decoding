import argparse
import pandas as pd
import matplotlib.pyplot as plt
from baseline import read_conll, to_chunks
from .utils import download_dataset


def read_entities(file_name, entity_index, span_type, span_delim, delim):
    types = []
    for sentence in read_conll(file_name, delim):
        cols = list(zip(*sentence))
        tags = cols[entity_index]
        entities = to_chunks(tags, span_type, span_delim)
        for entity in entities:
            type_name, *locs = entity.split(span_delim)
            types.append({"type": type_name, "length": len(locs)})
    return types


def read(dataset, entity_index, span_type, span_delim, delim):
    all_types = []
    for data_type, file_name in dataset.items():
        types = read_entities(file_name, entity_index, span_type, span_delim, delim)
        types = pd.DataFrame(types)
        types["dataset"] = data_type
        all_types.append(types)
    return pd.concat(all_types, ignore_index=False)


def mode(x):
    return x.value_counts().index[0]


def main():
    parser = argparse.ArgumentParser(description="Calculate the lengths of the entities")
    parser.add_argument("dataset")
    parser.add_argument("--datasets-index", "--datasets_index", default="configs/datasets.json")
    parser.add_argument("--cache", default="data")
    parser.add_argument("--entity-index", "--entity_index", default=-1, type=int)
    parser.add_argument("--span-type", "--span_type", default="iobes", choices=("iobes", "bio", "iob"))
    parser.add_argument("--span-delim", "--span_delim", default="@")
    parser.add_argument("--delim")
    args = parser.parse_args()

    dataset = download_dataset(args.dataset, args.datasets_index, args.cache)

    data = read(dataset, args.entity_index, args.span_type, args.span_delim, args.delim)

    print(f"Total number of entities: {len(data)}")
    print()
    print("Entity counts per type:")
    print(data.groupby(["type"]).agg(["count"])["length"])
    print()
    print("Entity counts per dataset:")
    print(data.groupby(["dataset"]).agg(["count"])["length"])
    print()
    print("Entity counts per type per dataset:")
    print(data.groupby(["dataset", "type"]).agg(["count"])["length"])
    print()
    print("Entity lengths stats:")
    print(data[["length"]].agg(["mean", "std", "min", "max", mode]).T)

    print("Entity lengths by type:")
    print(data[["type", "length"]].groupby(["type"]).agg(["mean", "std", "min", "max", mode]))
    print()
    print("Entity lengths by dataset:")
    print(data[["dataset", "length"]].groupby(["dataset"]).agg(["mean", "std", "min", "max", mode]))
    print()
    print("entity length by dataset by type:")
    print(data.groupby(["dataset", "type"]).agg(["mean", "std", "min", "max", mode]))


if __name__ == "__main__":
    main()
