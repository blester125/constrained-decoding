import argparse
from baseline.utils import read_conll
from iobes import parse_spans, SpanEncoding
from .utils import download_dataset, estimate_counts


def easy_end(file_name, emissions, surface_idx, entity_idx, span_type, delim, types=False):
    easy = []
    total = []
    type_surface = set()
    for sentence in read_conll(file_name, delim):
        cols = list(zip(*sentence))
        tags = cols[entity_idx]
        surfaces = cols[surface_idx]
        for entity in parse_spans(tags, span_type):
            end_token = surfaces[entity.end - 1]
            if (end_token, entity.type) in type_surface and types:
                continue
            type_surface.add((end_token, entity.type))
            # If we had never seen this token before we won't have any reason to think it is
            # an `I-` so it should be an easier E- to get so we don't have to check for if
            # the emissions exist
            if f"I-{entity.type}" not in emissions[end_token]:
                easy.append(entity)
            total.append(entity)
    return easy, total


def main():
    parser = argparse.ArgumentParser(description="Calculate the number of entities that have easy starts.")
    parser.add_argument("dataset")
    parser.add_argument("--datasets-index", "--datasets_index", default="configs/datasets.json")
    parser.add_argument("--cache", default="data")
    parser.add_argument("--surface-index", "--surface_index", default=0, type=int)
    parser.add_argument("--entity-index", "--entity_index", default=-1, type=int)
    parser.add_argument("--span-type", "--span_type", default=SpanEncoding.IOBES, type=SpanEncoding.from_string, choices=("iobes", "bio", "iob"))
    parser.add_argument("--types", action="store_true")
    parser.add_argument("--delim")
    args = parser.parse_args()

    dataset = download_dataset(args.dataset, args.datasets_index, args.cache)

    emissions, _ = estimate_counts(list(read_conll(dataset['train_file'], args.delim)), args.surface_index, args.entity_index)

    easy, total = easy_end(dataset['valid_file'], emissions, args.surface_index, args.entity_index, args.span_type, args.delim, args.types)

    print(f"There are {len(easy)} entities that end with an unambiguous tokens.")
    print(f"There are {len(total)} entities in the whole dataset.")

    print(f"{len(easy) / len(total) * 100}% of entities have easy ends.")


if __name__ == "__main__":
    main()
