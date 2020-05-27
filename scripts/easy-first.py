import argparse
from baseline.utils import read_conll
from iobes import parse_spans, SpanEncoding, TokenFunction
from .utils import download_dataset, estimate_counts, make_transition_mask


def easy_start(file_name, emissions, transition_mask, surface_idx, entity_idx, span_type, delim, types=False):
    unamb = []
    domd = []
    total = []
    type_surface = set()
    for sentence in read_conll(file_name, delim):
        cols = list(zip(*sentence))
        tags = cols[entity_idx]
        surfaces = cols[surface_idx]
        for entity in parse_spans(tags, span_type):
            start_token = surfaces[entity.start]
            if (start_token, entity.type) in type_surface and types:
                continue
            type_surface.add((start_token, entity.type))
            if not emissions[start_token]:
                continue
            if len(emissions[start_token]) == 1:
                unamb.append(entity)
            else:
                prev = tags[entity.start - 1] if entity.start > 0 else TokenFunction.GO
                all_tags = list(emissions[start_token].keys())
                possible_tags = []
                for tag in all_tags:
                    if transition_mask[(prev, tag)]:
                        possible_tags.append(tag)
                if len(possible_tags) == 1:
                    domd.append(entity)
            total.append(entity)
    return unamb, domd, total


def main():
    parser = argparse.ArgumentParser(description="Calculate the number of entities that have easy starts.")
    parser.add_argument("dataset")
    parser.add_argument("--datasets-index", "--datasets_index", default="configs/datasets.json")
    parser.add_argument("--cache", default="data")
    parser.add_argument("--surface-index", "--surface_index", default=0, type=int)
    parser.add_argument("--entity-index", "--entity_index", default=-1, type=int)
    parser.add_argument(
        "--span-type", "--span_type", default=SpanEncoding.IOBES, type=SpanEncoding.from_string,
    )
    parser.add_argument("--types", action="store_true")
    parser.add_argument("--delim")
    args = parser.parse_args()

    dataset = download_dataset(args.dataset, args.datasets_index, args.cache)

    emissions, _ = estimate_counts(
        list(read_conll(dataset["train_file"], args.delim)), args.surface_index, args.entity_index
    )

    transition_mask = make_transition_mask(dataset, args.span_type, args.entity_index, args.delim)
    unamb, domd, total = easy_start(
        dataset["valid_file"],
        emissions,
        transition_mask,
        args.surface_index,
        args.entity_index,
        args.span_type,
        args.delim,
        args.types,
    )

    print(f"There are {len(unamb)} entities that start with an unambiguous tokens.")
    print(f"There are {len(domd)} entities that have ambiguous starts but transitions dominate them.")
    print(f"There are {len(total)} entities in the whole dataset.")

    print(f"{len(unamb) / len(total) * 100}% of entities have unambiguous starts")
    print(f"{len(domd) / len(total) * 100}% of entities have dominated starts")
    print(f"{(len(unamb) + len(domd)) / len(total) * 100}% of entities have easy starts")


if __name__ == "__main__":
    main()
