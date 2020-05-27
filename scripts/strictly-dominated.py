import argparse
from baseline.utils import read_conll
from iobes import SpanEncoding, TokenFunction
from .utils import download_dataset, estimate_counts, make_transition_mask


def dominated(possible_tags, previous_tag, transition_mask):
    new_tags = []
    for tag in possible_tags:
        if transition_mask[(previous_tag, tag)]:
            new_tags.append(tag)
    return len(new_tags) <= 1


def strictly_dominated(file_name, emissions, transition_mask, surface_idx, entity_idx, delim):
    domd = 0
    ambig_domd = 0
    ambig_total = 0
    total = 0
    for sentence in read_conll(file_name, delim):
        cols = list(zip(*sentence))
        tags = cols[entity_idx]
        surfaces = cols[surface_idx]
        for i in range(len(surfaces)):
            if i == 0:
                prev = TokenFunction.GO
            else:
                prev = tags[i - 1]
            possible_tags = list(emissions[surfaces[i]])
            if dominated(possible_tags, prev, transition_mask):
                domd += 1
                if emissions[surfaces[i]] and len(emissions[surfaces[i]]) > 1:
                    ambig_domd += 1
            total += 1
            if emissions[surfaces[i]] and len(emissions[surfaces[i]]) > 1:
                ambig_total += 1
    return domd, ambig_domd, total, ambig_total


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
    domd, ambig_domd, total, ambig_total = strictly_dominated(
        dataset["valid_file"], emissions, transition_mask, args.surface_index, args.entity_index, args.delim
    )

    print(f"There are {domd} tokens that are strictly dominated.")
    print(f"There are {ambig_domd} ambiguous tokens that are strictly dominated.")
    print(f"There are {total} tokens in the whole dataset.")

    print(f"{domd / total * 100}% of tokens have are strictly dominated")
    print(f"{ambig_domd / ambig_total * 100}% of ambiguous tokens are strictly dominated")


if __name__ == "__main__":
    main()
