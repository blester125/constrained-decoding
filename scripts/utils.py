import functools
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, Counter
from baseline.utils import DataDownloader, read_conll
from mead.utils import index_by_label, read_config_file_or_json
from iobes import parse_spans, SpanEncoding, Span, TokenFunction, transitions_legality, transitions_to_tuple_map


ConllRow = List[str]
ConllSentence = List[ConllRow]
ConllCorpus = List[ConllSentence]

Params = functools.partial(defaultdict, Counter)
Emission = Params
Transition = Params


def download_dataset(dataset: str, datasets_index: str, cache: str) -> Dict[str, str]:
    dataset = index_by_label(read_config_file_or_json(datasets_index))[dataset]
    return DataDownloader(dataset, cache).download()


def read_all_tags(dataset, entity_idx, delim):
    tags = set()
    for ds in (dataset["train_file"], dataset["valid_file"], dataset["test_file"]):
        for sent in read_conll(ds, delim):
            tags.update(list(zip(*sent))[entity_idx])
    return tags


def make_transition_mask(dataset, span_type, entity_idx, delim):
    tags = read_all_tags(dataset, entity_idx, delim)
    return transitions_to_tuple_map(transitions_legality(tags, span_type))


def read_entities(
    file_name: str, entity_index: int = -1, span_type: SpanEncoding = SpanEncoding.IOBES, delim: Optional[str] = None
) -> List[Span]:
    entities = []
    for sentence in read_conll(file_name, delim):
        tags = list(zip(*sentence))[entity_index]
        for entity in parse_spans(tags, span_type):
            entities.append(entity)
    return entities


def tag_set(sentences: ConllCorpus, tag_index: int = -1) -> Set[str]:
    tags = set()
    for sentence in sentences:
        cols = list(zip(*sentence))
        tags.update(cols[tag_index])
    return tags


def estimate_counts(sentences: ConllCorpus, surface_index: int = 0, tag_index: int = -1) -> Tuple[Emission, Transition]:
    emiss = Emission()
    trans = Transition()

    for sentence in sentences:
        cols = list(zip(*sentence))
        text = cols[surface_index]
        tags = cols[tag_index]
        prev_tag = TokenFunction.GO
        for token, tag in zip(text, tags):
            emiss[token][tag] += 1
            trans[prev_tag][tag] += 1
            prev_tag = tag
        trans[prev_tag][TokenFunction.EOS] += 1
    return emiss, trans


def normalize_transitions(transitions: Transition) -> Tuple[Transition, Transition]:
    probs = defaultdict(Counter)
    rev_probs = defaultdict(Counter)
    for src, tgts in transitions.items():
        norm = sum(tgts.values())
        for tgt in tgts:
            probs[src][tgt] = tgts[tgt] / norm
            rev_probs[tgt][src] = tgts[tgt] / norm
    return probs, rev_probs
