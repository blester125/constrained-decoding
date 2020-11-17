# Constrained Decoding

**In Findings of EMNLP 2020**

Code used for experiments from [Constrained Decoding for Computationally Efficient Named Entity Recognition Taggers](https://arxiv.org/abs/2010.04362)

This contains [Mead-Baseline](https://github.com/dpressel/mead-baseline) configs for models used as well as scripts used to analyze datasets in an effort to describe *why* constrained decoding was so effective for all datasets but Ontonotes.

Scripts should be run from the top level of this repo with `python -m scripts.{script_name}`

Train models with `mead-train --config configs/conll.json`

# Citation

If you use the constrained decoding in Mead-Baseline to replace a CRF, or use the dataset analysis metrics we describe please cite the following (will be updated to the acl anthology bibtex once that is released):

```BibTex
@inproceedings{lester-etal-2020-constrained,
    title = "Constrained Decoding for Computationally Efficient Named Entity Recognition Taggers",
    author = "Lester, Brian  and
      Pressel, Daniel  and
      Hemmeter, Amy  and
      Ray Choudhury, Sagnik  and
      Bangalore, Srinivas",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.166",
    pages = "1841--1848",
    abstract = "Current state-of-the-art models for named entity recognition (NER) are neural models with a conditional random field (CRF) as the final layer. Entities are represented as per-token labels with a special structure in order to decode them into spans. Current work eschews prior knowledge of how the span encoding scheme works and relies on the CRF learning which transitions are illegal and which are not to facilitate global coherence. We find that by constraining the output to suppress illegal transitions we can train a tagger with a cross-entropy loss twice as fast as a CRF with differences in F1 that are statistically insignificant, effectively eliminating the need for a CRF. We analyze the dynamics of tag co-occurrence to explain when these constraints are most effective and provide open source implementations of our tagger in both PyTorch and TensorFlow.",
}
```
