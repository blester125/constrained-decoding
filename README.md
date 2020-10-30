# Constrained Decoding

**To Appear in Findings of EMNLP 2020**

Code used for experiments from [Constrained Decoding for Computationally Efficient Named Entity Recognition Taggers](https://arxiv.org/abs/2010.04362)

This contains [Mead-Baseline](https://github.com/dpressel/mead-baseline) configs for models used as well as scripts used to analyze datasets in an effort to describe *why* constrained decoding was so effective for all datasets but Ontonotes.

Scripts should be run from the top level of this repo with `python -m scripts.{script_name}`

Train models with `mead-train --config configs/conll.json`

# Citation

If you use the constrained decoding in Mead-Baseline to replace a CRF, or use the dataset analysis metrics we describe please cite the following (will be updated to the acl anthology bibtex once that is released):

```BibTex
@article{lester2020constrained,
  title={Constrained Decoding for Computationally Efficient Named Entity Recognition Taggers},
  author={Lester, Brian and Pressel, Daniel and Hemmeter, Amy and Choudhury, Sagnik Ray and Bangalore, Srinivas},
  journal={arXiv preprint arXiv:2010.04362},
  year={2020}
}
```
