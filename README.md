# Active Reasoning Vision-Language Models via Sequential Experimental Design

## Abstract

Visual perception in modern Vision-Language Models (VLMs) is constrained by a fundamental **perceptual bandwidth bottleneck**: a broad field-of-view inevitably sacrifices the fine-grained details necessary for complex reasoning.

We frame overcoming this limitation as a sequential decision-making process and formalise it through the lens of **Sequential Bayesian Optimal Experimental Design (S-BOED)**. While exact Bayesian inference is intractable in continuous gigapixel spaces, we derive principled yet tractable approximations that balance spatial coverage against resolution.

To validate this framework, we present a **training-free inference strategy** as a practical instantiation of the S-BOED objective for agents equipped with multiple vision tools. The strategy is designed as a flexible template that accommodates arbitrary optimisation algorithms — from efficient greedy sampling to look-ahead planning. Empirical evaluations on gigapixel-level benchmarks show that our approach significantly outperforms standard baselines and effectively narrows the gap towards human-annotated oracles.

## Citation

```bibtex
@inproceedings{liu2026activevlm,
  title     = {Active Reasoning Vision-Language Models via Sequential Experimental Design},
  author    = {Liu, Anjie and Gong, Ziqin and Song, Yan and Chen, Yuxiang and Liu, Xiaolong and Lu, Hengtong and Zhang, Kaike and Wei, Chen},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
