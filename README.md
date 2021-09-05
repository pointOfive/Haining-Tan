# Haining-Tan
A U of T Readings Course

Hello Haining!

We'll make a landing page that makes more sense as we progress, but for now, 
here's your [first question](BayesImportanceSampling.ipynb)!

## Readings and Questions

0. [Chapter 6.4.1 Importance Sampling (Givens/Hoeting)](https://librarysearch.library.utoronto.ca/permalink/01UTORONTO_INST/14bjeso/alma991106781097906196)
   - [Questions](BayesImportanceSampling.ipynb) and [Answers](Importance_Sampling.pdf)
1. [MADE: Masked Autoencoder for Distribution Estimation](https://arxiv.org/abs/1502.03509)
2. [(MAF) Masked Autoregressive Flow for Density Estimation](https://arxiv.org/abs/1705.07057)
3. [(IAF) Improving Variational Inference with Inverse Autoregressive Flow](https://arxiv.org/abs/1606.04934)
4. I (personally) found explanatory videos on YouTube and "internet blog articles" a good *first* way to understand these methods.
   - Please identify and use as many such additional resources as needed to faciliate your efficient understanding of the methodologies here (and of course include these resources as referenced materials).

   Questions (all coding may be done in pytorch if you wish):
   - How can neural networks be used to represent a probability density in the manner of MADE?
     - (Re)Implement the ideas of [this internet blog post](https://blog.tensorflow.org/2019/03/regression-with-probabilistic-layers-in.html) 
   - What distinguishes MAF methodology from MADE methodology?
     - (Re)Implement the ideas of [this documentation page](https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/AutoregressiveNetwork)
   - Why does IAF methodology exist when MAF methodology is already available?
     - (Re)Implement the ideas of [this documentation page](https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/MaskedAutoregressiveFlow)
   - What distinguishes Normalizing Flows from GANs?
