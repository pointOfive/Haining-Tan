# Bayesian Analysis of Normalizing Flow Models using Importance Sampling

#### A U of T Readings Course organized by Scott Schwartz and Haining Tan

Normalizing Flows are a widely used methodology which can approximate an arbitrarily complex data distribution by applying a series of invertible smooth (change of variables) transformations to a simple density. This Readings Course will explore the use of Importance Sampling to perform a Bayesian posterior analysis of a Normalizing Flow as a likelihood model using a suitable prior distribution as the importance sampling proposal function. By employing a Bayesian posterior analysis we intend to capture aleatoric uncertainty inherent in model fitting, and by using a Normalizing Flow as a likelihood we intend to capture epistemic uncertainty inherent in data. The work will be done in python using neural networks to learn the approximated function, and will address open questions implicit in this outline. E.g., how can one construct "a suitable prior" for this neural network methodology?

## Readings and Questions

0. [Chapter 6.4.1 Importance Sampling (Givens/Hoeting)](https://librarysearch.library.utoronto.ca/permalink/01UTORONTO_INST/14bjeso/alma991106781097906196)
   - [Questions](BayesImportanceSampling.ipynb) and [Answers](Importance_Sampling.pdf)
1. [MADE: Masked Autoencoder for Distribution Estimation](https://arxiv.org/abs/1502.03509)
   
   Questions:
   
   - How can neural networks be used to represent a probability density in the manner of MADE?
   - (Re)Implement the ideas of [this internet blog post](https://blog.tensorflow.org/2019/03/regression-with-probabilistic-layers-in.html) 

   [Answers](MADE.ipynb) and [Comments](MADE_comments.ipynb)
   
2. Variational Inference -- academic citation(s?) needed -- please find!

   - Do my [Comments](MADE_comments.ipynb) from the previous topic make sense?
      - Is there something that doesn't make sense to you?
   - Please find an academic manuscript (or serveral) on Variational Inference which you think is a good reference for Variational Inference and link them here.

3. [(MAF) Masked Autoregressive Flow for Density Estimation](https://arxiv.org/abs/1705.07057)

   - What distinguishes MAF methodology from MADE methodology?
   - (Re)Implement the ideas of [this documentation page](https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/AutoregressiveNetwork)
     
     - In the second block of code on this documentation page there is the comment
       
        `# Density estimation with MADE.`
       
        however, is the methodology coded up here MADE or MAF?  I.e., how do you rationalize what is meant by this code comment?

4. [(IAF) Improving Variational Inference with Inverse Autoregressive Flow](https://arxiv.org/abs/1606.04934)

   - Why does IAF methodology exist when MAF methodology is already available?
     - (Re)Implement the ideas of [this documentation page](https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/MaskedAutoregressiveFlow)

5. I (personally) found explanatory videos on YouTube and "internet blog articles" a good *first* way to understand these methods.
   - Please identify and use as many such additional resources as needed to faciliate your efficient understanding of the methodologies here (and of course include these resources as referenced materials).
