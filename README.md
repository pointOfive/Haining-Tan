# Bayesian Analysis of Normalizing Flow Models using Importance Sampling

#### A U of T Readings Course organized by Haining Tan and Prof. Scott Schwartz

Normalizing Flows are a widely used methodology which can approximate an arbitrarily complex data distribution by applying a series of invertible smooth (change of variables) transformations to a simple density. This Readings Course will explore the use of Importance Sampling to perform a Bayesian posterior analysis of a Normalizing Flow as a likelihood model using a suitable prior distribution as the importance sampling proposal function. By employing a Bayesian posterior analysis we intend to capture aleatoric uncertainty inherent in model fitting, and by using a Normalizing Flow as a likelihood we intend to capture epistemic uncertainty inherent in data. The work will be done in python using neural networks to learn the approximated function, and will address open questions implicit in this outline. E.g., how can one construct "a suitable prior" for this neural network methodology?

## Readings and Questions

0. [Chapter 6.4.1 Importance Sampling (Givens/Hoeting)](https://librarysearch.library.utoronto.ca/permalink/01UTORONTO_INST/14bjeso/alma991106781097906196)
   - [Questions](BayesImportanceSampling.ipynb) and [Answers](Importance_Sampling.pdf)
1. [MADE: Masked Autoencoder for Distribution Estimation](https://arxiv.org/abs/1502.03509)
   
   Questions:
   
   - How can neural networks be used to represent a probability density in the manner of MADE?
      - *Outputs from neural networks can define the parameters of distributions; hence, define distributions*
      - *MADE enforces autoregressive dependency in the outputs relative to the inputs:*
        - output i only depends on inputs 0 through i-1
        - if output i is the distribution for input i but only dependent on inputs 0 through i-1
        - then all outputs autoregressively define a joint distribution via the chain rule
   - (Re)Implement the ideas of [this internet blog post](https://blog.tensorflow.org/2019/03/regression-with-probabilistic-layers-in.html) 

   [Answers](MADE.ipynb) and [Comments](MADE_comments.ipynb)
   
2. [Variational Inference](https://arxiv.org/abs/1601.00670)

   - [Rejoinder](Variational_Inference.ipynb) to [Some Initial Comments](MADE_comments.ipynb)
   
     <details><summary>Some Further Comments</summary>
     <br>
     1. log(p(y)) is the expectation of the log likelihood under the prior so integrating over the (approximate) posterior is silly
     but is workable since we can correct for it with the "triangulation" between the posterior/prior/approximate posterior. 
     <br>
     2. The hyper parameters of `DenseLayer` have improper (unconstrained) hyperpriors, choices of which (including those for \sigma) 
     define the q(theta) approximation of the posterior p(theta|y).
     <br>
     3. For TF, `loss` is -log(p(y|theta) while `losses` is the KL-term; albeit, not very helpful as far as variable naming goes.
     Generally, losses specific to layers are accumulated in `losses` and then added to the `loss` associated with the output.
     <br>
     4. Gradient descent makes its step on each batch, thus, the targeted objective must be correct for each batch.
     </details>

3. [(MAF) Masked Autoregressive Flow for Density Estimation](https://arxiv.org/abs/1705.07057)

   - What distinguishes MAF methodology from MADE methodology?
   - (Re)Implement the ideas of [this documentation page](https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/AutoregressiveNetwork)
     
     - In the second block of code on this documentation page there is the comment
       
        `# Density estimation with MADE.`
       
        however, is the methodology coded up here MADE or MAF?  I.e., how do you rationalize what is meant by this code comment?

4. [(IAF) Improving Variational Inference with Inverse Autoregressive Flow](https://arxiv.org/abs/1606.04934)

   - Why does IAF methodology exist when MAF methodology is already available?
     - (Re)Implement the ideas of [this documentation page](https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/MaskedAutoregressiveFlow)
   - What would you say is the primary methodological capability that distinguishes Normalizing Flows from GANs?

5. The relationship between Normalizing Flows (NFs) and Gaussian Processes (GPs).

   - <**Enumerate your references here**>

   - An intial GP use case is given in the "Bonus: Tabula Rasa" section of the [Regression with Probabilitic Layers](https://blog.tensorflow.org/2019/03/regression-with-probabilistic-layers-in.html) article.
   - An initial GP specification is considered in the [Variational Inference Report](Variational_Inference.ipynb).

     <details><summary>A Bayesian/generative modeling GP specification</summary>
     <br>
     
     <table>
       <tr>
          <td>X ~ p(X)</td>
          <td>perhaps taken to be fixed, or i.i.d. uniforms over some range so \propto 1</td>
        </tr>
        <tr>
          <td>theta ~ p(theta)</td>
          <td>where theta parameterizes a covariance function K</td>
        </tr>
        <tr>
          <td>mu ~ p(mu)</td>
          <td>assumes an independent prior for possible non-0-centering</td>
        </tr>
        <tr>
          <td>f_x ~ GP = p(f_x|X,mu,theta)</td>
          <td>which for X=Reals samples continuous functions</td>
        </tr>
        <tr>
          <td>f_x ~ MNV(mu, COV=K(X, theta))</td>
          <td></td>
        </tr>
        <tr>
          <td>sigma^2 ~ p(sigma^2)</td>
          <td>assumes an independent prior for homoskedastic variance</td>
        </tr>
        <tr>
          <td>Y ~ MVN(E[Y] = f_x, sigma^2 I)</td>
          <td></td>
        </tr>
      </table>
      so
      <br>
      p(f|-) \propto MVN(E[Y] = S, sigma^2 I) MNV(E[S] = mu, COV(S) = K(X, theta)) p(theta, mu, sigma^2)
      <br>
      which for conjugate priors would allow a MNV posterior for p(S|-). 
     </details>

*I (personally) find explanatory videos on YouTube and "internet blog articles" a good *first* way to understand these methods. Please identify and use as many such additional resources as needed to faciliate your efficient understanding of the methodologies here (and of course include these resources as referenced materials).*

6. Some ideas for easy implementation of Bayesian approximation

   - [Dropout as Bayesian Approximation](https://arxiv.org/abs/1506.02142)
      - [Original Dropout Paper](https://jmlr.org/papers/v15/srivastava14a.html)
      - [In TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout)
      - [Dropout is not Bayesian Approximation](https://arxiv.org/abs/1711.02989)
   - [Batch Normalization as Bayesian Approximation](https://arxiv.org/abs/1802.06455)
      - [Original Batch Normalization Paper](https://arxiv.org/abs/1502.03167)
      - [In TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization)
      - [Batch Normalization is not Bayesian Approximation](https://openreview.net/forum?id=BJlrSmbAZ)

   Questions:
    
   1. Provide nice demonstrations of Dropout and Batch Normalization layers being useful in Neural Networks.
   2. What conclusions do you draw from the academic discourse regarding these two ideas?

7. Other Methodologies for Characterizing Uncertainty Estimation in Neural Networks
   - Begin by finding some online blogs enumerating some approaches
     - <**Enumerate your references here**>
   - Collect and summarize the manuscripts of some of the methodologies
     - <**Enumerate your references here**>

8. Empirical Bayes

   - <**Enumerate your references here**>

   Questions:
   
   1. What is empirical Bayes?
   2. Provide a nice, simple example illustrating the concept and use of empirical Bayes in practice.
