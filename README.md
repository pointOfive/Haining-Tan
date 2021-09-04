# Haining-Tan
A U of T Readings Course

0. For target distribution $p(\theta|x)$ (a posterior), and proposal distribution $p(\theta)$ (a prior), show that the normalized importance weights for importance sampling are proportional to $f(x|\theta)$ (the likelihood).

  > Hint: Monte Carlo integration of $g(\theta) p(\theta\x)$ can be based on importance sampling with the prior $p(\theta)$ as the proposal distribution since
  > $$\int g(\theta) p(\theta|x) d\theta = \int g(\theta) \frac{p(\theta|x)}{p(\theta)} p(\theta) d\theta $$

  - Why might it be better to use unnormalized importance weights in the above context?

  > Reading: [Chapter 6.4.1 Importance Sampling (Givens/Hoeting)](https://librarysearch.library.utoronto.ca/permalink/01UTORONTO_INST/14bjeso/alma991106781097906196)
