
These are simple derivations of the famous forward-backward algorithm by [Baum et. al. (1970)](https://projecteuclid.org/euclid.aoms/1177697196) for computing posteriors of hidden states in HMMs.  

# Hidden Markov model

### Specification
Let there be a sequence $\{x_1,\dots,x_n\}$, where each $x_t$ denotes the system as being in a hidden state at time $t$.  The sequence is a discrete time Markov chain, so

$$p(x_t|x_1,\dots,x_{t-1}) = p(x_t|x_{t-1})$$

There are $m$ hidden states, and $p(x_t = j|x_{t-1} = i)$ is the probability of transition from state $i$ to state $j$.

While the states are hidden, we observe a sequence of output values $\{y_1,\dots,y_n\}$, where each $y_t$ is drawn from a distribution $p(y_t|x_t)$ that depends on the current hidden state $x_t$.  Note that $y_t$ can be discrete or continuous.

![HMM](https://upload.wikimedia.org/wikipedia/commons/8/83/Hmm_temporal_bayesian_net.svg)

*(Image taken from Wikipedia)*

For simplicity, we'll assume these probabilities/distributions are the same across time. 

<!-- ### Example -->
<!-- Let $x_t \in \{\text{Sick}, \text{Healthy}\}$, and let $y_t$ be counts of the number of sneezes on day $t$.  -->

<!-- Suppose you recorded how many times you sneezed every day for a year (weirdo).  Can you tell which days you were sick from this data? -->

### Goal
We want to compute the posterior probabilities over possible hidden states $p(x_t|y_1,\dots,y_n)$ at all time points $t = 1,\dots,n$.

# Forward-backward algorithm

### Given

Assume we know for $t = 1,\dots,n$:

- All transition probabilities $p(x_t|x_{t-1})$ 
- All output probabilities $p(y_t|x_t)$ 

and also the distribution for the initial state $p(x_1)$.

<!-- Think of these as the HMM model parameters (which we'll need to tune later).  -->



### Motivation

From the image above, we see that emissions are conditionally independent of past emissions given the current hidden state.  Hence, we can factor our target posterior:
$$ \begin{align} p(x_t|y_1,\dots,y_n) &\propto p(x_t, y_1, \dots, y_n) \\ &=  p(y_{t+1},\dots,y_n|x_t, y_1,\dots,y_t) p(x_t, y_1,\dots,y_t) \\ &= \underbrace{p(y_{t+1},\dots,y_n|x_t)}_{\text{backward}} \underbrace{p(x_t, y_1,\dots,y_t)}_{\text{forward}}  \end{align}$$

### Algorithm
For each $t = 1,\dots,n$:

1. Use the forward algorithm to compute $p(x_1,y_1,\dots,y_t)$

2. Use the backward algorithm to compute $p(y_{t+1},\dots,y_n|x_t)$

3. Multiply the outcomes together to get $p(x_t|y_1,\dots,y_n)$

# Forward algorithm

### Motivation
Suppose we're interested in the distribution of the observed output sequence $p(y_1\dots,y_n)$. 

A brute force method:
$$\begin{align} p(y_1,\dots,y_n) &= \sum_{\{x_1,\dots,x_n\} } p(x_1,\dots,x_n) p(y_1,\dots,y_n|x_1,\dots,x_n)  \\
&= \sum_{\{x_1,\dots,x_n\} } p(x_1) \prod_{t=2}^n p(x_t|x_{t-1}) \prod_{t=1}^n p(y_t|x_t) \\
&= \sum_{\{x_1,\dots,x_n\} } p(x_1) p(y_1|x_t) \prod_{t=2}^n p(x_t|x_{t-1})  p(y_t|x_t) \end{align}$$

This takes $\mathcal{O}(nm^n)$ operations!  

Instead, here's a $\mathcal{O}(nm^2)$ method that uses dynamic programming in the form of the forward algorithm:

$$p(y_1,\dots,y_n) = \sum_{x_n = 1}^m \underbrace{p(x_n, y_1,\dots,y_n)}_{\text{use forward algorithm}}$$

Now we just need those summands.

### Algorithm

First compute:
$$p(x_1,y_1) = p(y_1|x_1)p(x_1)$$

Then for each $t = 2,\dots,n$ compute:
$$ \begin{align} p(x_t,y_1,\dots,y_t) &= \sum_{x_{t-1} = 1}^m p(x_t,x_{t-1}, y_1,\dots,y_t) \\ 
&= \sum_{x_{t-1} = 1}^m p(y_t|x_t,x_{t-1}, y_1,\dots,y_{t-1}) p(x_t|x_{t-1}, y_1,\dots,y_{t-1}) p(x_{t-1}, y_1,\dots,y_{t-1}) \\
&= \underbrace{p(y_t|x_t)}_{\text{known}} \sum_{x_{t-1} = 1}^m  \underbrace{p(x_t|x_{t-1})}_{\text{known}} \underbrace{p(x_{t-1}, y_1,\dots,y_{t-1})}_{\text{forward algorithm result for $t-1$}} \\ \end{align}$$


# Backward algorithm

We have the forward part needed to compute $p(x_t|y_1,\dots,y_n)$.  Now we need the backward part.

### Algorithm

For $t = n$:
$$p(y_{n+1}|x_n) = 1$$
Note that the notation is a formality since there is no observed $y_{n+1}$.

Then for each $t = n-1,\dots,1$ compute:
$$\begin{align}p(y_{t+1},\dots,y_n|x_t) &= \sum_{x_{t+1} = 1}^m p(y_{t+1},\dots,y_n,x_{t+1}|x_t) \\
&= \sum_{x_{t+1} = 1}^m p(y_{t+2}, \dots,y_n|y_{t+1}, x_t, x_{t+1}) p(y_{t+1},x_{t+1}|x_t) \\
&= \sum_{x_{t+1} = 1}^m p(y_{t+2}, \dots,y_n|y_{t+1}, x_t, x_{t+1}) p(y_{t+1}| x_t, x_{t+1}) p(x_{t+1}|x_t)\\
&= \sum_{x_{t+1} = 1}^m  \underbrace{p(y_{t+2}, \dots,y_n| x_{t+1})}_{\text{backward algorithm result for $t+1$}} \underbrace{p(y_{t+1}| x_{t+1})}_{\text{known}} \underbrace{p(x_{t+1}|x_t)}_{\text{known}}  \\ \end{align}$$

# Conclusion

Now we have all the pieces to compute $p(x_t|y_1,\dots,y_n)$.  We can use this to find the most likely state at any time $t$.

Of course, this isn't enough by itself:

- The Viterbi algorithm finds the most likely sequence of hidden states (i.e. $\{x_1,\dots,x_n\}$ such that $p(x_1,\dots,x_n|y_1,\dots,y_n)$ is maximized ).

- The Baum-Welch algorithm uses the forward-backward algorithm to compute maximum likelihood estimates of the HMM parameters (i.e. the probabilities/distributions that we took as "given").  


# Postface

Had to learn stuff about HMMs while working on changepoint problems, and I figured I might as well organize some notes for future reference.  Hopefully someone else also finds these derivations useful.

Credit given to Jeffrey Miller's [mini-lectures](https://www.youtube.com/user/mathematicalmonk), which were really easy to digest for someone new to the material like myself.  

For an introduction to HMMs, I recommend reading Sections I-III of:
*Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. Proceedings of the IEEE, 77(2), 257-286.*
