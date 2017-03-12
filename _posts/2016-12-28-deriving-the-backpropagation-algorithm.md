---
layout: post
title: 'Deriving the backpropagation algorithm'
---

Here are some notes containing step-by-step derivations of the backpropagation algorithm for neural networks.  

This post serves more as a reference than as an introduction to the subject.  It assumes the reader is already familiar with neural networks and is comfortable with differentiation and matrix algebra. 

Regarding notation:

- All vectors are column vectors unless otherwise specified.
- The notation $g(x)$ for scalar function $g: \mathbb{R} \to \mathbb{R}$ and vector or matrix $x$ means the function is being applied element-wise.

<hr>

# Logistic regression

### Setup

We observe data $(y_1, x_1), \dots, (y_n, x_n)$ where $y_i \in \{0, 1\}$ and $x_i$ are vectors of length $m$.

We assume the model $y \sim$ Bernoulli$\left(p(x) \right)$ where the mean response is $p(x) = \phi(w^T x)$.  

$\phi(z) = \frac{1}{1 + e^{-z}}$ is the expit (aka logistic) function.

<hr>

### Loss function

We estimate $w$ using the maximum likelihood approach.  In other words, our goal is to minimize the negative log-likelihood loss (aka cross-entropy loss):

$$\begin{align}
\mathcal{L}(w) &= - \log \prod_{i=1}^n p(x_i)^{y_i} (1 - p(x_i))^{1-y_i}  \\
&= - \sum_{i=1}^n \left[ y_i \log p(x_i) + (1 - y_i) \log (1 - p(x_i)) \right] \\
&= - \sum_{i=1}^n \left[ y_i \log \phi(w^T x_i) + (1 - y_i) \log \left(1 - \phi(w^T x_i) \right) \right] 
\end{align}$$

<hr>

### Differentiation

The derivative of the expit function is $\phi'(z) = \phi(z) \left(1 - \phi(z)\right)$.

Then by chain rule, we derive the derivative of $\mathcal{L}(w)$ with respect to $w$:

$$\begin{align}
\frac{\partial \mathcal{L}(w)}{\partial w} &= \sum_{i=1}^n \frac{\partial \mathcal{L}_i}{\partial \phi(w^T x_i)} \frac{\partial \phi(w^T x_i)}{\partial w^T x_i} \frac{\partial w^T x_i}{\partial w} \\
&= -\sum_{i=1}^n  \left[ \frac{y_i}{\phi(w^T x_i)} + \frac{1 - y_i}{1 - \phi(w^T x_i)} \right] \phi(w^T x_i) \left(1 - \phi(w^T x_i) \right) x_i \\
&= - \sum_{i=1}^n \left[y_i \left( 1 - \phi(w^T x_i) \right) - (1 - y_i) \phi(w^T x_i)  \right] x_i \\
&= - \sum_{i=1}^n \left[ y_i - \phi(w^T x_i)   \right] x_i \\
\end{align}$$

 which is a vector of length $m$.  

<hr>

# Softmax regression

We think of our binary response as representing an observation's membership in one of two classes.  With this in mind, we now generalize the logistic regression problem to handle $K$ classes.

### Setup

Our response $y$ indicates membership in one of $K$ classes.  We'll use a one-hot encoding for the response, meaning each $y$ is a vector of length $K$.  For example, $y = [1, 0, \dots, 0]^T$ denotes membership in the first class.

Statistically, we assume $y \sim$ Categorical$\left(p_1(x), \dots, p_K(x)\right)$ where the mean response is $p(x) = \left[p_1(x), \dots, p_K(x)\right]^T = \phi(w^T x)$.  

<br>

Now $w$ is an $m \times K$ matrix instead of a vector.

And $\phi$ is now the softmax function which takes input vector $z = [z_1, \dots, z_d]^T$ and outputs vector:

$$ \phi(z) = \begin{pmatrix} e^{z_1} / \sum_j e^{z_j} \\ \vdots \\ e^{z_d} / \sum_j e^{z_j} \end{pmatrix}$$

<hr>

### Loss function

We're still minimizing negative log-likelihood loss:

$$\begin{align}
\mathcal{L}(w) &= -\sum_{i=1}^n y_i^T \log p(x_i) \\
&= -\sum_{i=1}^n y_i^T \log \phi(w^T x_i) 
\end{align}$$

<hr>

### Differentiation <a id="softmax-regression-differentiation"></a>

To keep things simple, let's do this derivation for a single observation (hence, dropping the summation and index $i$ for now).  Our simplified loss function is:

$$\mathcal{L}(w) = -y^T \log \phi(w^T x)$$

Also, let's establish some shorthand notation:  

- $\phi = \phi(z)$ is the output vector, and $\phi_k$ denotes its $k^{th}$ element
- $z = w^T x$ is the input vector to $\phi$, and $z_k$ denotes its $k^{th}$ element

Then by chain rule:

$$\frac{\partial \mathcal{L}(w)}{\partial w} = \frac{\partial \mathcal{L}}{\partial \phi} \frac{\partial \phi}{\partial z} \frac{\partial z}{\partial w}$$

where:

- $\frac{\partial \mathcal{L}}{\partial \phi}$ is a row vector of length $K$:

$$\left(-y \odot \frac{1}{\phi}\right)^T = \begin{pmatrix} -\frac{y_1}{\phi_1} & -\frac{y_2}{\phi_2} & \dots & -\frac{y_K}{\phi_K} \end{pmatrix} $$

- $\frac{\partial \phi}{\partial z}$ is a $K \times K$ matrix (see Appendix for deriving the [derivative of the softmax function](#derivative-of-the-softmax-function)):

$$\begin{pmatrix} \phi_1 (1 - \phi_1) & -\phi_1 \phi_2 & \dots & - \phi_1 \phi_K \\ 
- \phi_2 \phi_1 & \phi_2 (1-\phi_2) & \dots & - \phi_2 \phi_K \\ 
\vdots  & \vdots & \ddots & \vdots \\ 
-\phi_K \phi_1 & -\phi_K \phi_2 & \dots & \phi_K (1-\phi_K) \end{pmatrix}$$

- $\frac{\partial z}{\partial w}$ is a $K \times m \times K$ tensor.  See Appendix for [derivation](#deriving-the-tensor-for-softmax-regression).  

<br>

Multiplication of the first two terms gives a row vector of length $K$:

$$\begin{align}
\frac{\partial \mathcal{L}}{\partial \phi}  \frac{\partial \phi}{\partial z}&= \begin{pmatrix} - y_1 (1-\phi_1) + \sum_{k \neq 1} y_k \phi_1 &  \dots & - y_K (1-\phi_K) + \sum_{k \neq K} y_k \phi_K \end{pmatrix} \\
&= \begin{pmatrix} - y_1 + \phi_1 \sum_{k=1}^K y_k  &  \dots & - y_K + \phi_K \sum_{k=1}^K y_k  \end{pmatrix} \\
&= \begin{pmatrix} \phi_1 - y_1  & \dots & \phi_K - y_K \end{pmatrix} \\ 
&= \left[\phi(w^T x) - y\right]^T
\end{align}$$

<br>

Then multiplying the resulting row vector with the tensor term gives an $m \times K$ matrix:

$$\begin{align}
\frac{\partial \mathcal{L}}{\partial \phi} \frac{\partial \phi}{\partial z} \frac{\partial z}{\partial w} &= \left[\phi(w^T x) - y\right]^T \left[ \frac{\partial w^T x}{\partial w}\right] \\
&= x \left[\phi(w^T x) - y\right]^T
\end{align}$$

It turns out the tensor multiplication works out to this simple form.  See Appendix for [details](#showing-steps-in-vector-tensor-multiplication).

<br>

Finally, putting everything back in terms of $n$ observations, we have the derivative of $\mathcal{L}(w)$ with respect to $w$:

$$\frac{\partial \mathcal{L}(w)}{\partial w} = -\sum_{i=1}^n x_i \left[y_i - \phi(w^T x_i) \right]^T$$

which is an $m \times K$ matrix. 

This looks very similar to the derivative in the logistic regression setting (which is honestly kind of anti-climactic after all that work).  In fact, softmax regression for $K = 2$ is equivalent to logistic regression.

<hr>


# Neural network

### Setup

The neural network framework still assumes $y \sim$ Categorical$\left(p_1(x), \dots, p_K(x)\right)$ but now with a recursively-defined mean response:

$$\begin{align}
\left[p_1(x), \dots, p_K(x)\right]^T = h^{(L)} &= \phi_L \left(z^{(L)} \right) \\
h^{(L-1)} &= \phi_{L-1}\left(z^{(L-1)} \right) \\
&\vdots \\
h^{(1)} &= \phi_1\left(z^{(1)} \right) \\
h^{(0)} &= x 
\end{align}$$

where $z^{(l)} = w_l^T h^{(l-1)}$ for $l = 1,\dots,L$.

- $h^{(0)}, \dots, h^{(L)}$ are vectors of length $m_l$ representing the collection of nodes at layer $l = 0, \dots, L$.  Notably, layers $0$ and $L$ are referred to as the input and output layers, respectively.

- $\phi_1, \dots, \phi_L$ are activation functions that map from $\mathbb{R}^{m_l}$ to $\mathbb{R}^{m_l}$.  

	- $\phi_L$ is typically the softmax function since we want $h^{(L)}$ entries, like probabilities, to sum to $1$.  Hence, when $L = 1$, a neural network with the softmax activation for its output layer is equivalent to softmax regression.

	- $\phi_1,\dots,\phi_{L-1}$ are often instead characterized by a nonlinear scalar function (e.g. expit, $\tanh$, ReLU) applied element-wise to input vectors.

- $w_1, \dots, w_L$ are $m_{l-1} \times m_l$ parameter matrices.  Notably, $w_1$ has dimensions $m \times m_1$ to match the input vector $x$, and $w_L$ has dimensions $m_{L-1} \times K$ to match the response $y$.


<hr>

### Loss function

Our goal is still to minimize the negative log-likelihood loss:

$$ \mathcal{L}(w_{1:L}) = - \sum_{i=1}^n y_i^T \log h_i^{(L)} $$

<hr>

### Differentiation

As it turns out, differentiation for neural networks looks similar to differentiation for softmax regression.

For simplicity, let's again assume a single observation, so the loss function is:

$$ \mathcal{L}(w_{1:L}) = - y^T \log h^{(L)} $$

Then:

$$\begin{align} \frac{\partial \mathcal{L}(w_{1:L})}{\partial w_L} &= \underbrace{\frac{\partial \mathcal{L}}{\partial h^{(L)}} \frac{\partial h^{(L)}}{\partial z^{(L)}}}_{\delta^{(L)}} \frac{\partial z^{(L)}}{\partial w_L} \\
\frac{\partial \mathcal{L}(w_{1:L})}{\partial w_{L-1}} &= \underbrace{\frac{\partial \mathcal{L}}{\partial h^{(L)}} \frac{\partial h^{(L)}}{\partial z^{(L)}} \frac{\partial z^{(L)}}{\partial h^{(L-1)}} \frac{\partial h^{(L-1)}}{\partial z^{(L-1)}}}_{\delta^{(L-1)}} \frac{\partial z^{(L-1)}}{\partial w_{L-1}} \\ 
&\vdots \\
\frac{\partial \mathcal{L}(w_{1:L})}{\partial w_1} &= \underbrace{\frac{\partial \mathcal{L}}{\partial h^{(L)}}  \frac{\partial h^{(L)}}{\partial z^{(L)}} \left[ \prod_{l=2}^L \frac{\partial z^{(l)}}{\partial h^{(l-1)}}  \frac{\partial h^{(l-1)}}{\partial z^{(l-1)}} \right]}_{\delta^{(1)}}  \frac{\partial z^{(1)}}{\partial w_1}
\end{align}$$

where:

- $\frac{\partial \mathcal{L}}{\partial h^{(L)}}$ is a row vector of length $K$:

$$\left(-y \odot \frac{1}{h^{(L)}}\right)^T = \begin{pmatrix} -\frac{y_1}{h^{(L)}_1} & -\frac{y_2}{h^{(L)}_2} & \dots & -\frac{y_K}{h^{(L)}_K} \end{pmatrix} $$

- $\frac{\partial h^{(l)}}{\partial z^{(l)}}$ are $m_l \times m_l$ matrices (i.e. derivative of activation function $\phi_l$ with respect to input vector $z^{(l)}$). 

- $\frac{\partial z^{(l)}}{\partial h^{(l-1)}} = \frac{\partial w_l^T h^{(l-1)}}{\partial h^{(l-1)}} = w_l^T$ are $m_l \times m_{l-1}$ matrices. 

- $\frac{\partial z^{(l)}}{\partial w_l}$ is an $m_l \times m_{l-1} \times m_l$ tensor (with same form as $\frac{\partial z}{\partial w}$ for softmax regression with $h_j^{(l-1)}$ replacing the $x_j$ elements).

for $l = 1, \dots, L$.

<br>

Notice that $\delta^{(l)} = \frac{\partial \mathcal{L}}{\partial z^{(l)}} = \frac{\partial \mathcal{L}}{\partial h^{(L)}}  \frac{\partial h^{(L)}}{\partial z^{(L)}} \cdots  \frac{\partial h^{(l)}}{\partial z^{(l)}} $ is a row vector, and $\frac{\partial z^{(l)}}{\partial w_l}$ is a tensor of a familiar form.  Then using what we learned from softmax regression:

$$\delta^{(l)}\frac{\partial z^{(l)}}{\partial w_l} = h^{(l-1)} \delta^{(l)}$$

<br>

Finally, returning to $n$ observations, the derivative of $\mathcal{L}(w_{1:L})$ with respect to $w_l$ is:

$$\frac{\partial \mathcal{L}(w_{1:L})}{\partial w_l} = \sum_{i=1}^n h_i^{(l-1)} \delta_i^{(l)}$$

for $l = 1,\dots,L$.

<hr>

### Backpropagation

The backpropagation algorithm is simply an efficient way for computing the derivatives by noticing that $\delta^{(l)}$ terms can be reused between layers:

1. Compute $h_i^{(1)}, \dots, h_i^{(L)}$ for $i = 1,\dots,n$. 

2. Initialize $\delta_i = \frac{\partial \mathcal{L}_i}{\partial h_i^{(L)}}\frac{\partial h_i^{(L)}}{\partial z_i^{(L)}}$ for $i = 1,\dots,n$.

3. For $l = L$ down to $1$:

	a. Compute $\frac{\partial \mathcal{L}(w_{1:L})}{\partial w_l} = \sum_{i=1}^n h_i^{(l-1)} \delta_i$.

	b. Update $\delta_i \gets \delta_i \frac{\partial z^{(l)}}{\partial h^{(l-1)}} \frac{\partial h_i^{(l-1)}}{\partial z_i^{(l-1)}}$ for $i = 1,\dots,n$.

<br>

The $\delta^{(l)}$ computations comprise most of the heavy-lifting since: $\frac{\partial h^{(l)}}{\partial z^{(l)}} = \phi'(z^{(l)})$ and $\frac{\partial z^{(l)}}{\partial h^{(l-1)}} = w_l^T$ can be large matrices.  There's not much we can do about the latter term, but see the [Appendix](#faster-computation-of-delta-terms) for ways to speed up multiplication involving the former term.

<hr>

# Appendix

### Derivative of the softmax function

Since the softmax function is $\phi: \mathbb{R}^d \to \mathbb{R}^d$,  its derivative is a Jacobian matrix:

$$\phi'(z) = \begin{pmatrix}
\frac{\partial \phi(z)_1}{z_1} & \frac{\partial \phi(z)_1}{z_2} & \dots & \frac{\partial \phi(z)_1}{z_d} \\
\frac{\partial \phi(z)_2}{z_1} & \frac{\partial \phi(z)_2}{z_2} & \dots & \frac{\partial \phi(z)_2}{z_d} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial \phi(z)_d}{z_1} & \frac{\partial \phi(z)_d}{z_2} & \dots & \frac{\partial \phi(z)_d}{z_d} \\
\end{pmatrix}$$

Let's derive each entry in this matrix.  We'll index the rows by $i$ and columns by $j$.

For $i = j$ entries, by quotient rule:

$$\begin{align}
\frac{\partial \phi(z)_i}{\partial z_i} &= \frac{e^{z_i} \sum_k e^{z_k} - e^{z_i} e^{z_i}}{ \sum_k e^{z_k} \sum_k e^{z_k} } \\
&= \frac{e^{z_i}}{\sum_k e^{z_k}} \left(1 - \frac{e^{z_i}}{\sum_k e^{z_k}} \right)\\
&= \phi(z)_i \left( 1 - \phi(z)_i \right)
\end{align}$$

And for $i \neq j$ entries, by quotient rule:

$$\begin{align}
\frac{\partial \phi(z)_i}{\partial z_j} &= \frac{0 \sum_k e^{z_k} - e^{z_i} e^{z_j}}{ \sum_k e^{z_k} \sum_k e^{z_k} } \\
&= - \frac{e^{z_i}}{\sum_k e^{z_k}} \frac{e^{z_j}}{\sum_k e^{z_k}} \\
&= - \phi(z)_i \phi(z)_j
\end{align}$$

### Deriving the tensor for softmax regression

This section pertains to deriving the form of the tensor mentioned in [this section](#softmax-regression-differentiation).

$$\frac{\partial z}{\partial w} = \frac{\partial w^T x}{\partial w} = 
\begin{pmatrix} 
\frac{\partial (w^T x)_1}{\partial w} \\ \frac{\partial (w^T x)_2}{\partial w} \\ \dots \\ \frac{\partial (w^T x)_K}{\partial w} 
\end{pmatrix} $$

is a rank-3 tensor of dimension $K \times m \times K$ where:

$$\begin{align}
\frac{\partial (w^T x)_1 }{\partial w} &= \begin{pmatrix}\frac{\partial \sum w_{j1} x_j}{\partial w_{11}} & \frac{\partial \sum w_{j1} x_j}{\partial w_{12}} & \dots & \frac{\partial \sum w_{j1} x_j}{\partial w_{1K}} \\ \frac{\partial \sum w_{j1} x_j}{\partial w_{21}} & \frac{\partial \sum w_{j1} x_j}{\partial w_{22}} & \dots & \frac{\partial \sum w_{j1} x_j}{\partial w_{2K}} \\ \vdots & \vdots & \ddots \vdots \\ \frac{\partial \sum w_{j1} x_j}{\partial w_{m1}} & \frac{\partial \sum w_{j1} x_j}{\partial w_{m2}} & \dots & \frac{\partial \sum w_{j1} x_j}{\partial w_{mK}} \end{pmatrix} \\
&= \begin{pmatrix} x_1 & 0 & \dots & 0 \\ x_2 & 0 & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ x_m & 0 & \dots & 0 \end{pmatrix}
\end{align}$$

$$\frac{\partial (w^T x)_2 }{\partial w} = \begin{pmatrix} 0 & x_1 & 0 & \dots & 0 \\ 0 & x_2 & 0 & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & x_m & 0 & \dots & 0 \end{pmatrix} $$

$$\vdots$$

$$\frac{\partial (w^T x)_K }{\partial w} = \begin{pmatrix} 0 & \dots & 0 & x_1 \\ 0 & \dots & 0 & x_2 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & \dots & 0 & x_m \end{pmatrix} $$

### Showing steps in vector-tensor multiplication

This section pertains to showing details for the tensor multiplication step mentioned in [this section](#softmax-regression-differentiation).

Let $a$ be a vector of length $K$.  Then:

$$\begin{align}
a^T \left[\frac{\partial w^T x}{\partial w}\right] &=  \begin{pmatrix} a_1 & a_2 & \dots & a_K \end{pmatrix}\begin{pmatrix} 
\frac{\partial (w^T x)_1}{\partial w} \\ \frac{\partial (w^T x)_2}{\partial w} \\ \dots \\ \frac{\partial (w^T x)_K}{\partial w} 
\end{pmatrix} \\
&= \sum_{j=1}^K a_j \frac{\partial (w^T x)_j }{\partial w} \\
&= a_1 \begin{pmatrix} x_1 & 0 & \dots & 0 \\ x_2 & 0 & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ x_m & 0 & \dots & 0 \end{pmatrix} + \dots + a_K \begin{pmatrix} 0 & \dots & 0 & x_1 \\ 0 & \dots & 0 & x_2 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & \dots & 0 & x_m \end{pmatrix} \\
&= \begin{pmatrix} a_1 x_1 & a_2 x_1 & \dots & a_K x_1 \\ a_1 x_2 & a_2 x_2 & \dots & a_K x_2 \\ \vdots & \vdots & \ddots & \vdots \\ a_1 x_m & a_2 x_m & \dots & a_K x_m \end{pmatrix} \\
&= x a^T
\end{align}$$

Technically, the result is a $1 \times m \times K$ tensor, but we can effectively drop the first dimension and just treat the result as an $m \times K$ matrix.

### Derivatives of other activation functions

While $\phi_L$ is usually the softmax function, we usually defer to other choices for $\phi_1,\dots,\phi_{L-1}$. 

For example, popular choices include element-wise application of

- $\phi(z) = \tanh(z) = \frac{1 - e^{-2z}}{1 + e^{-2z}}$ where $\phi'(z) = 1 - \tanh^2(z)$

- $\phi(z) = ReLU(z) = \max(0, z)$ where $\phi'(z) = \mathbf{1}_{z \gt 0}$

Then the derivative of the activation function $\phi_l$ characterized by $\phi$ is a diagonal matrix:

$$\frac{\partial \phi_l(z^{(l)})}{z^{(l)}} = \begin{pmatrix}  \phi'(z_1^{(l)}) & 0 & \dots & 0 \\ 0 & \phi'(z_2^{(l)}) & \dots & 0 \\ \vdots  & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & \phi'(z_{m_l}^{(l)})  \end{pmatrix}$$


### Faster computation of delta terms


#### Softmax output layer activation 

Since $\phi_L$ is likely the softmax function, we can use the result from softmax regression:

$$\delta^{(L)} = \left[h^{(L)} - y\right]^T$$

as opposed to performing the full $\frac{\partial \mathcal{L}}{\partial h^{(L)}}  \frac{\partial h^{(L)}}{\partial z^{(L)}}$ multiplication.

#### Activation functions with diagonal Jacobians

On the other hand, $\phi_{L-1},\dots,\phi_1$ are usually chosen to be element-wise applications of $\phi$ (e.g. expit, $\tanh$, ReLU).  Unlike the softmax function, their derivatives are diagonal matrices:

$$\frac{\partial \phi_l(z)}{\partial z} = \begin{pmatrix} \phi'(z_1) & 0 & 0 & \dots & 0 \\ 0 & \phi'(z_2) & 0 & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \dots & \phi'(z_d) \end{pmatrix}$$

Then in each iteration of backpropagation, the $\delta^{(l)} \to \delta^{(l-1)}$ update can be simplified to:

$$\begin{align}\delta^{(l-1)} &= \delta^{(l)} \frac{\partial z^{(l)}}{\partial h^{(l-1)}}  \frac{\partial h^{(l-1)}}{\partial z^{(l-1)}} \\
&= \delta^{(l)} w_l^T \frac{\partial h^{(l-1)}}{\partial z^{(l-1)}} \\
&= \begin{pmatrix}  \phi'(z^{(l-1)}_1) \sum_j^{m_l} \delta_j^{(l)} w^{(l)}_{1j}  \\  \phi'(z^{(l-1)}_2) \sum_j^{m_l} \delta_j^{(l)} w^{(l)}_{2j} \\ \vdots \\  \phi'(z^{(l-1)}_{m_{l-1}}) \sum_j^{m_l} \delta_j^{(l)} w^{(l)}_{m_{l-1}j} \end{pmatrix}^T \\
&= \left[ \delta^{(l)} w_l^T\right] \odot \left[\phi'\left(z^{(l-1)}\right)\right]^T
\end{align}$$

Thus, we can replace matrix multiplication of $\frac{\partial h^{(l-1)}}{\partial z^{(l-1)}}$ with the cheaper element-wise multiplication of $\left[\phi'\left(z^{(l-1)}\right)\right]^T$.

# Miscellaneous

Here's a collection of topics that I felt were loosely relevant, but I didn't know how to fit them into the post.

### Bias

We've failed to address inclusion of a bias term among the parameters.  

For logistic/softmax regression, this can be done by augmenting the inputs $\tilde{x} = [1, x]^T$.  The weight vector $w$ will include an additional element to ensure that the product $\tilde{z} = w^T \tilde{x}$ is computable.  The learned value is the bias.

For neural networks, we can similarly augment the inputs $\tilde{h}^{(l)} = [1, h^{(l)}]^T$ for $l = 0, \dots, L-1$.  The weight matrices $w_1, \dots, w_L$ will each include an additional row to ensure the products $\tilde{z}_l = w_l^T \tilde{h}^{(l-1)}$  are computable for $l = 1,\dots,L$. 

### Quadratic loss

Our derivations have been for negative log-likelihood loss for classification.  For regression problems in which $y$'s can take value beyond $\{1, 0\}$, the quadratic loss is often used instead:

$$ \mathcal{L}(w_{1:L}) = \frac{1}{2} \sum_{i=1}^n \left( y_i - h_i^{(L)}\right)^T \left( y_i - h_i^{(L)}\right) $$

The same chain rule pattern holds, but now the form of $\frac{\partial \mathcal{L}}{\partial h^{(L)}}$ is different:

$$\frac{\partial \mathcal{L}}{\partial h^{(L)}} = \left(h^{(L)} - y \right)^T$$

Everything else remains the same.

### Regularization

Maximum likelihood tends to overfit and we often include a penalty term in the loss function to guard against this.  For example:

$$\mathcal{L}(w_{1:L}) = \sum_{i=1}^n y^T \log h_i^{(L)} + \frac{\lambda}{2} \Vert  w \Vert_2^2$$

where $\lambda > 0$ and $\Vert w \Vert_2^2$ denotes the sum of squared $w_{i,j}^{(l)}$ (except bias terms).

Then:

$$\frac{\partial \mathcal{L}(w_{1:L})}{\partial w_l} = \frac{\partial \sum_{i=1}^n y^T \log h_i^{(L)}}{\partial w_l} + \frac{\lambda}{2} \frac{\partial \Vert w \Vert_2^2}{\partial w_l}$$

for which the former term we've already derived.  The latter term is:

$$\frac{\partial \Vert w \Vert_2^2}{\partial w_l} = 2 w_l$$

which is an $m_{l-1} \times m_l$ matrix.

Since regularization only contributes an additive term that doesn't depend on anything else, the backpropagation algorithm remains unchanged.  We can simply add $\lambda w_l$ to the corresponding $\frac{\partial \mathcal{L}(w_{1:L})}{\partial w_l}$ values afterwards.

### Training

Gradient-based optimization procedures begin with an initial guess $w^{(0)}$ and repeatedly perform an update step until convergence (i.e.$\vert w^{(t+1)} - w^{(t)}\vert \lt \epsilon$).  

Here, $w^{(t)}$ refers to the entire set of parameters at iteration $t$.  This can include all $w_1,\dots,w_L$ of a neural network.

<br>

Gradient descent update:

$$w^{(t+1)} \gets w^{(t)} - \eta \frac{\partial \mathcal{L}(w^{(t)})}{\partial w^{(t)}} $$

where $\eta \gt 0$ is the step size (aka learning rate) that is usually chosen to decrease per iteration.

<br>

For logistic regression, we can use the faster Newton's method update:

$$w^{(t+1)} \gets w^{(t)} - H_{\mathcal{L}}^{-1}(w^{(t)}) \frac{\partial \mathcal{L}(w^{(t)})}{\partial w^{(t)}} $$

where $H_{\mathcal{L}}^{-1}$ is the inverse of the Hessian of $\mathcal{L}(w)$:

$$ H_{\mathcal{L}}^{-1}(w^{(t)}) = \sum_{i=1}^n \phi(w^T x_i) \left(1 - \phi(w^Tx_i) \right) x_i x_i^T$$

<br>

For neural networks, gradient descent can get stuck in local optima since the loss function is no longer convex like it is for logistic regression.  Also, models with this many parameters typically require large training sets which may not fit in memory at once.

A popular alternative is (minibatch) stochastic gradient descent which involves computing the gradient using some manageable $n_{batch} \lt\lt n$ subset of points.  The resulting approximate gradient is noisier which can help the algorithm avoid local optima.  