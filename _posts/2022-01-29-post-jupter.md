---
title: My Jupyter post
categories:
- General
excerpt: |
  This is my own Jupyter post.
feature_text: |
  ## The Jupyter post
  Let's make this great.
feature_image: "https://picsum.photos/2560/600?image=1021"
image: "https://picsum.photos/2560/600?image=733"
---


```python
# Gaussian Processes
```


```python
# https://towardsdatascience.com/understanding-gaussian-process-the-socratic-way-ba02369d804
```

The core idea about the gaussian process is that instead of computing the posterior distributions over the paramateres as we did for all the other models here we are computing the posterior distribution over functions. In the GP given a set of points we are trying to get the ddistribution of the function that covers this points as good/close as it possible given our prior beliefs about the functions distribution. To do so, the GP makes several assumptions one of which is that the functions are jointly forming a multivariate Gaussian distribution. Let $X_{tr}, X_{te}$ be our training and test data correspondingly. Then, according to GP we have $p(\begin{pmatrix}f(X_{tr})\\f(X_{te})\end{pmatrix}) = N(\begin{pmatrix}m(X_{tr})\\m(X_{te})\end{pmatrix}, \begin{pmatrix}k(X_{tr},X_{tr}) & k(X_{tr},X_{te})\\
k(X_{te},X_{tr}) & k(X_{te},X_{te})\end{pmatrix})$.

Another assumption that we are making is that the response variable $y$ also has a Gaussian distribution conditional on the $f(x)$ i.e. $p(y|f(x),\theta) = N(y|f(x), \sigma^2I)$ or $y = f(x) + \epsilon, p(\epsilon) = N(0, \sigma^2I)$.  
In this setup our task is to get the predictive posterior $p(f(X_{te})|y,X_{te}, X_{tr}, y)$.

Since the distribution of the functions is jointly Gaussian it is easy to get the marginal distribution of $f(X_{tr})$.  
So, we have a prior: $p(f(X_{tr})) = N(f(X_{tr})|m(X_{tr}), K)$  
where $f(x)$-is not a function but rather a random variable,  
$m(x)$-is a function that characterises our knowledge about the mean,  
$K$-is the Gramm matrix.  
Now, note that by Linear Gaussian model usage we can get the marginal distribution for $y$ to be $p(y|X_{tr}, \theta) = \int p(y|f(X_{tr}), \theta)p(f(x))df(x) = \int N(y|f(X_{tr}), \sigma^2I)N(f(X_{tr})|m(X_{tr}), K)df(x) = N()$.
At this point we can write the joint distribution of $f(X_{te}), y$ as a multivariate Gaussian. The only problem is that we don't yet know $cov(y, f(X_{te}))$. Let's compute it as:

$cov(y, f(X_{te})) = E[(y-m(X_tr))(f(X_{te}) - m(X_{te}))] = E[(f(X_{tr}) + \epsilon - m(X_tr))(f(X_{te}) - m(X_{te}))] = E[(f(X_{te}) - m(X_{te}))(f(X_{tr}) - m(X_tr)) + (f(X_{te}) - m(X_{te}))\epsilon] = cov(f(X_{te}), f(X_{tr})) + E[(f(X_{te}) - m(x_{te})\epsilon)] = K(X_{te}, X_{tr}) + 0 = K(X_{te}, X_{tr})$.

Hence, we have:
$p(\begin{pmatrix}y\\f(X_{te})\end{pmatrix}) = N(0, \begin{pmatrix}K_{y}&K(X_{te}, X_{tr})\\K(X_{te}, X_{tr}) & K(X_{te}, X_{te})\end{pmatrix})$.  
It is easy to get from this the posterior distribution of $p(f(X_{te})|X_{te}, X_{tr}, y) = N(f(X_{te})|u_*,\Sigma_*)$ where  
$u_* = K_*^TK_y^{-1}y$  
$\Sigma_* = K_{**} - K_*^TK_y^{-1}K_*$


```python
# https://github.com/jasonweiyi/understanding_gaussian_process/blob/master/gp.py
import numpy as np

class GP:
    def __init__(self, kernel, noise_sigma):
        self.kernel = kernel
        self.sigma = noise_sigma
        
    def __get_gram(self, X1, X2):
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i][j] = self.kernel(X1[i], X2[j])
        return K
    
    def posterior(self, X_train, y, X_test):
        k_Xtr = self.__get_gram(X_train, X_train)
        k_y = 
        inv_k_y
        k_X_tr_te = self.__get_gram(X_train, X_test)
        k_X_te = self.__get_gram(X_test, X_test)
        mean = k_X_tr_te@inv_k_y@y
        covariance = K_X_te - k_X_tr_te.T@inv_k_y@k_X_tr_te
        return mean, covariance
```
