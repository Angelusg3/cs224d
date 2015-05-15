# [CS224d](http://cs224d.stanford.edu/) Notes
[TOC]

## Lecture #1
### TODO [Sat. 17, May]
- watch video of lecture #1
- read first three suggested readings

### Linear Algebra Review
#### Basic Notation

 - By convention, an n-dimensional vector is often thought of as a matrix with *n* rows and *1* column, known as a **column vector**. $$  \begin{equation}
     x =\begin{bmatrix}
         x_{1} \\
         x_{2} \\
         \vdots \\
         x_{n} \\
        \end{bmatrix}
  \end{equation} $$
  
#### Different Interpretation of Matrix-Vector Multiplication

 - inner product of each *row* of $A$ and $x$
 - **linear combination** of *columns* of $A$

 
#### Matrix-Matrix Product Interpretations
 - set of vector-vector products.
 - sum of outer products. $$ \begin{align*} 
C_{ij} &= \sum_{k = 1}^{n} A_{ik}B_{kj} \\
 &= A_{i1}B_{1j} + A_{i2}B_{2j} + \ldots + A_{in}B_{nj} \\
 &= [a_{1}b^{T}_{1}]_{ij} + [a_{2}b^{T}_{2}]_{ij} + \ldots + [a_{n}b^{T}_{n}]_{ij} \\ 
C &= \sum_{k = 1}^{n}a_{k}b^{T}_{k}
\end{align*}
$$
 - set of matrix vector product
 


### Probability Review

### Convex Optimization Review



## Lecture #2