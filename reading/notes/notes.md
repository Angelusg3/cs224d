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

#### Basic Properties of Matrix Multiplication
- Matrix multiplication is associative: $(AB)C = A(BC)$ 
- Matrix multiplication is distributive : $ A(B + C) = AB + AC $
- Matrix multiplication is not commutative: $ AB \ne BA $

#### Operations and Properties
##### The Identity Matrix and Diagonal Matrices
##### The Transpose
- $(A^{T})^{T} = A$
- $(AB)^{T} = B^TA^T$
- $ (A + ⧸⧸B)^{T} = A^T + B^T$ 
##### Symmetric Matrices
A square matrix $A \in \mathbb{R} ^{n×n}$ is **symmetric** if $A = A^T$ and **anti-symmetric** if $ A = - A^T$.
Any square matrix can be represented as a sum of symmetric matrix and anti-symmetric matrix:
$$ A = \frac{1}{2}(A + A^T) + \frac{1}{2}(A - A^T)$$
 It is common to denote the set of all symmetric matrices of size n as $\mathbb{S}^n$ , so that $A \in \mathbb{S}^n$ means that $A$ is a symmetric $n \times n$ matrix.

##### The Trace
The **trace** of square matrix $A \in \mathbb{R} ^{n×n}$ is: $$trA = \sum_{i = 1}^{n}A_{ii}$$.
The trace has the following properties:
- $trA = trA^T$
- $tr(A + B) = trA + trB$
- $tr(tA) = t(trA)$
- $tr AB = tr BA$
- $ tr ABC = tr BCA = tr ACB $

some notes


### Probability Review

### Convex Optimization Review



## Lecture #2