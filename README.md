Targeted Subspace Principal Component Analysis (tsPCA)
===========================================

tsPCA decomposes high-dimensional data into low-dimensional orthogonal subspaces where pre-specified signals of interest are demixed into distinct subspaces. For example, when you want to separate signals related to variables A, B, and C in the original data, tsPCA decomposes the data into demixed subspaces A, B, C, and a subspace that is free of the signals related to A, B, and C. When tsPCA is applied to a neural population activity, tsPCA decomposes neural population activity into orthogonal subspaces where targeted task-related signals are demixed. tsPCA can identify subspaces for any types of variables, including continuous, discrete, and categorical variables.

## Installation
It will be eventually made available from PyPl. Please download directly from github for now.

## How to use
tsPCA_demo.ipynb describes an example of the basic implementation.

