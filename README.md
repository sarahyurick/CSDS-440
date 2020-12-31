The spam datasets come from: 
http://www.ecmlpkdd2006.org/challenge.html#download

The "Toys and Games" and "Patio, Lawn and Garden" datasets come from: 
http://jmcauley.ucsd.edu/data/amazon/?fbclid=IwAR2bgs6Prgz_KOqgN8E0HD7-pPj5YDk7YNmVqLu-vS76LQuZ8kdfJ8mxs-w

The Titanic dataset comes from: 
https://www.kaggle.com/c/titanic

The Game of Thrones dataset comes from: 
https://www.kaggle.com/mylesoneill/game-of-thrones. 
The updated "isAlive" values were collected and prepared by Sarah Yurick.

To run self-taught clustering, do %run selftaught_clustering.py a b c d e f g, 
where a is the datasets to use ("spam", "amazon", or "survival"), 
b is the number of iterations to do self-taught clustering, 
c is a hyperparameter related to weighting the auxiliary versus target data, 
d is the maximum number of features to use, 
e is the maximum number of rows to use from the auxiliary dataset, 
f is the maximum number of rows to use from the target dataset, 
and g is to specify if you want to perform dimensionality reduction before self-taught clustering.
g may be any of the following: pca, sparse, truncated, kernel, LPP, 
pca_double, sparse_double, truncated_double, kernel_double, and LPP_double.