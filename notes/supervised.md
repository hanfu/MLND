selecting algos:
https://docs.microsoft.com/en-us/azure/machine-learning/studio/algorithm-choice
http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

center for all learnings: DATA

function approximation
induction from example to generalization
deduction is rule to specifications

* cls vs. reg: decided by output type: continuous or discrete

## cls

### decision tree
feature selection: max information gain
ID3: top-down learning algo, its bias
expressness of DT: 
prevent overfitting


### naive bayes
* naive being each feature is considered independent for calc.
```python
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(x_tr, y_tr)
pred = clf.predict(x_te)
```
able to swap cause and effect
MAP hypothesis
maximum likelihood hypothesis
Bayes optimal clf = a weighted vote
Bayes network - representation of joint distribution
sampling to do approx. inference
naive Bayes network


### logicstic reg
* pm: idea is # of misclsed data points,
but use log loss func in reality 
* just one linear line after log transform on both side log(wx+b) = log(y)
y = logistic_func(x)

#### logistic function
s-shaped curve f=L/(1+exp(-k(x-x0)))
L: f_max
x0: midpoint value
k:steepness

### svm
logistic reg line + seperate line space
optimize space by quardratic problem
points close to the line are called support vector
kernal trick for additional dimension projection


### kernel trick
create dimension for better feature
generalize x Transpose y to a k(x, y)


### neural network
perceptron: threshold unit
network of boolean functions
perceptron rules - finite time for linearly seperable
general differentiable rule: back-propagation, gradient descend
preference/restricted bias of nn
back-propagation: 
..* error cost of prediction to result, square difference
..* partial derivative of previous layers weight, output, and bias 
..* aggregate error cost for all outputs, determine the overall error cost, and recursive back

### Ensemble learning
bagging (bootstrap aggregation): randomly choose sample and learn the subset. when predict, average outcomes from all learning models (equal vote)
boosting: bagging with possibility weight
weak learner, agnostic: algo better than stat expectation
no overfitting

## reg

parametric vs. non-parametric
* space: small vs. big
* training time: long vs. short
* predition: fast vs. slow
* biased (with knowledge) vs. un-biased


### linear reg


### decision tree

### knn
non-parametric, instance-based
average k n ns
a lazy algo as lazy vs. eager learning
generalize nearest to similarity
work for both reg and clf
curse of dimensionality: info needed grows exp 2^d to generalize accurately


### kernel reg
weight knns by distance

