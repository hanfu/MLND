description and summarization


## CLUSTERING



### k nearest neighbor

### single link clustering

### k-means clustering
start k points randomly
cluster data to nearest k
move k to averaging center clustered data
### Expectation Maximization
MAP: MLE with weights of prior's possibility distribution. MLE is a special case of uniform distribution
k-means with soft clustering: one point can belong to multiple gaussian clusters with partial weights, like MAP to MLE
Expectation: find P
Maximization: move k centers
will not diverge, non-decreasing likelihood, but may never converge(improving by tiny in each round)
may stucked at local optimal, so random start
work with all distributions

```py
clt = GaussianMixture(random_state=0, n_components=2)
clt.fit(data)
preds = clt.predict(data)
centers = clt.means_
score = silhouette_score(data, preds)
```

### Impossibility

## FEATURE SELECTION

### subset of features to pass along
### filtering and wrapping
wrapping: slow but useful
filtering: simple, but ignore bias
### relavent vs. useful
strong and weak relavence

## FEATURE TRANSFORMATION
### PCA ICA LDA RCA
### probability vs. linear algebra
### data structure