
# PRE-PROCESSING

## make datasets
```python
import pandas, numpy
rawdata = pandas.read_csv(file.csv)
dataframe = numpy.asarray(rawdata)
df[[col1, col2]]
X = df[:,0:coln]
y = data[2]
```

## save/load pickes
refer to tf_tutorails/1.ipynb

## inspection
Seaborn Scatter Matrix 
yourdf.describe().astype(int)
pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

### bag of words
* remove punctuation
yourstring.translate(str.maketrans("","", string.punctuation))
* tokenize words
yourstring.split(" ")
* count word frequences
collection.Counter(yourstring)
returns a Counter dict
* sklearn module way
from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer(lowercase, stop_words, token_pattern)
count_vector.fit()/transform()/fit_transform()
..* fit() and fit_transform() will auto scale the data, so use them for training; scaler params are saved and used in transform() for testing

### missing data
### generalization
### normalization ( scaling to [0,1])
log transform for large spanning data np.log(x+1)
```py
yourdf[newcol] = yourdf[oldcol].apply(lambda x: np.log(x + 1))
df_log = np.log(df)
true_point = np.exp(log_point)
```
normalization of all features to [0,1] for equal significance
```py
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)

yourdf_transform = pd.DataFrame(data = yourdf)
yourdf_transform[newcol] = scaler.fit_transform(yourdf[oldcol])

```
### standardization (0 mean with unit std)
works better than normalization
required for graident descent

(regularization is a mean to prevent overfitting, by regulate parameter numbers using cost functions in the performance metrics)

### encoding
pandas.get_dummies(yourdf)

# MODELING

## select model (train)
* model fit
```python
from sklearn import Clf
yourclf = Clf()
yourclf.fit(x,y)
```
* model selection
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.model_selection import KFold
kf = FKold(total)

```
```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


def fit_model(X, y):
    cv_sets = ShuffleSplit(n_splits=10, test_size = 0.20, random_state = 0)

    regressor = DecisionTreeRegressor()
    params = {'max_depth':list(range(1,11))}
    scoring_fnc = make_scorer(performance_metric)
	grid = GridSearchCV(estimator=regressor, param_grid = params, scoring = scoring_fnc, cv=cv_sets)
	model = grid.fit(X, y)
    return model.best_estimator_

yourmodel = fit_model(x,y)
yourmodel.get_params()
```

### feature selection
```py
from sklearn.base import clone
df_reduced = df[df.columns.values[(np.argsort(importancelist)[::-1])[:5]]]
clf_reduced = (clone(clf)).fit(X_train_reduced, y_train)

```

```py
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()
```

### PCA
```py
from sklearn.decomposition import PCA
pca = PCA(n_components=6, random_state=0)
pca_95 = PCA(0.95)#least components that achieve 0.95 explained variance ratio
pca.fit_transform(data)
biplot
pca.explained_variance_ratio_
log_point = pca.inverse_transform(pca_point)

```

* training set vs. testing set
..* cross_validation, KFold for training set

* grid search (parameters) for cross validation
to print the grid with ordered params
```py
plt.plot(*zip(*sorted(result.items())))
```

* underfit vs. overfit
..* underfit = high bias, overall difference, poor training set
..* overfit = high variance, local difference, poor testing set

* Model Complexity Graph: compare train/test errors on each complex degree, ie. poly degree

* Learning Curve: series of graph that (x,y) be (metric, total training points) for both training and testing sets, at different complex degrees

## metrics
```python
from sklearn import accuracy_score

y_pred = yourclf.predict(x_test)
acc = accuracy_score(y_test, y_predc)
```

### clf: Confusion Matric
|        | GuessP | Guess N |
| -----  | ------ | ------- |
| Real P | True P | False N |
| Real N | False P|  True N |

* Type 1 Error: False P, healthy 'patient'
* Type 2 Error: False N, sick 'goodman'


SENSITIVITY (RECALL): TP / RealP
SPECIFICITY: TN / RealN
* ACCURACY: TP+TN / total
..* not a good metrics if data is skewed
..* for naive bayes, p(Guess P) = SENSITIVITY * RealP + (1-SPECIFICITY) * (RealN)

* PRECISION (of guess positives): TP / Guess P
* RECALL (of real positives)(SENSITIVITY): TP / Real P
* F1 = 2* (PxR)/(P+R)
..* harmonic mean of PRECISION and RECALL
* F-beta: for beta, 0==precision .. 1 .. infinity==recall
* ROC curve: x,y as (TP/allP, NP/allP) for each where x is each data point


### reg: Error
* Mean Absolute Error: absolute difference
..* not differentiatable
* Mean Squared Error: square the abs error
* R-square (coefficient of determination): define a simple reg (average) and calc MSE, R2 = 1-(MSE_yourmodel/MSE_simple)
* R2 score of 0.40 means that 40 percent of the variance in Y is predictable from X.