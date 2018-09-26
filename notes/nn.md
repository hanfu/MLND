perception tool
	image, language, robot reaction, speech

## Multi-layer Perceptron NN
with ReLU, it is a multinomial Logistic Clf: logistic regression model for each of multiple output labels
DL is about to scale up the Multinomial Logistic Clf
* large dataset
* huge computation

### linear reg: wx + b = y
w: weight
b: bias
w, x, b can be vectors and they take products to become w1x1 + w2x2...

w.shape = [x.lenghth * y.length]
b.length = y.length

one basic but dreamy optimization:
given a line of w1xi+w2xj+b=0, result = y(0 or 1)
if point a mistakenly labeled:
```py
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b
```
generalize it we find deltaW = deltaY * Xi, given y=1 or 0
which is __graident descent!!__: (y_goal - y_t) * x_t
updated weights: w(t + 1) = w(t) + dy * x(t)
y_t+1 = x(t) ⋅ w(t + 1) 
	  = x(t) * w(t) + x(t) * (dy x(t)) 
	  = y_t+ dy [x(t) ⋅ x(t)], where dot product is always positive
	  so y_t+1 is closer to y_goal
in gd, the correctly labeled point will tell the model "go further" to secure

### step function
each percepton is one regression followed by one step function
fire or not?

#### linear property:
fast, GPU-ready
stable, small change in input means small change in output
can represent input addition(x1+x2), but not input multiplication(x1 * x2)
good derivative, constant
__need non-linearality__: ReLU, or multiple layers

#### Rectified Linear Unit
easy function and easy derivative
can add into the model between linear models

#### Multi-Laye Perceptron
linearly combine two linear model gets a non-linear model
re-format the connections a pyrimaid becomes a neural network!
x1\
x2-p1-out
x1-p2/
x2/

x1-p1\
  x  out
x2-p2/

p1 and p2 is the new 'hidden layer'

#### data requirement
standardized input: mean at 0, equal small std
uniform distributed weights

### logistic reg: log(wx+b) = log(y)
each input param x_i have its own LR for each output label y_i

### softmax function, sigmoid curve
for image clf: softmax(y_i) to make correct label to 1 and others to 0 and sum(y_i) = i

each label have its own linear reg model (called logit), and results are passed into softmax

```py
def softmax(np_array):
    return np.exp(x)/sum(np.exp(x))
```

if logit score gets extreme at the same time, softmax score gets extreme to 1 or to 0, steeper sigmoid curve 
vice versa, softmax score gets uniform if logit score shrinks
that is reason for __regularization__ terms, otherwise weights just go extreme, models go extreme certain, derivatives lost stability, no room for gradient descent

### one hot encoding for labels
one feature on to 1, off the others to 0

### cross-entropy loss, or called log loss  
* measure distance between softmax prediction and one-hot labels
cross entropy is also the information entropy equation
entropy H(p) = -sum(p * lg(p))
The entropy of a distribution P measures how many bits you need on average to encode data from P with the code that is optimal for P.
cross entropy H(p,q) = -sum(p * lg(q)), where p is real(goal) distribution and q is predicted distribution
entropy is cross entropy when prediction is correct
The cross-entropy of a distribution P with respect to a distribution Q measures how many bits you need on average to encode data from P with the code that is optimal for Q.
the difference is call relavent entropy, or KL divergence

* not symmetric
* D(pred, test), where pred and test for each sample data are vector with length of label numbers
* D should be low for right pred

### training loss
average D for all input points
need to minimize the loss for optmization
loss = 1/N * sum(D(S(weight * input + bias), label))

### optimization
__gradient descent!__
* minimize loss:
w' = w - step * d_loss/d_w where d_loss/d_w is steepest
* get closer to y_goal
d_loss/d_w = d_y * x_t
rule of thumb: GD takes 3 times more computing than the loss

### overfitting
start with big layers and overwork, then trim down
* early termination when validation test result drops

#### regularization: penaltize on dimensions
add large weight penaltizer term to loss function
L1 = sum(w_i), good for feature selection
weights likely to go to 1 or 0, rather than 0.5+0.5
L2 = sum(w_i^2), good for model training
weights likely to become homogeniously small (.5^2 + .5^2), rather than 1^2+0^2

#### dropout
new technique that just omit/terminate half perception activations, and scale the reminders by 2, to get same result expectation for evaluation
kinda ensemble sense, use votes and prevent dominent weights, 


### Stochastic GD
take a sample for each GD round, take smaller learning rate and much more iterations
overall much more faster, widely used in DL
* momentum to better guide to destination
* learning rate decay for faster at beginning and finer at finishing
* hyper-parameters: the dark side, for example faster learning rate may end at poorer prediction than a smaller learning rate
* if result not desired, lower the learning rate first
* ADAGRAD as a better alternative, it considers momentum from previous steps, to overcome local minima

### Back Propagation
Derivative of loss func flows back from result and passes through layers (all simple funcs: linear, ReLU) to produce new steps on weights for next iteration
* computationaly powerful
* chain rule:
d_g(f(x))/d_x  = d_g/d_f * d_f/d_x

### gradient vanishing
when derivative is tiny and no much descent
use other activations, like hyperbolic tangent, or ReLU

### hidden layers
width (size of single layer) is not much helpful
depth (# of layers) is helpful! both computational and performance, also catch natural patterns

## Tenserflow in python
```py
graph = tf.Graph()
with graph.as_default():
	inputs = tf.constant()
	randombatch = tf.placeholder(shape=batchsize)#for stochasticGD
	variables = tf.Variable() 
	weight = tf.Variable(truncated_normal([shape]))
	bias = tf.Variable(tf.zeros([length]))
	computations = tf.matmul(a,b)
	tf.nn.relu
	tf.nn.dropout
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2()) + tf.nn.l2_loss()
	optimizer = tf.train.model(learning_rate).minimize(loss)
	yourpred = tf.yourfunc(yourvar, *a)

with tf.session(graph=graph) as session:
	tf.global_variables_initializer().run() #one-time initializer
	feed_dict = {randombatch:yourbatch}
	for step in range(steps):
		_,pred = session.run([optimizer,yourpred], feed_dict=feed_dict)
```
```py
batch_size = 128
hidden_layer_size = 1024
beta = 0.01

graph_relu_l2_dropout_lrdecay = tf.Graph()
with graph_relu_l2_dropout_lrdecay.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  weight_0 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_layer_size]))
  bias_0 = tf.Variable(tf.zeros([hidden_layer_size]))
  weight_1 = tf.Variable(tf.truncated_normal([hidden_layer_size, num_labels]))
  bias_1 = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  pre_relu = tf.matmul(tf_train_dataset, weight_0) + bias_0
  # introduce dropout to hidden layer
  h_layer = tf.nn.dropout(tf.nn.relu(pre_relu), 0.5)
  # introduce dropout, again to output layer
  logits = tf.nn.dropout(tf.matmul(h_layer, weight_1) + bias_1, 0.5)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits)) + beta * (tf.nn.l2_loss(weight_0) + tf.nn.l2_loss(weight_1))
  
  # Optimizer.
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.5
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)    
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weight_0) + bias_0),weight_1)+bias_1)
  #test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weight_0) + bias_0),weight_1)+bias_1)
  
num_steps = 1001

with tf.Session(graph=graph_relu_l2_dropout_lrdecay) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 100 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
    
    #session.run([optimizer], feed_dict=feed_dict)
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
```

## keras
```py
# Imports
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

# Building the model
# Note that filling out the empty rank as "0", gave us an extra column, for "Rank 0" students.
# Thus, our input dimension is 7 instead of 6.
#del model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(7,)))
model.add(Dropout(.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.1))
model.add(Dense(2, activation='softmax'))

# Compiling and running the model
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=200, batch_size=100, verbose=0)
model.evaluate(X_train, y_train)
```