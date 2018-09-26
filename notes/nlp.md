issues
* rare words
* different words sharing same meaning

## word embedding
turn words to vectors in a multi-D space, the vector is called embedding
similar words are spatially closer, by cossin metric not l2 function (direction, not distance)
or just normalize the embedding before comparing
called __word2vec__ models

### skip-gram
* use the current word of to predict its neighbors, a "nearby word" model
turn one word to a embedding vector, by the trained model weights
randomly initialized 
during training, use embeding to predict its context word (one at a time, one to one prediction model) in the window (word around it in corpus) via logistic model
word>embedding(narrow)>log-reg>softmax>crossentropy>target words(very wide)
we’re not actually going to use that neural network for the task we trained it on! Instead, the goal is actually just to learn the weights of the hidden layer–we’ll see that these weights are actually the “word vectors” that we’re trying to learn
* sampled softmax on output
randomly pick target words
no cost in performance
finally we have a word to vector where simialr words are closer, and can do analogy prediction

#### optimization
* subsampling: drop high-freq words like the
they count for large amont of data, and also somewhat meaninglett
* negative sampling: only update limited number of 0/false output and associated weights in each round 
The google paper says that selecting 5–20 words works well for smaller datasets, and you can get away with only 2–5 words for large datasets.

### Continuous Bag Of Words, CBOW
* uses each of these contexts to predict the current word, multi to one prediction model 

### analogy
both sementic and syntactic

### tSNE visualization
t-distributed Stochastic Neighbor Embedding
transform embedding space to 2d plot, perserving neighbor relations

## recurrent nn
map variable length input to fixed length output
think of a time-sequenced input series x_i
take into history by a recurrent model r_i = r(y_i-1)
use one universal model w for input, add recurrent, to get y_i = w(x_i) + r(y_i-1)

### optimization
when backprop, compute till the beginning or as much as we can afford
all adjustment will apply to one same w <- bad for gradient descent since unstability
to solve unstability, aka. gradient exploding/vanishing

#### gradient exploding/vanishing
* gradient clipping for exploding: set the max limit
vanishing will make model momoryless: only remember the recent events
* Long short-term memory, LSTM
substitute w from plain nn to "memory cell" nn

#### LSTM
method of write, read, and forget
control methods by factors from [0,1], (control gate), a countinuous and derivable func will work for back-prop
each gate is one log reg with their own weights, they are trained to remember or forget things optimally
regularization: l2 ok, dropout ok but only apply them on input/output, not on recurrent connections

### prediction
predict next word/letter's prob distribution, and sample from it
beam search: make more than one-step pred, and compare joint distribution, to prevent one-step
beam search can predict variable length output for variable steps, by re-input the output as memory, even start from fixed input
__how google translate works!__
but also sound recog,

## Implementation in tf
TODO 
just too time consuming