weight sharing: statistical invariance
feature map
pros: 
more information from feature map, without parameters!
take sparsely connected layers, not fully-connected ones for mlpnn, 
similarly, locally connected layer  to pick up regional patterns, and save computations
that is __convolution!__ 

###convolution
make feature map deeper
patch/kernal
usually a square matrix same size as patch and outputs product of two matrix
transfer wide feature map to deep map
stride: moving steps of kernal
padding: edge of convolution, valid or same
* 1x1 convolution
	essentially a small nn, and computation-friendly

### Fully Connected layer
 is a standard, non convolutional layer, where all inputs are connected to all output neurons. This is also referred to as a "dense" layer, and is what we used in the previous two lessons.

### Pooling
make feature map smaller
: "combine" or "reduce" a feature map's patch's information, to reduce map size, like max_pooling(input_map), average_pooling, 

###typical cnn
[conv + pool]*  + [fully_conn]*

### augmentation
we want the model to learn invariant representation of image
cnn have some built-in translation invariance
for rotation invariance and scale invariance, we augment the data, by randomly rotate, scale and translate the pics

## inception module
for one conv level, use multiple computations and concat them:
[1x1 conv, 3x3 conv, ...., average_pool]

### transfer learning
remove last (few) fully-connect layers or conv layers, which are tailered to the dataset
add new layers to new dataset to the remaining network, which is already trained to be good at catching general patterns from pics
train the whole model
up to big/small dataset and close/far objects, replacement of layer and updates on trained weights may differ
the last layer's output (usually after a pooling layer, a deep output) of transferred model is called __bottleneck__ features

### keras
[famous cnn arch, easy import](https://keras.io/applications/)


```py
graph_conv = tf.Graph()

with graph_conv.as_default():

  # Input data with channels number
  tf_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))

  # Variables of patch and fully-connected reg
  patch_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, output_depth], stddev=0.1))
  patch_biases = tf.Variable(tf.zeros([output_depth]))

  reg_weights = tf.Variable(tf.truncated_normal(
      [image_size // 4 * image_size // 4 * depth, output_size], stddev=0.1))
  	  #divide 4 is because stride is 2,2 for feature map, conv'ed twice
  reg_biases = tf.Variable(tf.constant(1.0, shape=[output_size]))


  # Computations
  def model(data):
    # list here is stride in 4-d patch, layer1_weights, order by dimension declaration
    conv = tf.nn.conv2d(data, patch_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    shape = hidden.get_shape().as_list()
    #shape=[10000, 7, 7, 16], then reshape to 1-d array for each single  perceptron, make it to vanila multi-layer nn
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    output = tf.nn.relu(tf.matmul(reshape, reg_weights) + reg_biases)
	return output  
  #vanilla mlnn
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean()
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
```
```py
import keras
from keras.datasets import cifar10
# load the pre-shuffled train and test data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# standardize input [0,255] --> [0,1]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
# one-hot encode the labels
num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# break training set into training and validation sets
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.summary()
# compile the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# train the model
from keras.callbacks import ModelCheckpoint   
checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)
hist = model.fit(x_train, y_train, batch_size=32, epochs=100,
          validation_data=(x_valid, y_valid), callbacks=[checkpointer], verbose=2, shuffle=True)
# load the weights that yielded the best validation accuracy
model.load_weights('model.weights.best.hdf5')
# evaluate and print test accuracy
score = model.evaluate(x_test, y_test, verbose=0)
# get predictions on the test set
y_hat = model.predict(x_test)
```
```py
from keras.preprocessing.image import ImageDataGenerator

# create and configure augmented image generator
datagen_train = ImageDataGenerator(
    width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)
    height_shift_range=0.1,  # randomly shift images vertically (10% of total height)
    horizontal_flip=True) # randomly flip images horizontally

model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=epochs, verbose=2, callbacks=[checkpointer])
```