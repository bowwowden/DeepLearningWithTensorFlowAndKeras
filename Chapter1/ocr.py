import tensorflow as tf 
import numpy as np 
from tensorflow import keras

# Network and training parameters
EPOCHS = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10 # NUMBER OF OUTPUTS
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2 # how much TRAIN is reserved for VALIDATION
DROPOUT = 0.3 # randomly drop some. form of regularization.

# loading mnist
mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# X_train is 60000 rows of 28x28 values
RESHAPED = 784

X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize inputs to be written in [0, 1]
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# One hot representation of the labels
Y_train = tf.keras.utils.to_categorical(Y_train, NB_CLASSES)
Y_test = tf.keras.utils.to_categorical(Y_test, NB_CLASSES)

# Build the model
model = tf.keras.models.Sequentual()
model.add(keras.layers.Dense(N_HIDDEN,
                        input_shape=(RESHAPED,),
                        name='dense_layer', activation='relu'))
# Dropout
model.add(keras.layers.Dropout(DROPOUT))
model.add(keras.layers.Dense(N_HIDDEN,
                        name='dense_layer_2', activation='relu'))
model.add(keras.layers.Dense(NB_CLASSES,
                        name='dense_layer_3', activation='softmax'))

# Summary of the model
model.summary()

# Compiling the model
model.compile(optimizer='SGD',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

# Training the model
model.fit(X_train, Y_train,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# Evaluating the model
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)




