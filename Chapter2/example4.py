import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import tensorflow.keras as K
from tensorflow.keras.layers import Dense, Flatten

# Logistic regression
((train_data, train_labels), (test_data, test_labels)) = tf.keras.datasets.mnist.load_data()

train_data = train_data/np.float32(255)
train_labels = train_labels.astype(np.int32)
test_data = test_data/np.float32(255)
test_labels = test_labels.astype(np.int32)

model = K.Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(10, activation='sigmoid')
])
model.summary()


model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(
                                from_logits=True), metrics=['accuracy'])

history = model.fit(x=train_data, y=train_labels, epochs=50, verbose=1, validation_split=0.2)

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("Pred {} Conf: {:2.0f}% True ({})".format(predicted_label,
                                                        100*np.max(predictions_array),
                                                        true_label),
                                                        color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, 
    color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


predictions = model.predict(test_data)
i = 56
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_data)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()