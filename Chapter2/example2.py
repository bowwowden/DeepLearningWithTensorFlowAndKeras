import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import tensorflow.keras as K
from tensorflow.keras.layers import Dense

# Generate a random data set

np.random.seed(0)
area = 2.5 * np.random.randn(100) + 25
price = 25 * area + 5 + np.random.randint(20, 50, size = len(area))
data = np.array([area, price])
data = pd.DataFrame(data = data.T, columns=['area', 'price'])
plt.scatter(data['area'], data['price'])
plt.show()

# Normalize 

data = (data - data.min()) / (data.max() - data.min())

model = K.Sequential([
                    Dense(1, input_shape = [1,], activation=None)
])
model.summary()


# Define loss

model.compile(loss='mean_squared_error', optimizer='sgd')


# Train

model.fit(x=data['area'], y=data['price'], epochs=100, 
          batch_size=32, verbose=1, validation_split=0.2)


# Get prediction

y_pred = model.predict(data['area'])

# Plot graph

plt.plot(data['area'], y_pred, color='red', label="Predicted Price")
plt.scatter(data['area'], data['price'], label="Training Data")
plt.xlabel("Area")
plt.ylabel("Price")
plt.legend()
plt.show()



