#A house has a base cost of 50k, and every additional bedroom adds a cost of 50k. This will make a 1-bedroom house cost
# 100k, a 2-bedroom house cost 150k etc.
#How would you create a neural network that learns this relationship so that it would predict a 7-bedroom house as
# costing close to 400k etc.

import tensorflow as tf
import numpy as np

# Function to create a 1 layer neural network with 1 unit
def model(x, y):
    tf_model = tf.keras.Sequential(tf.keras.layers.Dense(units=1, input_shape=[1]))
    tf_model.compile(optimizer='sgd', loss='mse')
    tf_model.summary()
    tf_model.fit(x, y, epochs=1000)
    return tf_model



if __name__ == "__main__":
    #Create a small dataset of bedroom as input and price as output
    bedroom = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    price = np.array([1, 1.5, 2, 2.5, 3, 3.5], dtype=float)

    house_model = model(bedroom, price)

    #Test
    new_bedroom_size = np.array([7, 8, 9, 10], dtype=float)
    predicted_price = house_model.predict(new_bedroom_size)

    for i in range(len(new_bedroom_size)):
        print("For {} bedrooms the price is {}".format(int(new_bedroom_size[i]), predicted_price[i]*100000))
    print("Done...")