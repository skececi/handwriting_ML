import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy

(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[5000], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()

num_pixels = X_train.shape[1] * X_train.shape[2]
# num_pixels = 28 * 28 = 784
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_train = X_train / 255
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32') / 255
X_test = X_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

model = create_model()

model.fit(X_train, y_train, validation_data=(X_test, y_test))


def create_model():
    model = Sequential()

    model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



"""
y = softmax(Wx + b), where
y is vector of outcomes
W is weights (will learn)
x is input images
b is intercept (will learn)
################################
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
"""
