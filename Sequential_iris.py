import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1],TRAIN_URL)
TEST_URL = "http://download.tensorflow.org/data/iris_training.csv"
test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

df_iris_train = pd.read_csv(train_path, header=0)
df_iris_test = pd.read_csv(test_path, header=0)

iris_train = np.array(df_iris_train)
iris_test = np.array(df_iris_test)

x_train = iris_train[:,:4]
y_train = iris_train[:,4]
x_test = iris_test[:,:4]
y_test = iris_test[:,4]

x_train = x_train - np.mean(x_train, axis=0)
x_test = x_test - np.mean(x_test, axis=0)

X_train = tf.cast(x_train,tf.float32)
X_test = tf.cast(x_test,tf.float32)
Y_train,Y_test = tf.cast(y_train,tf.int16),tf.cast(y_test,tf.int16)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(4,)))
model.add(tf.keras.layers.Dense(4,activation="relu"))
model.add(tf.keras.layers.Dense(3,activation="softmax"))

model.compile(optimizer=tf.keras.optimizers.SGD(lr = 0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
              ,metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# model.load_weights("Sequential_mnist_weights.h5")
model.fit(X_train,Y_train,batch_size=12,epochs=100,validation_split=0.2)

model.evaluate(X_test,Y_test,verbose=2)

model.save_weights(r"D:\python_project\tensorflow2.0\h5\Sequential_iris2_weights.h5")



