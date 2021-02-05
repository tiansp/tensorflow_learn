import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(train_x,train_y),(test_x,test_y) = mnist.load_data()

# X_train = train_x.reshape((60000,28*28))
# X_test = test_x.reshape((10000,28*28))

X_train , X_test = tf.cast(train_x/255.0,tf.float32),tf.cast(test_x/255.0,tf.float32)
Y_train,Y_test = tf.cast(train_y,tf.int16),tf.cast(test_y,tf.int16)
print(X_train.shape,Y_train.shape)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation="relu"))
model.add(tf.keras.layers.Dense(10,activation="softmax"))

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['sparse_categorical_accuracy'])

# model.load_weights("Sequential_mnist_weights.h5")
model.fit(X_train,Y_train,batch_size=64,epochs=5,validation_split=0.2)

model.evaluate(X_test,Y_test,verbose=2)

model.save_weights("D:\python_project\tensorflow2.0\h5\Sequential_iris2_weights.h5")

for i in range(4):

    num = np.random.randint(1,10000)


    plt.subplot(1,4,i+1)
    plt.axis('off')
    plt.imshow(test_x[num],cmap='gray')
    plt.title("y="+str(test_y[num]))

plt.show()

