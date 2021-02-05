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
Y_train = tf.one_hot(tf.constant(y_train,dtype=tf.int32),3)

X_test = tf.cast(x_test,tf.float32)
Y_test = tf.one_hot(tf.constant(y_test,dtype=tf.int32),3)

learn_rate = 0.5
iter = 50
display_step = 10

np.random.seed(612)
W1 = tf.Variable(np.random.randn(4,16),dtype=tf.float32)
B1 = tf.Variable(tf.zeros([16]),dtype=tf.float32)
W2 = tf.Variable(np.random.randn(16,3),dtype=tf.float32)
B2 = tf.Variable(tf.zeros([3]),dtype=tf.float32)

acc_train = []
acc_test = []
cce_train = []
cce_test = []

for i in range(iter+1):
    with tf.GradientTape() as tape:
        Hidden_train = tf.nn.relu(tf.matmul(X_train,W1)+B1)
        PRED_train = tf.nn.softmax(tf.matmul(Hidden_train,W2)+B2)
        Loss_train = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=Y_train,y_pred=PRED_train))

        Hidden_test = tf.nn.relu(tf.matmul(X_test, W1) + B1)
        PRED_test = tf.nn.softmax(tf.matmul(Hidden_test, W2) + B2)
        Loss_test = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=Y_test, y_pred=PRED_test))

    accuracy_train = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(PRED_train.numpy(),axis=1),y_train),tf.float32))
    accuracy_test = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(PRED_test.numpy(), axis=1), y_test), tf.float32))

    acc_train.append(accuracy_train)
    acc_test.append(accuracy_test)
    cce_train.append(Loss_train)
    cce_test.append(Loss_test)

    grads = tape.gradient(Loss_train,[W1,B1,W2,B2])
    W1.assign_sub(learn_rate * grads[0])
    B1.assign_sub(learn_rate * grads[1])
    W2.assign_sub(learn_rate * grads[2])
    B2.assign_sub(learn_rate * grads[3])

    if i % display_step == 0:
        print("i:%i,TrainLoss:%f,testacc:%f,testloss:%f"%(i,accuracy_train,accuracy_test,Loss_test))


plt.figure(figsize=(10,3))

plt.subplot(121)
plt.plot(cce_train,color = 'b',label = 'train')
plt.plot(cce_test,color = 'r',label = 'test')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()

plt.subplot(122)
plt.plot(acc_train,color = 'b',label = 'train')
plt.plot(acc_test,color = 'r',label = 'test')
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.legend()

plt.show()


