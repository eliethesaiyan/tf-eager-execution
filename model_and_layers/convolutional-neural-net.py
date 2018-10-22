''' This programs implements a simple multi-layer(2) perceptron
    and fetch minist dataset using a dataloader class to feed to the model
    and performs a simple evaluation(accuracy) on testing dataset
'''

import tensorflow as tf
import numpy as np


tf.enable_eager_execution()

class DataLoader():

    def __init__(self):
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        # train dataset
        self.train_data = mnist.train.images
        self.train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

        self.eval_data = mnist.test.images
        self.eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    def get_batch(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_data)[0],batch_size)
        return self.train_data[index, :], self.train_labels[index]


class CNN(tf.keras.Model):

    def __init__(self):
        
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[5,5],
            padding ="same",
            activation= tf.nn.relu
            )

        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=2)

        self.conv2 = tf.keras.layers.Conv2D(
            filters = 64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu
            )
        
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
        self.dense1 = tf.keras.layers.Dense(units=1024,activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        inputs = tf.reshape(inputs,[-1, 28,28,1]) 
        x = self.conv1(inputs) # [batch_SIZE,28,28,32] 
        x = self.pool1(x)      # [batch_size,14,14,32] 
        x = self.conv2(x) # [batch_size,14,14,64] 
        x = self.pool2(x)      # [batch_size,7,7,64]
        x = self.flatten(x)    # [batch_size,7*7*64]
        x = self.dense1(x)     # [batch_size,1024]
        x = self.dense2(x)     # [batch_size,10]

        return x

    def predict(self,inputs):
        logits = self(inputs)
        return tf.argmax(logits , axis=-1)


num_batches = 1000
batch_size = 50
learning_rate = 0.001

model = CNN()
data_loader = DataLoader()
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_index)
    with tf.GradientTape() as tape:
        y_logit_pred = model(tf.convert_to_tensor(X))
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_logit_pred)
        print("batch %d: loss %f"% (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars =zip(grads, model.variables))

num_eval_samples = np.shape(data_loader.eval_labels)[0]
y_pred = model.predict(data_loader.eval_data).numpy()
print("test accuracy : %f" % (sum(y_pred == data_loader.eval_labels)/num_eval_samples))

    


