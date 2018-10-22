''' This programs perform a simple a regression on unity-base normalized(feature scaling) dataset
    and uses gradient descent to find the minimal(local) of the loss function
'''

import tensorflow as tf
import numpy as np

tf.enable_eager_execution()



X_raw = np.array([2013, 2014, 2015, 2016,2017], dtype=np.float32)
y_raw = np.array([1200, 1400, 1500, 16500,17500], dtype=np.float32)

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min()) # Unity base normalization X = X-X.min/X.max-X.min
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min()) 


#gradient descendent 
a, b = 0, 0

num_epoch = 10000
learning_rate = 1e-3

# calculate gradients by hand in numpy
for e in range(num_epoch):
    y_pred = a * X + b
    grad_a, grad_b = (y_pred - y).dot(X), (y_pred - y).sum()
    a, b = a - learning_rate * grad_a, b - learning_rate * grad_b
    print("a = "+str(a),"b = "+str(b))


# using  the Gradient Descedent Optimizer from tensorflow
X = tf.constant(X)
y = tf.constant(y)

c  = tf.get_variable('c', dtype=tf.float32, shape =[], initializer = tf.zeros_initializer)
d  = tf.get_variable('d', dtype=tf.float32, shape =[], initializer = tf.zeros_initializer)
variables =[c,d]
optimizer = tf.train.GradientDescentOptimizer(learning_rate= learning_rate)

for e in range(num_epoch):
    # Use tape to record gradient info of the loss
    with tf.GradientTape() as tape:
        y_pred = c * X + d 
        loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
    grads = tape.gradient(loss,variables)
    print(grads[0].numpy(), grads[1].numpy())
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))



