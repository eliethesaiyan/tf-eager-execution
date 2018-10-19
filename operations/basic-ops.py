''' This program tests simples operations( addition,multiplication)
    on constants  and matrices tensors (matmul)
'''

import tensorflow as tf

tf.enable_eager_execution()

a = tf.constant(1)
b = tf.constant(1)
c = tf.add(a, b) # equivalent of a + b

print(c)

A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])

C = tf.matmul(A, B)

print(C)


