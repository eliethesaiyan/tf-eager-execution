import tensorflow as tf

'''This programs verify if eager execution
 enabled and working in your tensorflow 
 by making simple matrix multiplication'''


tf.enable_eager_execution()

A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])

C = tf.matmul(A,B)
 
print(C)
