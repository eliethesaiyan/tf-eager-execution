''' This programs does a simple differentiation of simple  X^2 function
    and returns the slope(gradient) at a given value 
    X^2' = 2x, at x equal to 3, y=3^2 and y_grad=2*3

'''

import tensorflow as tf

tf.enable_eager_execution()


x = tf.get_variable('x', shape=[1], initializer= tf.constant_initializer(3.))

with tf.GradientTape() as tape: # all steps under with are recorded and can be further accessed usin tape" 
    y = tf.square(x)

y_grad = tape.gradient(y, x) # differentiate y with respect to x

# prints the value of y and its gradient at x = 3
print([y.numpy(), y_grad.numpy()])

