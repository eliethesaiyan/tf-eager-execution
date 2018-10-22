''' This program shows how to use a dense model as class and 
    how to override the model __call__ function of tf.keras.Model.  
    The introduced Dense layers acts as an encapsulation of action(tf.matmul(x, Kernel)+b)
'''

import tensorflow as tf

tf.enable_eager_execution()

X = tf.constant([[1.0, 2.0,3.0], [4.0,5.0,6.0]])
y = tf.constant([[10.0],[20.0]])


class Linear(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(units=1, kernel_initializer=tf.zeros_initializer(),
                     bias_initializer=tf.zeros_initializer())
    def call(self, input):
        output = self.dense(input)
        return output


model = Linear()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

for i in range(100):
    with tf.GradientTape() as tape:
         y_pred = model(X)
         loss = tf.reduce_mean(tf.square(y_pred - y))
    grads = tape.gradient(loss, model.variables)
    print(grads[0].numpy(), grads[1].numpy())
    optimizer.apply_gradients(grads_and_vars = zip(grads, model.variables))
print("Final parameter values")
print(model.variables)
