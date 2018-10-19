''' This programs perform a simple a regression on unity-base normalized(feature scaling) dataset
    and uses gradient descent to find the minimal(local) of the loss function
'''

import tensorflow as tf

tf.enable_eager_execution()



X_raw = np.array([2013, 2014, 2015, 2016,2017], dtype=np.float32)
y_raw = np.array([1200, 1400, 1500, 16500,17500], dtype=np.float32)

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min()) # Unity base normalization X = X-X.min/X.max-X.min
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min()) 


#gradient descendent 



