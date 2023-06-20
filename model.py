import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.linalg import LinearOperatorFullMatrix, LinearOperatorBlockDiag

tf.keras.utils.set_random_seed(42)

class CRNN(Model):
  def __init__(self, n, m):
    '''
    n -- latent space dimension
    m -- input size
    '''
    super().__init__()
    self.ftype = tf.float64
    self.n, self.m = n, m
    self.W = tf.Variable(tf.random.uniform((1,) + (n+m,)*2, dtype=self.ftype)) # trainable param

    # trainable dense layers
    self.f = Dense(n, dtype=self.ftype)
    self.g = Dense(m, dtype=self.ftype)
    
    # constant dense layers
    self.h = tf.keras.layers.Dense(m, dtype=self.ftype)
    self.h.trainable = False
    self.r = Dense(m*m, dtype=self.ftype)
    self.r.trainable = False

    proj_h = np.hstack([np.identity(n), np.zeros((n, m))]) # projector on hidden space (first n coords)
    proj_y = np.hstack([np.zeros((m, n)), np.identity(m)]) # projector measured space (last m coords)
    self.proj_h = tf.convert_to_tensor(proj_h, dtype=self.ftype)
    self.proj_y = tf.convert_to_tensor(proj_y, dtype=self.ftype)
    
  def call(self, A_prev, J_prev, alpha_prev, x):
    x = tf.cast(x, self.ftype)
    alpha = alpha_prev + A_prev@tf.expand_dims(self.f(x), 2) # todo replace A_prev with  tf.linalg.inv(A_prev)
    
    beta = tf.expand_dims(self.g(x), 2)
    # K    = self.h(x) # TODO dich kakaia-to
    S    = tf.reshape(self.r(x), (-1, self.m, self.m))
    B = S@tf.transpose(S, [0, 2, 1]) 

    U = tf.linalg.LinearOperatorBlockDiag([LinearOperatorFullMatrix(A_prev),
                                          LinearOperatorFullMatrix(B)]).to_dense()
    gamma = tf.concat([alpha, beta], 1)
    
    # transform the graph state by performing a Gaussian operation
    U = self.W@U@tf.transpose(self.W, [0, 2, 1])
    # w_inv_t = tf.transpose(tf.linalg.inv(self.W), [0, 2, 1])
    gamma = self.W@gamma   # todo replace w with w_inv
    # L = w_inv_t@L
    
    # read out the lattice and stabilizer phases
    y = self.proj_y@gamma     #  Пy@L
    
    # project out the measured register
    A = self.proj_h@U@tf.transpose(self.proj_h, [1, 0])
    alpha = self.proj_h@gamma
    return A, J_prev, alpha, y    # TODO replace J_prev with counted new J

  def process_img(self, image):
    batch_size = image.shape[0]
    A = tf.eye(self.n, batch_shape=[batch_size], dtype=self.ftype)
    J = tf.zeros((batch_size, self.n, self.n), dtype=self.ftype)
    alpha = tf.zeros((batch_size, self.n, 1), dtype=self.ftype)
    flat_img = tf.cast(tf.reshape(image, (batch_size, -1)), self.ftype)
    for i in range(0, flat_img.shape[1]+1-self.m, self.m):
      A, J, alpha, y = self.call(A, J, alpha, flat_img[:, i:i+self.m])
    return y[:, 0, 0]