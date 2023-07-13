import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.linalg import LinearOperatorFullMatrix
import numpy as np

class ContextualRNNCell(layers.Layer):
    '''
    n -- latent space dimension
    m -- input size
    '''
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.n = units
        self.state_size = [tf.TensorShape((self.n, self.n)), tf.TensorShape((self.n, self.n)), tf.TensorShape((self.n, 1))]
        self.ftype = tf.float32
        
    def build(self, input_shape):  

        self.m = input_shape[-1]
        n, m = self.n, self.m
        self.W = tf.Variable(tf.linalg.eye(*(n+m,)*2, batch_shape=(1,), dtype=self.ftype)) # trainable param

        # trainable dense layers
        self.f = layers.Dense(n, dtype=self.ftype)
        self.g = layers.Dense(m, dtype=self.ftype)
        
        # constant dense layers
        self.h = layers.Dense(m, dtype=self.ftype)
        self.h.trainable = False
        self.r = layers.Dense(m*m, dtype=self.ftype)
        self.r.trainable = False

        proj_h = np.hstack([np.identity(n), np.zeros((n, m))]) # projector on hidden space (first n coords)
        proj_y = np.hstack([np.zeros((m, n)), np.identity(m)]) # projector measured space (last m coords)
        self.proj_h = tf.convert_to_tensor(proj_h, dtype=self.ftype)
        self.proj_y = tf.convert_to_tensor(proj_y, dtype=self.ftype)
        
    def call(self, input, states):
        A_prev, J_prev, alpha_prev = states
        x = input
        
        x = tf.cast(x, self.ftype)
        # alpha = alpha_prev + tf.linalg.inv(A_prev)@tf.expand_dims(self.f(x), 2)
        alpha = alpha_prev + tf.expand_dims(self.f(x), 2)
        
        beta = tf.expand_dims(self.g(x), 2)
        # K    = self.h(x) # TODO dich kakaia-to
        S    = tf.reshape(self.r(x), (-1, self.m, self.m))
        B = S@tf.transpose(S, [0, 2, 1]) 

        U = tf.linalg.LinearOperatorBlockDiag([LinearOperatorFullMatrix(A_prev),
                                            LinearOperatorFullMatrix(B)]).to_dense()
        gamma = tf.concat([alpha, beta], 1)
        
        # transform the graph state by performing a Gaussian operation
        U = self.W@U@tf.transpose(self.W, [0, 2, 1])
        w_inv_t = tf.transpose(tf.linalg.inv(self.W), [0, 2, 1])
        gamma = w_inv_t@gamma
        # L = w_inv_t@L
        
        # read out the lattice and stabilizer phases
        y = self.proj_y@gamma     #  Пy@L
        
        # project out the measured register
        A = self.proj_h@U@tf.transpose(self.proj_h, [1, 0])
        alpha = self.proj_h@gamma
        return y[..., 0], (A, J_prev, alpha)    # TODO replace J_prev with counted new J
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        A = tf.eye(self.n, batch_shape=[batch_size], dtype=self.ftype)
        J = tf.zeros((batch_size, self.n, self.n), dtype=self.ftype)
        alpha = tf.zeros((batch_size, self.n, 1), dtype=self.ftype)
        
        return A, J, alpha



def get_rnn_layer(units, **kwargs):
    import os
    if os.environ["model"] == "crnn":
        return layers.RNN(
                      ContextualRNNCell(units),
                      **kwargs)
    elif os.environ["model"] == "lstm":
        return layers.LSTM(units,
                      **kwargs)
    

class Encoder(tf.keras.layers.Layer):
    def __init__(self, text_processor, units, input_dim):
        super(Encoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.units = units

        # The embedding layer converts tokens to vectors
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, input_dim,
                                                mask_zero=True)

        # The RNN layer processes those vectors sequentially.
        self.rnn = get_rnn_layer(units, return_state=True)

  

    def call(self, x):
        # 2. The embedding layer looks up the embedding vector for each token.
        x = self.embedding(x)

        # 3. The GRU processes the sequence of embeddings.
        x, *last_state = self.rnn(x)

        return last_state

    def convert_input(self, texts):
        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis]
        context = self.text_processor(texts).to_tensor()
        last_state = self(context)
        return last_state

class Decoder(tf.keras.layers.Layer):

  def __init__(self, text_processor, units):
    super(Decoder, self).__init__()
    self.text_processor = text_processor
    self.vocab_size = text_processor.vocabulary_size()
    self.word_to_id = tf.keras.layers.StringLookup(
        vocabulary=text_processor.get_vocabulary(),
        mask_token='', oov_token='[UNK]')
    self.id_to_word = tf.keras.layers.StringLookup(
        vocabulary=text_processor.get_vocabulary(),
        mask_token='', oov_token='[UNK]',
        invert=True)
    self.start_token = self.word_to_id('[START]')
    self.end_token = self.word_to_id('[END]')

    self.units = units


    # 1. The embedding layer converts token IDs to vectors
    self.embedding = tf.keras.layers.Embedding(self.vocab_size,
                                               units, mask_zero=True)

    # 2. The RNN keeps track of what's been generated so far.
    self.rnn = get_rnn_layer(units, return_sequences=True, return_state=True)

    
    # 4. This fully connected layer produces the logits for each
    # output token.
    self.output_layer = tf.keras.layers.Dense(self.vocab_size)

  def call(self,
          x,
          state,
          return_state=False):

    # 1. Lookup the embeddings
    x = self.embedding(x)

    # 2. Process the target sequence.
    y, *new_states = self.rnn(x, initial_state=state)
    if len(new_states) == 1:
        new_states = new_states[0]

    # Step 4. Generate logit predictions for the next token.
    logits = self.output_layer(y)

    if return_state:
      return logits, new_states
    else:
      return logits

  def get_initial_state(self, batch_size):
    start_tokens = tf.fill([batch_size, 1], self.start_token)
    done = tf.zeros([batch_size, 1], dtype=tf.bool)
    embedded = self.embedding(start_tokens)
    return start_tokens, done, self.rnn.get_initial_state(embedded)

  def tokens_to_text(self, tokens):
    words = self.id_to_word(tokens)
    result = tf.strings.reduce_join(words, axis=-1, separator=' ')
    result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
    result = tf.strings.regex_replace(result, ' *\[END\] *$', '')
    return result

  def get_next_token(self, last_token, done, state, temperature = 0.0):
    logits, state = self(
      last_token,
      state,
      return_state=True)

    if temperature == 0.0:
      next_token = tf.argmax(logits, axis=-1)
    else:
      logits = logits[:, -1, :]/temperature
      next_token = tf.random.categorical(logits, num_samples=1)

    # If a sequence produces an `end_token`, set it `done`
    done = done | (next_token == self.end_token)
    # Once a sequence is done it only produces 0-padding.
    next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)

    return next_token, done, state

class Translator(tf.keras.Model):

  def __init__(self, units,
               context_text_processor,
               target_text_processor,
               input_dim):
    super().__init__()
    # Build the encoder and decoder
    encoder = Encoder(context_text_processor, units, input_dim)
    decoder = Decoder(target_text_processor, units)

    self.encoder = encoder
    self.decoder = decoder

  def call(self, inputs):
    context, x = inputs
    
    encoder_state = self.encoder(context)
    logits = self.decoder(x, encoder_state)
    
    #TODO(b/250038731): remove this
    try:
      # Delete the keras mask, so keras doesn't scale the loss+accuracy.
      del logits._keras_mask
    except AttributeError:
      pass

    return logits

  def translate(self,
              texts, *,
              max_length=50,
              temperature=0.0):
    # Process the input texts
    last_state = self.encoder.convert_input(texts)
    if type(last_state) == list:
      batch_size = tf.shape(last_state[0])[0]
    else:
      batch_size = tf.shape(last_state)[0]


    # Setup the loop inputs
    tokens = []
    next_token, done, _ = self.decoder.get_initial_state(batch_size)
    state = last_state    
    for _ in range(max_length):
      # Generate the next token
      next_token, done, state = self.decoder.get_next_token(
          next_token, done,  state, temperature)

      # Collect the generated tokens
      tokens.append(next_token)
      
      if tf.executing_eagerly() and tf.reduce_all(done):
        break

    # Stack the lists of tokens and attention weights.
    tokens = tf.concat(tokens, axis=-1)   # t*[(batch 1)] -> (batch, t)

    result = self.decoder.tokens_to_text(tokens)
    return result