import tensorflow as tf
from tensorflow import keras
from keras import Sequential, Input
from keras.layers import Embedding, Concatenate, LayerNormalization
# from tensorflow.keras import layers
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
import tensorflow.keras.backend as K
from co_occurrence import load_cooccurrence_matrix

class BatchGenerator(tf.keras.utils.Sequence):

  def __init__(self, 
               matrix,
               dim=4,
               batch_size=1024, 
               seed=42,
               shuffle=True):
    # Call on Sequence object to fit the dataset, i.e. co-matrix
    super(BatchGenerator).__init__()

    # Set additional informations
    # list of word and context ids of the partition
    self.matrix = matrix
    self.dim = dim
    self.rng = np.random.default_rng(seed)
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.set_pairs = set()

    # the method on_epoch_end should be triggered once at the very beginning as well as
    # at the end of each epoch
    self.on_epoch_end()
  
  def on_epoch_end(self):
    """
    Update indexes after each epoch
    """
    self.rng.shuffle(self.matrix, axis=0)
    self.__build_batches()

  def __build_batches(self):

    padsize = self.batch_size - (len(self.matrix) % self.batch_size)
    padding = self.rng.choice(self.matrix, padsize, axis=0)
    matrix = np.vstack((self.matrix, padding))
    self.batches = np.split(matrix, len(matrix) / self.batch_size)

  def __len__(self):
    """
    Denotes the number of batches per epoch.
    """
    # return len(self.batches)
    return int(np.floor(self.matrix.shape[0] / self.batch_size))

  def __getitem__(self, index):
    """
    Generate one batch of data.

    When the batch corresponding a given index is called, the generator executes the
    __getitem__ method to generate it.
    """
    # Generate the indexes for the 'index' batch
    batch = self.batches[index]

    # Batch indexes that identify male, female and neutral words (both for the target and the context)
    # e.g., male_target_indexes = [12, 27] this means that batch[12] and batch[27] both
    # contain target male word.
    male_target_indexes = batch[np.where(batch[:, 3] == 1), 0][0].astype(np.int32)
    male_context_indexes = batch[np.where(batch[:, 3] == 1), 1][0].astype(np.int32)
    female_target_indexes = batch[np.where(batch[:, 3] == 2), 0][0].astype(np.int32)
    female_context_indexes = batch[np.where(batch[:, 3] == 2), 1][0].astype(np.int32)
    neutral_target_indexes = batch[np.where(batch[:, 3] == 0), 0][0].astype(np.int32)
    neutral_context_indexes = batch[np.where(batch[:, 3] == 0), 1][0].astype(np.int32)
    
    # X = (batch_size, target_idx, context_idx, co_occ)
    X = (batch[:, 0], batch[:, 1])

    # Our Y are the probabilities that a word appears in the window of a target word
    y = tf.reshape(batch[:, 2], (batch[:, 2].shape[0], 1))
    info = [male_target_indexes, 
            male_context_indexes, 
            female_target_indexes, 
            female_context_indexes, 
            neutral_target_indexes,
            neutral_context_indexes]

    return X, y, info

class CustomEmbedding(Embedding):
  def __init__(self,
               gender_dim=1,
               **kwargs):
    super(CustomEmbedding, self).__init__(**kwargs)

    self.gender_dim = gender_dim

  @tf_utils.shape_type_conversion
  def build(self, input_shape=None):
    """
    In the embedding model by Zhao et al. the vector consists of two parts
      w = [w_a, w_g],
    where w_g of size (gender_dim, 1) denotes the gender information.
    """
    self.embeddings_a = self.add_weight(
        shape=(self.input_dim, self.output_dim - self.gender_dim),
        initializer=self.embeddings_initializer,
        name='embeddings_a',
        regularizer=self.embeddings_regularizer,
        constraint=self.embeddings_constraint,
        experimental_autocast=False)
    
    self.embeddings_g = self.add_weight(
        shape=(self.input_dim, self.gender_dim),
        initializer=self.embeddings_initializer,
        name='embeddings_g',
        regularizer=self.embeddings_regularizer,
        constraint=self.embeddings_constraint,
        experimental_autocast=False)
    self.built = True
  
  def call(self, inputs):
    dtype = backend.dtype(inputs)
    if dtype != 'int32' and dtype != 'int64':
      inputs = math_ops.cast(inputs, 'int32')
    out_a = embedding_ops.embedding_lookup_v2(self.embeddings_a, inputs)
    out_g = embedding_ops.embedding_lookup_v2(self.embeddings_g, inputs)
    if self._dtype_policy.compute_dtype != self._dtype_policy.variable_dtype:
      # Instead of casting the variable as in most layers, cast the output, as
      # this is mathematically equivalent but is faster.
      out_a = math_ops.cast(out_a, self._dtype_policy.compute_dtype)
      out_g = math_ops.cast(out_g, self._dtype_policy.compute_dtype)
    return K.concatenate([out_a, out_g])

# @tf.autograph.experimental.do_not_convert
class Glove(tf.keras.Model):
  def __init__(self, 
               vocab_size, 
               embedding_size, 
               gender_dim=1, 
               alpha=3/4, 
               x_max=100):
    super(Glove, self).__init__()

    self.embedding_size = embedding_size

    # Loss's params
    self.alpha = alpha
    self.x_max = x_max
    self.lambda_d = tf.constant(0.8)
    self.lambda_e = tf.constant(0.8)
    
    self.male_idxs = male_idxs
    self.female_idxs = female_idxs
    self.n_gender_indexes = 2 * len(self.male_idxs) + 2 * len(self.female_idxs)
    """
    Training variables:
      0: target_word_embedding_a,
      1: target_word_emebdding_g,
      2: target_bias_embedding
      3: ...
    """
    self.target_embedding = CustomEmbedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        gender_dim=gender_dim,
        input_length=1,
        # embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0),
        name="target_embedding"
    )
    self.target_bias = Embedding(
        input_dim=vocab_size,
        output_dim=1,
        input_length=1,
        # embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0),
        name="target_bias"
    )
    self.context_embedding = CustomEmbedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        gender_dim=gender_dim,
        input_length=1,
        # embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0),
        name="context_embedding"
    )
    self.context_bias = Embedding(
        input_dim=vocab_size,
        output_dim=1,
        input_length=1,
        # embeddings_initializer=tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0),
        name="context_bias"
    )
    self.ls_tracker = keras.metrics.Mean(name='loss')

  def call(self, inputs):
    """
    This method computes the model's forward pass.
    """
      
    # Extract tokens from the input given by the generator
    target_idx, context_idx = inputs

    target_emb = self.target_embedding(target_idx)
    target_bias = self.target_bias(target_idx)
    context_emb = self.context_embedding(context_idx)
    context_bias = self.context_bias(context_idx)

    # Alternatively a softmax to compute jg
    dots = tf.keras.layers.Dot(axes=-1)([target_emb, context_emb])
    y_pred = tf.keras.layers.Add()([dots, target_bias, context_bias])
    # y_pred = tf.keras.layers.Dense(self.en)

    return y_pred

  def __compute_vg(self):
    """
    Computes the gender direction.
    """
    male_target = K.sum(tf.gather(self.trainable_variables[0], self.male_idxs), axis=0)
    male_context = K.sum(tf.gather(self.trainable_variables[3], self.male_idxs), axis=0)

    female_target = K.sum(tf.gather(self.trainable_variables[0], self.female_idxs), axis=0)
    female_context = K.sum(tf.gather(self.trainable_variables[3], self.female_idxs), axis=0)

    vg = male_target + male_context
    vg -= female_target + female_context
    vg = vg / self.n_gender_indexes
    vg = tf.reshape(vg, (self.embedding_size-1, 1))
    return vg

  def __compute_jd(self, 
                   male_target_indexes,
                   male_context_indexes,
                   female_target_indexes,
                   female_context_indexes):
    """
    Computes the component of the loss that minimizes the negative distances between words
    in the male and female group.
    """
    male_target = K.sum(tf.gather(self.trainable_variables[1], male_target_indexes), axis=0)
    male_context = K.sum(tf.gather(self.trainable_variables[4], male_context_indexes), axis=0)

    female_target = K.sum(tf.gather(self.trainable_variables[1], female_target_indexes), axis=0)
    female_context = K.sum(tf.gather(self.trainable_variables[4], female_context_indexes), axis=0)

    loss = male_target + male_context
    loss -= female_target + female_context
    return -tf.norm(loss, ord=1)
  
  def __compute_je(self, vt, neutral_target_indexes, neutral_context_indexes):
    """
    Retains the w_a embeddings to be retained in the null space of the gender direction v.
    """
    neutral_target = tf.gather(self.trainable_variables[0], neutral_target_indexes)
    neutral_context = tf.gather(self.trainable_variables[3], neutral_context_indexes)
    
    neutral_target = K.square(tf.matmul(neutral_target, vt))
    neutral_context = K.square(tf.matmul(neutral_context, vt))
    loss = K.sum(neutral_target) + K.sum(neutral_context)
    return loss

  def train_step(self, data):
    """
    Defines the logic for one training step. Unpacks the data. Its structure depends
    on the model and on what you pass to fit().
    """

    # Unpack data from BatchGenerator
    X, y, info = data
    male_target_indexes, male_context_indexes, female_target_indexes, female_context_indexes, neutral_target_indexes, neutral_context_indexes = info

    tr_vars = self.trainable_variables

    with tf.GradientTape() as tape:
     
      # model.call()
      y_pred = self(X, training=True)
      
      # Forward pass
      loss_jd = self.__compute_jd(male_target_indexes,
                                  male_context_indexes,
                                  female_target_indexes,
                                  female_context_indexes)
      
      vg = self.__compute_vg()
      
      loss_je = self.__compute_je(vg, neutral_target_indexes, neutral_context_indexes)
      
      loss = K.sum(tf.pow(K.clip(y / self.x_max, 0.0, 1.0), self.alpha) * K.square(y_pred - K.log(y)))
      loss += self.lambda_d * loss_jd
      loss += self.lambda_e * loss_je

    # Obtain gradients wrt to the training variables, we are interested in tr_vars[0,1,3,4]
    grads = tape.gradient(loss, tr_vars)

    # Update the network weights: gradient descent
    self.optimizer.apply_gradients(zip(grads, tr_vars))
    self.ls_tracker.update_state(loss)
    return {'loss': self.ls_tracker.result()}
  
  @property
  def metrics(self):
    return [self.ls_tracker]

def create_model(vocab_size= 100000, embedding_size=300):
  return Glove(vocab_size, embedding_size)

class MinMaxConstraintEmbeddings(tf.keras.callbacks.Callback):
  def __init__(self, min=-1., max=1., epsilon=1e-12):
    super(MinMaxConstraintEmbeddings, self).__init__()
    self.min = min
    self.max = max
    self.epsilon = epsilon

  def on_batch_begin(self, *args, **kwargs):
    W = self.model.trainable_weights
    target_word = tf.concat(W[0:2], axis=-1)    #[w_a, w_g]
    target_bias = W[2]
    context_word = tf.concat(W[3:5], axis=-1)   #[c_a, c_g]
    context_bias = W[5]

    for i, w in enumerate([target_word, target_bias, context_word, context_bias]):
      std = tf.math.divide(tf.subtract(w, tf.reduce_min(w)),
                           tf.math.maximum(tf.subtract(tf.reduce_max(w), tf.reduce_min(w)),
                                           self.epsilon))
      x = tf.add(tf.math.multiply(std, 
                     tf.subtract(tf.constant(self.max),
                                 tf.constant(self.min))),
                tf.constant(self.min))
      self.model.trainable_weights[i] = x

if __name__ == "__main__":
    training_generator = BatchGenerator(load_cooccurrence_matrix())
    model = create_model()
    model.compile('adam')
    history = model.fit(training_generator, epochs=5, callbacks=[MinMaxConstraintEmbeddings()])