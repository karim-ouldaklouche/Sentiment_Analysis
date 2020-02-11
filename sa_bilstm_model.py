
import tensorflow as tf

class SentimentAnalysisBiLSTMModel(tf.keras.Model):
  """
  The class implement an architecture of recurrent neural network with
  - one embedding layer
  - two bi-directionnal LSTM layers
  - one dense layer
  """
  def __init__(self, vocab_size, input_length, output_dim, units, dropout=0.2):
      """
      Arguments : 
      vocab_size : Integer
      input_length : Integer
      output_dim : Integer
      units : Integer
      dropout : Float
      """
      super(SentimentAnalysisBiLSTMModel, self).__init__()

      self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                              input_length=input_length,
                                              output_dim=output_dim)

      self.dropout = tf.keras.layers.Dropout(dropout)
      self.rnn1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True), 
                                              merge_mode='concat')
      self.rnn2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units), merge_mode='concat')

      self.fc = tf.keras.layers.Dense(1, activation='sigmoid')

  def call(self, inputs):
      """
      Arguments : 
      inputs : List

      Returns
      output : List 
      """
      output = self.embedding(inputs)
      output = self.dropout(output)
      output = self.rnn1(output)
      output = self.rnn2(output)
      output = self.fc(output)

      return output
