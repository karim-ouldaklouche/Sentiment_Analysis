import tensorflow as tf 

import matplotlib.pyplot as plt

from sa_preparate import SA_Preparate
from sa_bilstm_model import SA_BiLSTMModel

def main():
  print('Start Preparate Data')

  sa_Preparate = SA_Preparate()

  # The aclImdb_all.csv dataset comes from the data preparation done with the base line model (SVM and Logisitc Regression)  
  dataset_all = pd.read_csv('../data/aclImdb_all.csv',sep=',')

  print(dataset_all.shape)
  print(dataset_all.columns)

  # split the dataset 
  trainset, testset, valset = sa_Preparate.split_data_for_train_test_val_set(dataset_all, train_size = 0.8, test_val_size=0.5, 
                            stratify_column="polarity")                                

  # Apply the preprocess_text() function
  trainset['text_process'] = trainset['text'].apply(sa_Preparate.preprocess_text)
  print(trainset['text_process'].head())

  testset['text_process'] = testset['text'].apply(sa_Preparate.preprocess_text)
  print(testset['text_process'].head())

  valset['text_process'] = valset['text'].apply(sa_Preparate.preprocess_text)
  print(valset['text_process'].head())

  # The tokenization of the train, test and val set have to be done at the same time
  textset = pd.concat([trainset['text_process'], testset['text_process'], valset['text_process']])

  tensor, data_tokenizer = sa_Preparate.tokenize(textset, 'all')

  # Split the tensor for the train, test and valset
  tensor_train = tensor[0:trainset.shape[0],:]
  tensor_test = tensor[trainset.shape[0]:trainset.shape[0]+testset.shape[0],:]
  tensor_val = tensor[trainset.shape[0]+testset.shape[0]:,:]

  BUFFER_SIZE_TRAIN = len(tensor_train)
  BUFFER_SIZE_TEST = len(tensor_test)
  BUFFER_SIZE_VAL = len(tensor_val)

  BATCH_SIZE=64

  train_dataset = tf.data.Dataset.from_tensor_slices((tensor_train,tf.expand_dims(trainset['polarity'],1))).shuffle(BUFFER_SIZE_TRAIN)
  train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

  val_dataset = tf.data.Dataset.from_tensor_slices((tensor_val,tf.expand_dims(valset['polarity'],1))).shuffle(BUFFER_SIZE_VAL)
  val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

  test_dataset = tf.data.Dataset.from_tensor_slices((tensor_test,tf.expand_dims(testset['polarity'],1))).shuffle(BUFFER_SIZE_TEST)
  test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)

  # Size of the vocabulary
  vocab_size = len(data_tokenizer.word_index)+1
  print(vocab_size)
  
  steps_per_epoch = len(tensor_train) // BATCH_SIZE

  model = SA_BiLSTMModel(vocab_size=vocab_size, 
                input_length=sa_Preparate.get_max_tensor(tensor_train),
                output_dim=100,
                units = 64)

  optimizer = tf.keras.optimizers.Adam(0.001)

  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

  test_loss = tf.keras.metrics.Mean(name='test_loss')
  test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')

  @tf.function
  def train_step(inp, target):
    with tf.GradientTape() as tape:
      predictions = model(inp)
      loss = loss_object(target, predictions)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
    train_accuracy(target, predictions)

  @tf.function
  def test_step(inp, target):
    predictions = model(inp)
    loss = loss_object(target, predictions)

    test_loss(loss)
    test_accuracy(target, predictions)

  EPOCHS = 50

  train_loss_results = []
  train_accuracy_result = []
  test_loss_results = []
  test_accuracy_result = []

  for epoch in range(EPOCHS):

    for (batch, (x_train, y_train)) in enumerate(train_dataset):
      train_step(x_train, y_train)

    for (batch, (x_test, y_test)) in enumerate(test_dataset):
      test_step(x_test, y_test)

    template = 'Epoch {} Train : Loss {:.4f} Accuracy {:.4f} - Test : Loss {:.4f} Accuracy {:.4f}'

    print(template.format(epoch+1, 
                        train_loss.result(), train_accuracy.result(),
                        test_loss.result(), test_accuracy.result()))
  
    train_loss_results.append(train_loss.result().numpy())
    train_accuracy_result.append(train_accuracy.result().numpy())
    test_loss_results.append(test_loss.result().numpy())
    test_accuracy_result.append(test_accuracy.result().numpy())

    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    
  plt.plot(train_loss_results)
  plt.plot(test_loss_results)
  plt.show()
  
  plt.plot(train_accuracy_result)
  plt.plot(test_accuracy_result)
  plt.show()
  
if __name__ == '__name__':
  main()
