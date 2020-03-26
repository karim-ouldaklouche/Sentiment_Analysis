
import tensorflow as tf 

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
  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

  early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss')
    
  model.compile(optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy'])
  
  EPOCHS = 10

  model.fit(x=tensor_train, 
                    y=trainset['polarity'].values,
                    validation_data=(tensor_val, 
                    valset['polarity'].values),
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    callbacks=[early_stop])

  checkpoint_dir = './sa_bilstm_checkpoint'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, 
                                model=model)

  checkpoint.save(file_prefix=checkpoint_prefix)

  test_score = model.evaluate(tensor_test,
                        testset['polarity'].values,
                        batch_size=BATCH_SIZE)

  print(test_score)
  
if __name__ == '__name__':
  main()
  
