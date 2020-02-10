import numpy as np
import pandas as pd 

import time

from sklearn.utils import shuffle

import spacy
import copy
from sklearn.model_selection import train_test_split
import time
import pickle

import tqdm

import glob

import unicodedata
import re
import os
import io

class SA_Preparate:

    def parse_directory_for_files(self, path):
        """
        Parameters : 
        path : String

        Returns
        List 
        """
        print('Start parse directory for files : {}'.format(path))
        start = time.time()

        files = [f for f in glob.glob(path + '**.txt', recursive=True)]
        print('End parse in {}'.format(time.time()-start))

        return files

    def read_files_to_dataframe(self, files, polarity, type):
        """
        Parameters : 
        files : List 
        polarity : String
        type : String

        Returns
        dataframe : DataFrame
        """
        start = time.time()
        print('start read {} files with {} polarity'.format(type, polarity))
        rows = []
        for file in files:
            with open(file, mode='r', encoding="utf8") as f:
                rows.append({'text':f.read(), 'polarity':polarity, 'type':type})

        dataframe = pd.DataFrame(rows, columns=['text','polarity','type'])
        print('end read {} files with {} polarity in {} \n'.format(type, polarity, time.time()-start))
        return dataframe

    def unicode_to_ascii(s):
        """
        Parameters : 
        text : String

        Returns
        text : String
        """
        return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')

    def preprocess_text(self, text):
        """
        Parameters : 
        text : String

        Returns
        text : String
        """
        text = text.lower().strip()
        text = self.unicode_to_ascii(text)

        # creating a space between a word and the punctuation following it
        text = re.sub(r"([?.!,¿;])", r" \1 ", text)
        text = re.sub(r'[" "]+', " ", text)

        # replacing everything with space except (1-9, a-z, A-Z, ".", "?", "!", ",",";")
        text = re.sub(r"[^1-9a-zA-Z?.!,¿;]+", " ", text)

        return text

    def get_embedding(self, dataset, nlp, typ):
        """
        Parameters : 
        dataset : DataFrame
        type : String

        Returns
        embedding : List 
        """
        start = time.time()
        print('start embedding for type {}'.format(typ))
        embedding = []

        for index, row in dataset.iterrows():
            # print('Index : ',index)
            embedding.append(nlp(str(row['text_process'])).vector)

        print('end embedding type {} in {} \n'.format(typ, time.time()-start))
        return embedding

    def dump_data_in_file(self, data, filename):
        """
        Parameters : 
        dataset : DataFrame
        filename : String
        """
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def unicode_to_ascii(self, file):
        """
        Converts the unicode file to ascii
        Parameters : 
        path : String

        Returns
        List 
        """
        return ''.join(c for c in unicodedata.normalize('NFD', file)
            if unicodedata.category(c) != 'Mn')

    def split_data_for_train_test_set(self, dataset, train_size, stratify_column):
        
        """
        Parameters : 
        dataset : dataFrame
        train_size : float
        stratify_column : String

        Returns
        trainset, testset : DataFrame
        """
        trainset, testset = train_test_split(dataset, train_size=train_size, 
                               random_state=42, stratify=dataset[stratify_column])

        return trainset, testset
		
    def split_data_for_train_test_val_set(self, dataset, train_size, test_val_size, stratify_column):
        """
        Parameters : 
        dataset : dataFrame
        train_size : float
        test_val_size : float
        stratify_column : String

        Returns
        trainset, testset, valset : DataFrame
        """
        trainset, test_val_set = train_test_split(dataset, train_size=train_size, 
                               random_state=42, stratify=dataset[stratify_column])

        testset, valset = train_test_split(test_val_set, test_size=test_val_size,
                                   random_state=42, stratify=test_val_set[stratify_column])

        return trainset, testset, valset

    def split_tensor_for_train_test_val(self, tensor, trainset_shape_0, testset_shape_0):
        """
        Parameters : 
        tensor : List
        trainset_shapae_0 : Integer 
        testset_shape_0 : Integer 

        Returns
        List 
        """
        return tensor[0:trainset_shape_0,:], \
               tensor[trainset_shape_0:trainset_shape_0+testset_shape_0,:], \
               tensor[trainset_shape_0+testset_shape_0:,:] 

    def tokenize(self, data, type):
        """
        Parameters : 
        data : Series
        type : String

        Returns
        tensor : List
        data_tokenizer : tf.keras.preprocessing.text.Tokenizer
        """
        start = time.time()
        print('start tokenize type {}'.format(type))

        data_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        data_tokenizer.fit_on_texts(data)

        tensor = data_tokenizer.texts_to_sequences(data)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

        print('end tokenize type {} in {} \n'.format(type, time.time()-start))
        return tensor, data_tokenizer
