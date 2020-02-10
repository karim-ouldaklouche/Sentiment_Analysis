import pandas as pd

from sklearn.utils import shuffle

import spacy

from sklearn.metrics import accuracy_score, f1_score

import joblib

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sa_preparate import SA_Preparate

def main():
    print('Start svm-logreg model')

    sa_preparate = SA_Preparate()

    # The data are in the "data" directory. Change for your own path .
    path_file_pos_train = '../data/aclImdb/train/pos/'
    path_file_neg_train = '../data/aclImdb/train/neg/'
    path_file_pos_test = '../data/aclImdb/test/pos/'
    path_file_neg_test = '../data/aclImdb/test/neg/'

    files_pos_train = sa_preparate.parse_directory_for_files(path_file_pos_train)
    files_neg_train = sa_preparate.parse_directory_for_files(path_file_neg_train)

    files_pos_test = sa_preparate.parse_directory_for_files(path_file_pos_test)
    files_neg_test = sa_preparate.parse_directory_for_files(path_file_neg_test)

    pos_trainset = sa_preparate.read_files_to_dataframe(files_pos_train, polarity=1,type='train')
    neg_trainset = sa_preparate.read_files_to_dataframe(files_neg_train, polarity=0,type='train')

    pos_testset = sa_preparate.read_files_to_dataframe(files_pos_test, polarity=1,type='test')
    neg_testset = sa_preparate.read_files_to_dataframe(files_neg_test, polarity=0,type='test')

    # Concatenation of the dataframe
    dataset_all = pd.concat([pos_trainset,  neg_trainset,
    						   pos_testset, neg_testset])

    # Shuffle
    dataset_all = shuffle(dataset_all)

    # Write the dataframe into one file
    dataset_all.to_csv('../data/aclImdb_all.csv')

    """
    Split the dataset
    """
    trainset, testset = sa_preparate.split_data_for_train_test_set(dataset=dataset_all,
    	   train_size=0.8,
    	   stratify_column='polarity')

    """
    Preprocessing of the data before the embedding
    """
    trainset['text_process'] = trainset['text'].apply(lambda x: sa_preparate.preprocess_text(x))
    testset['text_process'] = testset['text'].apply(lambda x: sa_preparate.preprocess_text(x))

    """
    Embedding of the text with the small model with spacy
    """
    nlp=spacy.load("en_core_web_sm")

    trainset_embedding = sa_preparate.get_embedding(trainset, nlp, typ='train')
    sa_preparate.dump_data_in_file(trainset_embedding, 'trainset_embedding.plk')

    testset_embedding = sa_preparate.get_embedding(testset, nlp, typ='test')
    sa_preparate.dump_data_in_file(testset_embedding, 'testset_embedding.plk')

    # Check the embedding
    print(len(trainset_embedding[0]))
    print(len(testset_embedding[0]))

    """
    SVM model
    """

    X = trainset_embedding
    y = trainset['polarity']
    clf_svm = SVC(gamma='scale')

    clf_svm.fit(X, y)

    joblib.dump(clf_svm, 'clf_svm.dump')

    clf_svm = joblib.load('clf_svm.dump')

    train_predict_svm = clf_svm.predict(trainset_embedding)
    test_predict_svm = clf_svm.predict(testset_embedding)

    # Accuracy
    accuracy_score_train_svm = accuracy_score(train_predict_svm, trainset['polarity'])
    accuracy_score_test_svm  = accuracy_score(test_predict_svm, testset['polarity'])

    # F1
    f1_score_train_svm = f1_score(train_predict_svm, trainset['polarity'])
    f1_score_test_svm  = f1_score(test_predict_svm, testset['polarity'])

    print("TRAINSET - len : {}".format(len(train_predict_svm)))
    print("TRAINSET - accuracy svm : {} %".format(round(100*accuracy_score_train_svm,2),"%"))
    print("TRAINSET - f1 svm : {} %".format(round(100*f1_score_train_svm,2),"%"))
    print('\n')

    print("TESTSET - len : {}".format(len(test_predict_svm)))
    print("TESTSET - accuracy svm : {} %".format(round(100*accuracy_score_test_svm,2),"%"))
    print("TESTSET - f1 svm : {} %".format(round(100*f1_score_test_svm,2),"%"))

    """
    Logistic Regression
    """

    clf_lr = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')

    clf_lr.fit(X, y)

    joblib.dump(clf_lr, 'clf_lr.dump')

    clf_lr = joblib.load('clf_lr.dump')

    train_predict_lr = clf_lr.predict(trainset_embedding)
    test_predict_lr = clf_lr.predict(testset_embedding)

    # Accuracy
    accuracy_score_train_lr = accuracy_score(train_predict_lr, trainset['polarity'])
    accuracy_score_test_lr  = accuracy_score(test_predict_lr, testset['polarity'])

    # F1
    f1_score_train_lr = f1_score(train_predict_lr, trainset['polarity'])
    f1_score_test_lr  = f1_score(test_predict_lr, testset['polarity'])

    print("TRAINSET - len : {}".format(len(train_predict_svm)))
    print("TRAINSET - accuracy lr : {} %".format(round(100*accuracy_score_train_lr,2),"%"))
    print("TRAINSET - f1 lr : {} %".format(round(100*f1_score_train_lr,2),"%"))
    print('\n')

    print("TESTSET - len : {}".format(len(test_predict_lr)))
    print("TESTSET - accuracy lr : {} %".format(round(100*accuracy_score_test_lr,2),"%"))
    print("TESTSET - f1 lr : {} %".format(round(100*f1_score_test_lr,2),"%"))

if __name__ == '__main__':
    main()

