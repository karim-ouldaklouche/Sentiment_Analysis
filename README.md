# Sentiment_Analysis

Based on the IMDB dataset, several models have been used in order to tackle the task of sentiment analysis.

The dataset can be download at the adress : [Dataset link](https://ai.stanford.edu/~amaas/data/sentiment/)

The language used is Python version 3.7.

The first base line models are logistic regression and svm. Scikit-learn and spacy have been used for the models and the embedding of the sentences.

The table above gives the accuracy and the F1 scores for the both models (train and test set) :

|			       |   SVM                     | Logistic Regression      ||
|  ---       |:-----------:|:-----------:|:-----------:|:----------:|
|             |  Acc        | F1          |    Acc      |     F1     |
| Trainset   |  73.95 %    |  73.91 %    |  76.05 %    |  76.08 %   |
| Testset    |  73.15 %    |  72.89 %    |  75.82 %    |  75.83 %   ||


