# Sentiment Analysis

Based on the IMDB dataset, several models have been used in order to do the task of sentiment analysis.

The dataset can be download at the adress : [Dataset link](https://ai.stanford.edu/~amaas/data/sentiment/)

The language used is Python version 3.7.

The first baseline models are logistic regression and svm. Scikit-learn and spacy have been used for the models and the embedding of the sentences.

To improve the scores, a recurrent neural network based on tensorflow 2.1.0 has been used.
The architecture is composed of several layers :
* One of embedding
* Two bidirectionnal of LSTM layers
* One final dense layer

A GPU version of tensorflow 2 has been used on colab.research of google.

The table above gives the accuracy all the models (train and test set) :

<table>
  <tr>
    <td></td>
    <td>SVM</td>
    <td>Logistic Regression</td>
    <td>Recurrent Neural Network<td>
  </tr>
  <tr>
    <td>Trainset</td>
    <td>73.95 %</td>
    <td>76.05 % </td>
    <td>93.64 %</td>
  </tr>
  <tr>
    <td>Testset</td>
    <td>73.15 % </td>
    <td>75.82 %</td>
    <td>89.34 %</td>
  </tr>
</table>

We don't have any overfitting from the models according to the scores. Of course, improvements can be possible like a better preparation of the data, tuning the hyperparmeter for the differents models.



