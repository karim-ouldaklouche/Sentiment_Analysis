# Sentiment Analysis

Based on the IMDB dataset, several models have been used in order to tackle the task of sentiment analysis.

The dataset can be download at the adress : [Dataset link](https://ai.stanford.edu/~amaas/data/sentiment/)

The language used is Python version 3.7.

The first base line models are logistic regression and svm. Scikit-learn and spacy have been used for the models and the embedding of the sentences.

The table above gives the accuracy and the F1 scores for the both models (train and test set) :

<table>
  <tr>
    <td></td>
    <td colspan="2">SVM</td>
    <td colspan="2">Logistic Regression</td>
  </tr>
  <tr>
    <td></td>
    <td>Accuracy</td>
    <td>F1</td>
    <td>Accuracy</td>
    <td>F1</td>
  </tr>
  <tr>
    <td>Trainset</td>
    <td>73.95 %</td>
    <td>73.91 % </td>
    <td>76.05 % </td>
    <td>76.08 %</td>
  </tr>
  <tr>
    <td>Testset</td>
    <td>73.15 % </td>
    <td>72.89 %</td>
    <td>75.82 % </td>
    <td>75.83 %</td>
  </tr>
</table>

To improve the scores, a recurrent neural network based on tensorflow 2 has been used.
The architecture is composed of several layers :
* One of embedding
* Two bidirectionnal of LSTM layers
* One final dense layer

The test with one bidirectionnal LSTM layer has not given much better scores than the SVM and logistic regression. 

<table>
  <tr>
    <td>Trainset</td>
    <td>Testset</td>
  </tr>
  <tr>
    <td>Testset</td>
    <td>94.67 % </td>
    <td>88.78 %</td>
  </tr>
</table>

