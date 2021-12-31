# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model trained was a [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) model, trained on [UCI Census Data](https://archive.ics.uci.edu/ml/datasets/census+income).

The model uses all the Scikit Learn default hyperperameters, except that the max iteration is set to 500.

## Intended Use

The model is intended for predicting if a person's salary will be above or below $50k.

## Training Data

The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). The target class had two categories, ">50k" and "<50k".

The original data set has 32561 rows, and a 75-25 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the categorical features, a MinMaxScaler was used on continuous features, and a label binarizer was used on the labels.

## Evaluation Data
Evaluation data  was taken from the 75-25 split of the training data, described above.

## Metrics
The model was evaluated using F1 score. The overall f1 score for the test set was 0.6286.

### Education Sliced Metrics
|Group|# Samples| Precision | Recall | F1|
|---|---|---|---|---|
| PRESCHOOL |12| 1.0000| 1.0000 | 1.0000|
| 1ST-4TH |41| 1.0000| 1.0000 | 1.0000|
| 5TH-6TH |65| 1.0000| 0.5000 | 0.6667|
| 7TH-8TH |131| 1.0000|  0.1000 | 0.1818|
| 9TH |120| 1.0000| 0.0000 | 0.0000|
| 10TH |183| 1.0000|  0.0000 | 0.0000|
| 11TH |240| 1.0000| 0.2857 | 0.4444|
| 12TH |68| 1.0000| 0.1429 | 0.2500|
| HS-GRAD |2128| 0.6607| 0.2277 | 0.3387|
| SOME-COLLEGE |1490| 0.6868| 0.4371 | 0.5342|
| PROF-SCHOOL |97| 0.9000| 0.8400 | 0.8690|
| ASSOC-VOC |259| 0.7302| 0.6301 | 0.6765|
| ASSOC-ACDM |179|0.7750| 0.5536 | 0.6458|
| BACHELORS |999| 0.7277| 0.7536 |0.7404|
| MASTERS |360| 0.7602| 0.8660 | 0.8096|
| DOCTORATE |82| 0.8167| 0.8750 | 0.8448|


## Ethical Considerations

Based on our metrics from the education slices above, we can see that there is definitely some bias when looking at certain education groups together. For example, any group below HS-GRAD (a non high school graduate) has a precision of 1.0, with low recall and f1 scores. The model becomes much less bias above this group (SOME-COLLEGE and beyond).

## Caveats and Recommendations

It is not recommended that this model be used for populations with educations less than high school graduate, as the model bias increases drastically.


