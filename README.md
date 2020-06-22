# ML_LoanRisk

## Recsources
- Data: [Loan data](https://github.com/JasmeerSangha/ML_LoanRisk/blob/master/LoanStats_2019Q1.zip)
- Tools: scikit Python 3 library

## Summary
Credit risk is an inherently unbalanced classification problem, as the number of good loans easily outnumber the number of risky loans. Therefore, I will need to employ different techniques to train and evaluate models with unbalanced classes. I will use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling. Effectiveness of the models will be judged by three attributes:
- confusion matrix
- balanced accuracy score
- classification report

(both the accuracy score and classification report can be derived from the confusion matrix and thus both will be discussed to a lesser degree)
## Analysis
### Oversampling
The will first employ two different oversampling methods to create a balance between the two populations of data. 
1. RandomOverSampler chooses random points with in the population and duplicates them in order to increase the data pool. From the table below one can see this created a large amount of Type 1 errors resulting in an abysmal precision score, a common theme throughout this report. The balanced accuracy score sits at 0.67 which is low. It is clear that this method is only useful if low precision is not an issue, as its sensitivity is a reasonable 70%.


ROS | Predicted True | Predicted False
--- | --- | ---
**Actual True** | 71 | 30
**Actual False** | 6142 | 10962



2. SMOTE creates points that are reasonably similar to points in the smaller population in order to increase the data pool. Again, when looking at the chart, this method would only be useful if precision had no bearing on the decision. The SMOTE method returns values hovering around similar values, an accuracy of 0.65 and a sensitivity of 65%.


SMOTE | Predicted True | Predicted False
--- | --- | ---
**Actual True** | 65 | 36
**Actual False** | 5726 | 11378


### Undersampling
The will first employ two different undersampling methods to create a balance between the two populations of data. 
1. RandomUnderSampler chooses random points with in the larger population in order to create a balanced weighting between both groups. Once again, the model produces an acceptable sensitivy but poor precision result. In this case, all statstics were worse than previous models thus I would not recommend this method.

RUS | Predicted True | Predicted False
--- | --- | ---
**Actual True** | 57 | 44
**Actual False** | 7076 | 10028



2. SMOTEENN is a combination of over and under sampling, this method over samples the data and then undersamples it by removing points that are closer to ths opposing data set, effectively better handling of outliers. This model again has a very similar output. If these were our only options, I would recommend this method as it has the best sensitivity thus far, 73%, and a balanced accuracy score of 0.65. The precision value is still nothing to write home about.


SMOTEENN | Predicted True | Predicted False
--- | --- | ---
**Actual True** | 74 | 27
**Actual False** | 7464 | 9640

### Ensembles

The aforementioned methods were suitable though results were underwhelming, we can bootstrap ensembles of weak learners together to create a more robust model.

1. BalancedRandomForestClassifier creates many simple tree models. The solutions of each are compared to one another and the more frequent conclusion is chosen as the predicted outcome. This model has created the smallest amount of Type 1 errors thus far, resulting in a marginally higher precision, 2%. The balanced accuracy score, 0.73, suggests this model is also the same order of accuracy.

brfc | Predicted True | Predicted False
--- | --- | ---
**Actual True** | 63 | 38
**Actual False** | 2674 | 14430

2. EasyEnsembleClassifier is an ensemble of AdaBoost learners trained on different balanced boostrap samples. This method has a precision an order of magnitude higher than previous models (10%) and a sensitivity of 92%. With an accuracy score of 0.93, this is clearly the best method of those discussed in this report. 


eec | Predicted True | Predicted False
--- | --- | ---
**Actual True** | 93 | 8
**Actual False** | 983 | 16121


## Conclusion

After testing oversampling, undersampling and ensemble classifiers, many had sensitivity and accuracy scores ranging from 0.55-0.75. The EasyEnsembleClassfier was for and away the best option as it had a precision higher than previous models (10%) and a sensitivity of 92%, and an accuracy score of 0.93.
