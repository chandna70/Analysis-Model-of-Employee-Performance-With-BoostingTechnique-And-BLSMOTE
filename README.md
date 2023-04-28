# Analysis Model of Employee Performance With Boosting Technique And BLSMOTE

## 1. Motivation the author
The perfomance is the part of extremely important and interesting, because it is proved for its benefit, the company wants employees to work truly according to their skills to achieve well results, without any well results from all employees, then the success in achieving goals will be difficult to achieve.

Best employee performance is the one of illustratation from quality of human resources. This performance represents as person’s success. Human of resources are such as having a critical thinking, curiosity, status, organization, and educational background.

Machine learning is the study of computer algorithms that can improve automatically through experience and by the use of data (T. Mitchell, 1997). It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as “training data”, in order to make predictions or decisions without being explicitly programmed to do so.

In statistics and machine learning, ensembling technique is the method of combining from various set of learners (individual modeling) together, to obtain better predictive performance. Generally, ensembling technique consists of bagging, boosting, and stacking.

- Bagging (bootstrap aggregating) involves having each model in the ensemble vote with equal weight.
- Boosting is sequential ensemble learning technique to convert weak base learners to strong learner that performs better and is less bias.
- Stacking is ensemble technique that combines all machine learning algorithms via meta-learning

From those explanations, we will be analysis and predicting the employee performance with using boosting.
## 2. Preparing the tools for analysis and building model
Through building model and visualization, we will use R version **4.1.1**. For building model, that applies **XGBoost** (Extreme Gradient Boosting) with package xgboost in R, and for the visualization, we will plot data with _ggplot2_, _ggpubr_, and _ggcorrplot_.

BLSMOTE (Borderline-SMOTE) algorithms attempt to learn the borderline of each class, where these borderline instances and the ones nearby are more likely to be misclassified than the ones far from the borderline (H.Han et al., 2005). Finding the optimal _k value_, we will use _factoextra_ and _NbClust_.

## 3. Metadata

Each of number of training and testing data are 8153 and 3000. There are 21 predictor variables and the rest are predictive variable. Here the list of predictor variables:
|variable|Data Type|Variable|Data Type|
| --- | --- | --- | --- |
|job_level|Category type|Education_level|Category type|
|job_duration_in_current_job_level|Numeric type|GPA|Numeric type|
|person_level|Category type|year_graduated|Numeric type|
|job_duration_in_current_person_level|Numeric type|job_duration_from_training|Numeric type|
|job_duration_in_current_branch|Numeric type|branch_rotation|Numeric type|
|Employee_type |Category type|job_rotation|Numeric type|
|gender|Category type|assign_of_otherposition|Numeric type|
|age|Numeric type|annual.leaves|Numeric type|
|marital_status_maried.Y.N|Binary type|sick_leaves|Numeric type|
|number_of_dependeces|Numeric type|Last_achievement_.|Numeric type|
|Achievemnet_above_100._during3quartal|Numeric type|



## 4. Exploration the Data
From these density plots, here what we got:
![plot](https://github.com/chandna70/Analysis-Model-of-Employee-Performance-With-BoostingTechnique-And-BLSMOTE/blob/main/image/Distribution%20Density%20of%20Continous%20data.png)

- Job duration in current job level and job duration in current person level are seem like similar distribution.
- GPA, assign of other position, and sick leaves are the most likely to have many zero value data. These variables may have several outlier.
- Year graduated and Age are seem like they are skewed to the left.


It needs some the enlightenments of the statement from the second point. From these distribution, it will be served into Box-Whisker plot
![plot](https://github.com/chandna70/Analysis-Model-of-Employee-Performance-With-BoostingTechnique-And-BLSMOTE/blob/main/image/Boxplot%20data.png)

Based on this Box-Whisker, many predictors consist of outlier data

![plot](![plot](https://github.com/chandna70/Analysis-Model-of-Employee-Performance-With-BoostingTechnique-And-BLSMOTE/blob/main/image/Boxplot%20data.png)
)



