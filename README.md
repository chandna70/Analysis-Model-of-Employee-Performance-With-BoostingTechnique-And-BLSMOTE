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
![plot](https://github.com/chandna70/Analysis-Model-of-Employee-Performance-With-BoostingTechnique-And-BLSMOTE/blob/main/image/Distribution%20Density%20of%20Continous%20data.png "The visualization of numeric variable")

- Job duration in current job level and job duration in current person level are seem like similar distribution.
- GPA, assign of other position, and sick leaves are the most likely to have many zero value data. These variables may have several outlier.
- Year graduated and Age are seem like they are skewed to the left.


It needs some the enlightenments of the statement from the second point. From these distribution, it will be served into Box-Whisker plot
![plot](https://github.com/chandna70/Analysis-Model-of-Employee-Performance-With-BoostingTechnique-And-BLSMOTE/blob/main/image/Boxplot%20data.png "Box-Plot visualization")

Based on this Box-Whisker, many predictors consist of outlier data

![plot](https://github.com/chandna70/Analysis-Model-of-Employee-Performance-With-BoostingTechnique-And-BLSMOTE/blob/main/image/Rplot01.png "Density visualization to each 0 and 1 class")

Through from this visualization, I guess that some maximum value of the predictor from 0-class is bigger than maximum value of the predictor from 1-class. Furthermore, we will explain for the category data.

![plot](https://github.com/chandna70/Analysis-Model-of-Employee-Performance-With-BoostingTechnique-And-BLSMOTE/blob/main/image/Categorical%20data%20to%20target%20class.png "Category Data for each target class")

We can say that:

- The proportion of JG04 of 0-class is bigger than the proportion of JG04 of 1-class
- For the person level, PG03 of 0-class is bigger than PG03 of 1-class
- RM type A of 0-class is bigger than RM type A of 1-class
- Female employee isn’t the best performance relatively
- Every employee who has been married is not the best performance
- The person who has education level 4 is not the best performance
- Through these statement points, we don’t see any impact whether the employee is the best or not. Hence we will make these categories data to transfrom dummy variable. (One-hot encoding).

This data needs check the multicolinearity. Before the data plots the visualization, we need the category data to transform with one-hot encoding data.
![plot](https://github.com/chandna70/Analysis-Model-of-Employee-Performance-With-BoostingTechnique-And-BLSMOTE/blob/main/image/Multicolinearity.png "Correlation of the predictors")

From this plot data, we can see there are some predictor variable which indicate multicolinearity. Then we need L2 Regularization for handle this problem.

## 5.Preprocessing data
Based on the proportion of the target class, it needs to generate data by BLSMOTE. Before generating data with BLSMOTE, k value must be found.
According to the graph of the optimal k, optimal value reaches at (2,0.45). Then we got the optimal k is 2. Now we can generate target class of data.
```
fviz_nbclust(df_train[,!(colnames(df_train)=="Best.Performance")],kmeans,method='silhouette')+theme_black()
```
![plot](https://github.com/chandna70/Analysis-Model-of-Employee-Performance-With-BoostingTechnique-And-BLSMOTE/blob/main/image/Rplot03.png "Optimal number of k clusters")
According to the graph of the optimal k, optimal value reaches at (2,0.45). Then we got the optimal k is 2. Now we can generate target class of data.
```
prop.table(table(df_train$Best.Performance))
data_training_new<-BLSMOTE(df_train[,(!colnames(df_train)=="Best.Performance")],df_train$Best.Performance,K=2)
data_training_new<-data_training_new$data
prop.table(table(data_training_new$class))
```
![plot](https://github.com/chandna70/Analysis-Model-of-Employee-Performance-With-BoostingTechnique-And-BLSMOTE/blob/main/image/propotions.JPG "Before and after from the proportion of the target class")

## 6. Building the model data

Before build the model, it makes sure that data must be convert to xgb.DMatrix. XGBoost accepts only xgb.DMatrix data.

```
##make x_train,y_train,x_tesst,y_test
x_train<-data_training_new[,(!colnames(data_training_new)=="class")]
y_train<-as.numeric(data_training_new$class)
x_test<-df_test
y_test<-reference_data
##Modelling
#xgboost
x_train_xgb<-xgb.DMatrix(as.matrix(x_train),label=y_train)
x_test_xgb<-xgb.DMatrix(as.matrix(x_test),label=y_test)
```
Through x_train_xgb and x_test_xgb has been obtained, the model must define the hyperparameter from XGBoost and find the best hyperparameter.

```
#xgboost
x_train_xgb<-xgb.DMatrix(as.matrix(x_train),label=y_train)
x_test_xgb<-xgb.DMatrix(as.matrix(x_test),label=y_test)
params_xgb<-list(booster = "dart", 
                 objective = "binary:logistic",eta=0.3,gamma=1,max_depth=5,
                 min_child_weight=2,subsample=1,colsample_bytree=1,lambda=1.25,alpha=0.75)
xgb_cv<-xgb.cv(params=params_xgb,
               data=x_train_xgb,nrounds=600,nfold=5,showsd = T, 
               early.stop.round = 35, maximize = F,metrics=c('auc'))
gb_dt <- xgb.train(params = params_xgb,
                   data = x_train_xgb,
                   nrounds = xgb_cv$best_iteration,
                   print_every_n = 2,
                   eval_metric=c('auc'),
                 watchlist=list(train=x_train_xgb,eval=x_test_xgb))
```

## 7. Evaluation from the model
Through from our model has been built, then we have to see how far its performance. We will use the performance with confusion_matrix
![plot](https://github.com/chandna70/Analysis-Model-of-Employee-Performance-With-BoostingTechnique-And-BLSMOTE/blob/main/image/Capture.JPG "Confusion Matrix")
From this image, here it is for the intepretation:

- The accuracy and sensitivity are seems relatively good and balance
- The other side, the specificity are sounds like not good. It’s only 19.26%, which means that the proportion of true negative is less than the sum of true negative and false positive .
- Kappa’s value is 0.0324. It indicates that the accuracy of model where the data are just randomly assigned.
- The prevalence indicate that 83.73% is often appear the worst performance from our testing data

From gb_dt model, we can look for the variable important. From variable important, we know how much the predictor variables contribute for the model.

