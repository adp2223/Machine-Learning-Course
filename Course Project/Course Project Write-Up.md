---
title: "Machine Learning Course Project Write-Up"
author: "Alok Pattani"
date: "Sunday, February 22, 2015"
output: html_document
---

**LOADING EXTERNAL PACKAGES USED IN THIS WORK**

```r
library(lattice)
library(ggplot2)
library(caret)
library(rpart)
library(MASS)
library(grid)
library(mvtnorm)
library(modeltools)
library(stats4)
library(strucchange)
library(zoo)
library(sandwich)
library(randomForest)
library(combinat)
library(klaR)
```

**LOAD IN PROVIDED TRAINING AND TESTING DATA SETS**

```r
pmltraining <- read.csv("pml-training.csv")
pmltesting <- read.csv("pml-testing.csv")

summary(pmltraining)
summary(pmltesting)
```

**SOME INITIAL LOOKING AT PROVIDED TRAINING SET, REMOVING USELESS DATA**

```r
#Identify columns with NAs or blanks, to be eliminated later
colsToKeep <- (colSums(is.na(pmltraining)) == 0 & colSums(pmltraining == "") == 0)
#First 7 columns aren't really germane to prediction, mark for elimination as well
colsToKeep[1:7] <- FALSE

#Subset down to meaningful columns and eliminate "no window" rows (look different)
usabledata <- subset(pmltraining, new_window == "no", select = colsToKeep)

head(usabledata)
dim(usabledata)
summary(usabledata)
```

**TAKE USABLE DATA FROM PROVIDED TRAINING SET, SPLIT INTO "NEW" TRAINING AND TEST SETS**

```r
set.seed(23)
inTrain <- createDataPartition(y = usabledata$classe, p = 0.75, list = FALSE)

training <- usabledata[inTrain, ]
testing <- usabledata[-inTrain, ]

#Creating training data set without response for use in some models
training_noresponse <- subset(training, select = -c(classe))
```

**BUILD PREDICTION USING CLASSIFICATION TREE ON TRAINING DATA**

```r
rpart_model <- rpart(classe ~ ., data = training)

confusionMatrix(predict(rpart_model, type = "class"), training$classe)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3704  545   53  238   93
##          B  115 1509  128   77  166
##          C  109  375 2022  338  331
##          D  136  193  166 1522  138
##          E   40  167  145  186 1918
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7406          
##                  95% CI : (0.7334, 0.7477)
##     No Information Rate : 0.2847          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6706          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9025   0.5411   0.8043   0.6446   0.7249
## Specificity            0.9099   0.9582   0.9031   0.9475   0.9543
## Pos Pred Value         0.7995   0.7564   0.6369   0.7063   0.7809
## Neg Pred Value         0.9591   0.8969   0.9562   0.9316   0.9391
## Prevalence             0.2847   0.1935   0.1744   0.1638   0.1836
## Detection Rate         0.2570   0.1047   0.1403   0.1056   0.1331
## Detection Prevalence   0.3214   0.1384   0.2203   0.1495   0.1704
## Balanced Accuracy      0.9062   0.7496   0.8537   0.7961   0.8396
```

**BUILD PREDICTION USING BAGGING ON TRAINING DATA**

```r
bag_model <- bag(y = training$classe, x = training_noresponse, 
    bagControl = bagControl(fit = ctreeBag$fit, predict = ctreeBag$pred, 
    aggregate = ctreeBag$aggregate))  

confusionMatrix(predict(bag_model, newdata = training), training$classe)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4073   51    3    3    2
##          B   10 2704   29    4    8
##          C    9   23 2470   44    2
##          D    7   10    9 2304   13
##          E    5    1    3    6 2621
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9832         
##                  95% CI : (0.981, 0.9852)
##     No Information Rate : 0.2847         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9788         
##  Mcnemar's Test P-Value : 1.057e-10      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9924   0.9695   0.9825   0.9759   0.9906
## Specificity            0.9943   0.9956   0.9934   0.9968   0.9987
## Pos Pred Value         0.9857   0.9815   0.9694   0.9834   0.9943
## Neg Pred Value         0.9970   0.9927   0.9963   0.9953   0.9979
## Prevalence             0.2847   0.1935   0.1744   0.1638   0.1836
## Detection Rate         0.2826   0.1876   0.1714   0.1598   0.1818
## Detection Prevalence   0.2867   0.1911   0.1768   0.1626   0.1829
## Balanced Accuracy      0.9934   0.9826   0.9880   0.9863   0.9946
```

**BUILD PREDICTION USING RANDOM FOREST ON TRAINING DATA**  
Random forest already does some cross validation, providing the out of sample error estimate by default.

```r
ranfor_model <- randomForest(classe ~ ., data = training)

ranfor_model
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.49%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4099    3    0    1    1 0.001218324
## B   12 2771    6    0    0 0.006453926
## C    0   12 2498    4    0 0.006364360
## D    0    0   23 2335    3 0.011012283
## E    0    0    1    4 2641 0.001889645
```

**BUILD PREDICTION USING LINEAR DISCRIMINANT ANALYSIS ON TRAINING DATA**

```r
lda_model <- lda(classe ~ ., data = training)

confusionMatrix(predict(lda_model)$class, training$classe)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3376  413  230  145  109
##          B   85 1786  263   92  449
##          C  335  339 1661  268  244
##          D  296  112  296 1762  249
##          E   12  139   64   94 1595
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7063          
##                  95% CI : (0.6987, 0.7137)
##     No Information Rate : 0.2847          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6282          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8226   0.6404   0.6607   0.7463   0.6028
## Specificity            0.9130   0.9235   0.9003   0.9209   0.9737
## Pos Pred Value         0.7901   0.6677   0.5834   0.6490   0.8377
## Neg Pred Value         0.9282   0.9146   0.9263   0.9488   0.9160
## Prevalence             0.2847   0.1935   0.1744   0.1638   0.1836
## Detection Rate         0.2342   0.1239   0.1152   0.1222   0.1107
## Detection Prevalence   0.2964   0.1856   0.1975   0.1884   0.1321
## Balanced Accuracy      0.8678   0.7819   0.7805   0.8336   0.7883
```

**BUILD PREDICTION USING NAIVE BAYES ON TRAINING DATA**

```r
nb_model <- NaiveBayes(classe ~ ., data = training)

confusionMatrix(predict(nb_model, newdata = training)$class, training$classe)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1186   89   44    0   34
##          B  338 1710  268   74  572
##          C 1909  601 1773  818  369
##          D  565  192  304 1166  357
##          E  106  197  125  303 1314
## 
## Overall Statistics
##                                           
##                Accuracy : 0.496           
##                  95% CI : (0.4878, 0.5042)
##     No Information Rate : 0.2847          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3792          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity           0.28899   0.6131   0.7053  0.49386  0.49660
## Specificity           0.98380   0.8923   0.6893  0.88235  0.93788
## Pos Pred Value        0.87657   0.5773   0.3241  0.45124  0.64254
## Neg Pred Value        0.77659   0.9058   0.9172  0.89899  0.89231
## Prevalence            0.28472   0.1935   0.1744  0.16380  0.18357
## Detection Rate        0.08228   0.1186   0.1230  0.08089  0.09116
## Detection Prevalence  0.09387   0.2055   0.3795  0.17927  0.14188
## Balanced Accuracy     0.63639   0.7527   0.6973  0.68811  0.71724
```

**APPLYING RANDOM FOREST MODEL TO TESTING DATA**  
Based on the above results, random forest has the highest accuracy of the models tested. So we'll use the random forest model on the testing data as a further cross validation, and to estimate "out-of-sample" accuracy.

```r
testing$ranfor_pred_classe <- predict(ranfor_model, newdata = testing)

#Estimate of out-of-sample error rate (as a percentage):
round((1 - with(testing, confusionMatrix(ranfor_pred_classe, classe))$overall[1])*100, 1)
## Accuracy 
##      0.6
```

**SETTING UP FINAL TESTING DATA SET, THEN APPLYING PREDICTION**

```r
#Take provided testing data set, limit down to same columns as training set was
finaltesting <- subset(pmltesting, select = colsToKeep)

finaltesting$ranfor_pred_classe <- predict(ranfor_model, newdata = finaltesting)

answers <- finaltesting$ranfor_pred_classe

answers
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

**WRITING PREDICTIONS ON PROVIDED FINAL TESTING SET TO INDIVIDUAL FILES**

```r
#Function provided by course to create 1 file per answer
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)
```
