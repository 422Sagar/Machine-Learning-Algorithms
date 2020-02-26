##Importing forestfire data and saving it into a local variable. Also checking the summary statistics.

forestfires <- read.csv(file.choose())
view(forestfires)
ffire <- forestfires
attach(ffire)

summary(ffire)
colnames(ffire)
sum(is.na(ffire))
dim(ffire)
View(ffire)
#We don't need first two columns because that information is redundant.
#So,we'll remove first two columns of month and day.
ffire <- ffire[,-c(1,2)]
dim(ffire)
str(ffire) #size_category should be a factor variable


#Dividing the dataset into training and test data

library(caret)

x <- createDataPartition(ffire$size_category, p = 0.7, list = F)
?createDataPartition
ff_train <- ffire[x,]
ff_test <- ffire[-x,]

dim(ff_train)
dim(ff_test)


##Applying Support Vector Classifier(linear kernel) on the training dataset

??svm
library(e1071)
ff.lsvm <- svm(ff_train$size_category~., data = ff_train, kernel = 'linear', cost = 1,scale = T)
summary(ff.lsvm) #Total 125 support vectors

ff.lsvm$index #indices of support vectors
lsvm.predict <- predict(ff.lsvm, ff_test)
library(caret)
confusionMatrix(lsvm.predict, ff_test$size_category)

# Confusion Matrix and Statistics

# Reference
# Prediction large small
# large    39     2
# small     2   111

# Accuracy : 0.974           
# 95% CI : (0.9348, 0.9929)
# No Information Rate : 0.7338          
# P-Value [Acc > NIR] : 8.314e-16       

# Kappa : 0.9335          

# Mcnemar's Test P-Value : 1               
                                          
#           Sensitivity : 0.9512          
#           Specificity : 0.9823          
#        Pos Pred Value : 0.9512          
#        Neg Pred Value : 0.9823          
#            Prevalence : 0.2662          
#        Detection Rate : 0.2532          
#  Detection Prevalence : 0.2662          
#     Balanced Accuracy : 0.9668          
#                                        
#      'Positive' Class : large 



#Tuning the hyperparameters to find the optimal value of cost and gamma

set.seed(1)
library(e1071)
colnames(ff_train)
dim(ff_train)
dim(ff_train$FFMC)
dim(ff_train$size_category)
View(ff_train)
tune.lsvm <- tune(svm, ff_train$size_category~., data = ff_train, kernel = 'linear',ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100),
                                                                                                  gamma = c(0.5, 1, 2, 3, 4)))
summary(tune.lsvm) #cost = 10, gamma = 0.5 gives the least error


#Building the model again with new value of cost and gamma

lsvm.best <- svm(size_category~., data = ff_train, kernel = 'linear', cost = 10,
                 gamma = 0.5, scale = T )
summary(lsvm.best) #Total 78 support vectors

lsvm.best$index #indices of support vectors
blsvm.predict <- predict(lsvm.best, ff_test)

confusionMatrix(blsvm.predict, ff_test$size_category) #96.75% accuracy

# Confusion Matrix and Statistics

# Reference
# Prediction large small
# large    39     2
# small     2   111

# Accuracy : 0.974           
# 95% CI : (0.9348, 0.9929)
# No Information Rate : 0.7338          
# P-Value [Acc > NIR] : 8.314e-16       

# Kappa : 0.9335          

# Mcnemar's Test P-Value : 1               
                                          
#           Sensitivity : 0.9512          
#           Specificity : 0.9823          
#        Pos Pred Value : 0.9512          
#        Neg Pred Value : 0.9823          
#            Prevalence : 0.2662          
#        Detection Rate : 0.2532          
#  Detection Prevalence : 0.2662          
#     Balanced Accuracy : 0.9668          
                                         
#      'Positive' Class : large 

#Applying Support Vector Machines(Radial basis function,RBF kernel) 

library(e1071)

ff.rsvm <- svm(ff_train$size_category~., data = ff_train, kernel = 'radial', cost = 1,gamma = 0.5, scale = T )
summary(ff.rsvm) #Total  support vectors

ff.rsvm$index #indices of support vectors
rsvm.predict <- predict(ff.rsvm, ff_test)
library(caret)
confusionMatrix(rsvm.predict, ff_test$size_category)

# Confusion Matrix and Statistics

# Reference
# Prediction large small
# large     1     0
# small    40   113

# Accuracy : 0.7403          
# 95% CI : (0.6635, 0.8075)
# No Information Rate : 0.7338          
# P-Value [Acc > NIR] : 0.4693          

# Kappa : 0.0354          

# Mcnemar's Test P-Value : 6.984e-10       
                                          
#            Sensitivity : 0.024390        
#            Specificity : 1.000000        
#         Pos Pred Value : 1.000000        
#         Neg Pred Value : 0.738562        
#             Prevalence : 0.266234        
#         Detection Rate : 0.006494        
#   Detection Prevalence : 0.006494        
#      Balanced Accuracy : 0.512195        
                                          
#       'Positive' Class : large

#Tuning the hyperparameters to find the optimal value of cost and gamma

set.seed(1)
library(e1071)
??tune
tune.rsvm <- tune(svm, ff_train$size_category~., data = ff_train, kernel = 'radial',ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100),
                                                                                                  gamma = c(0.5, 1, 2, 3, 4)))
summary(tune.rsvm) #cost = 1, gamma = 0.5 gives the least error


# To build the model again with new values of cost & gamma, we don't have to follow all the previous steps.
# tune.rsvm object not only gives us the best possible value of cost and gamma but saves the best model as well.
# Extracting best model out of tune.rsvm-

rsvm.best <- tune.rsvm$best.model

summary(rsvm.best) #Total  support vectors

rsvm.best$index #indices of support vectors
brsvm.predict <- predict(rsvm.best, ff_test)

confusionMatrix(brsvm.predict, ff_test$size_category)


#ROC curve of best models for both the kernels so far-

library(ROCR)
#Function to draw ROC plots and returning area under curve
rocplot <- function(pred, truth){
  pred1 <- prediction(pred, truth)
  perf <- performance(pred1, 'tpr', 'fpr')
  plot(perf, colorize = T)
  auc <- performance(pred1, 'auc')
  auc <- unlist(slot(auc, "y.values"))
  return(auc)
  }

#Getting predicted values from the best linear and radial models 
lsvm.best <- svm(size_category~., data = ff_train, kernel = 'linear', cost = 10,
               gamma = 0.5, scale = T, decision.values = T )
blsvm.predict <- attributes(predict(lsvm.best, ff_test, decision.values = T))$decision.values

rsvm.best <- svm(size_category~., data = ff_train, kernel = 'radial', cost = 1,
               gamma = 0.5, scale = T,decision.values = T )
brsvm.predict <- attributes(predict(ff.rsvm, ff_test, decision.values = T))$decision.values


#Plotting ROCs of the best linear and radial model.
par(mfrow = c(1,2))
auc.lsvm.best <- rocplot(blsvm.predict, ff_test$size_category)
legend("center", legend= auc.lsvm.best,
       col=c("green"), lty=1, cex=1)
auc.rsvm.best <- rocplot(brsvm.predict, ff_test$size_category)
legend("center", legend= auc.rsvm.best,
       col=c("red"), lty=1, cex=1)



# Conclusions-
# SVM with linear kernel has a better accuracy for this dataset as compared to radial kernel.
# Linear Kernel
# Accuracy - 96.75%
# Area under ROC curve- 0.997
# 
# Radial Kernel
# Accuracy - 72.73%
# Area under ROC curve- 0.472




