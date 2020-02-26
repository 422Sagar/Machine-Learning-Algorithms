## Importing the data set
fraud_analysis <- read.csv(file.choose())
View(fraud_analysis)
colnames(fraud_analysis)
str(fraud_analysis)
library(DataExplorer)

# Creating the report using create_report function 
create_report(fraud_analysis)
library(caret)
library(corpcor)
library(psych)
range(fraud_analysis$Taxable.Income)
pairs.panels(fraud_analysis)

Risky_Good = ifelse(fraud_analysis$Taxable.Income<= 30000, "Risky", "Good")
FC = data.frame(fraud_analysis,Risky_Good)
View(FC)
colnames(FC)

library(dplyr)
fraud_train <- FC[1:300,]
colnames(fraud_train)
fraud_test <- FC[301:600,]
colnames(fraud_train)

# Applying c5.0
library("C50")
?C5.0
model_DT <- C5.0(fraud_train[,c(-3,-7)],fraud_train$Risky_Good)
summary(model_DT)
plot(model_DT)
pred_DT <- as.data.frame(predict(model_DT,fraud_test))
table(pred_DT)
pred_test_df <- predict(model_DT,newdata=fraud_test)
mean(pred_test_df==fraud_test$Risky_Good)

#0.82 accuracy
library(gmodels)
CrossTable(pred_test_df,fraud_test$Risky_Good)
confusionMatrix(fraud_test$Risky_Good,pred_test_df)

# Confusion Matrix and Statistics

# Reference
# Prediction Good Risky
# Good   246     0
# Risky   54     0

# Accuracy : 0.82            
# 95% CI : (0.7718, 0.8618)
# No Information Rate : 1               
# P-Value [Acc > NIR] : 1               

# Kappa : 0               

# Mcnemar's Test P-Value : 5.498e-13       
                                          
#           Sensitivity : 0.82            
#           Specificity :   NA            
#        Pos Pred Value :   NA            
#        Neg Pred Value :   NA            
#            Prevalence : 1.00            
#        Detection Rate : 0.82            
#  Detection Prevalence : 0.82            
#     Balanced Accuracy :   NA            
#                                        
#      'Positive' Class : Good

