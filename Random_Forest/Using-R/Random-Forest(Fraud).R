## Importing the data set
fraud_analysis <- read.csv(file.choose())
View(fraud_analysis)
colnames(fraud_analysis)
str(fraud_analysis)
library(DataExplorer)
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

library(randomForest)
fit.forest <- randomForest(fraud_train$Risky_Good~.,data=fraud_train[,c(-3,-7)], na.action=na.roughfix,importance=TRUE)
mean(fraud_train$Risky_Good==predict(fit.forest,fraud_train[,c(-3,-7)])) #0.94333 accuracy 

# Predicting test data 
pred_test <- predict(fit.forest,newdata=fraud_test[c(-3,-7)])
mean(pred_test==fraud_test$Risky_Good) # Accuracy =  0.79
library(caret)
library(psych)

# Cross table 
library(gmodels)
rf_perf<-CrossTable(fraud_test$Risky_Good, pred_test,
                    prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
                    dnn = c('actual default', 'predicted default'))

# Total Observations in Table:  300 


#                | predicted default 
# actual default |      Good |     Risky | Row Total | 
# ---------------|-----------|-----------|-----------|
#           Good |       236 |        10 |       246 | 
#    |     0.787 |     0.033 |           | 
# ---------------|-----------|-----------|-----------|
#          Risky |        52 |         2 |        54 | 
#    |     0.173 |     0.007 |           | 
# ---------------|-----------|-----------|-----------|
#   Column Total |       288 |        12 |       300 | 
# ---------------|-----------|-----------|-----------|

# 236/300 = 0.7866667