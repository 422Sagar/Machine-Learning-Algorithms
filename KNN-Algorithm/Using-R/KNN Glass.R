## Importing data set
glass <- read.csv(file.choose())
View(glass)
glass$Type
table(glass$Type)
str(glass)
install.packages('caTools')  
install.packages('dplyr')    
install.packages('ggplot2')  
install.packages('class')     
install.packages('caret')    
install.packages('corrplot') 
library(caTools)
library(dplyr)
library(ggplot2)
library(class)
library(caret)
library(corrplot)

# Normalizing the data
normalize <- scale(glass[,1:9])
View(normalize)

# binding the normalized and actual data
glass_data <- cbind(normalize,glass[10])
View(glass_data)
library(DataExplorer)

# Creating the report using the Create_report function 
create_report(glass_data)
set.seed(123)

# Splitting the glass data into train and test using split function 
?sample.split
sample <- sample.split(glass_data$Type,SplitRatio = 0.70)
glass_train <- subset(glass_data,sample==T)
glass_test <- subset(glass_data,sample==F)
?knn
View(glass_train)
colnames(glass_test)

# Applying the knn algorithm to both train and test using knn function 
?knn
glass_model <- knn(glass_train[1:9],glass_test[1:9],glass_train$Type,k=1)

error <- mean(glass_model!=glass_test$Type)

# Creating the confusion matrix
?confusionMatrix
confusionMatrix(glass_model,as.factor(glass_test$Type))

class(glass_model)
class(glass_test$Type)
glass_mode_k <- NULL
error.rate <- NULL

for (i in 1:10) {
  glass_model_k <- knn(glass_train[1:9],glass_test[1:9],glass_train$Type,k=i)
  error.rate[i] <- mean(glass_model!=glass_test$Type)
  
}

knn.error <- as.data.frame(cbind(k=1:10,error.type =error.rate))

ggplot(knn.error,aes(k,error.type))+ 
  geom_point()+ 
  geom_line() + 
  scale_x_continuous(breaks=1:10)+ 
  theme_bw() +
  xlab("Value of K") +
  ylab("Error") 

#k=3 shows sharp decine in the error rate. So, we select k=3 to prepare our model
glass_model <- knn(glass_train[1:9],glass_test[1:9],glass_train$Type,k=1)

error <- mean(glass_model!=glass_test$Type)
confusionMatrix(glass_model,as.factor(glass_test$Type))
final_glass_knn <- knn(glass_train[1:9],glass_test[1:9],glass_train$Type,k=3)
summary(final_glass_knn)
error_final <- mean(final_glass_knn!=glass_test$Type)
confusionMatrix(final_glass_knn,as.factor(glass_test$Type))
# Accuracy : 0.7385          
# 95% CI : (0.6146, 0.8397
