# Importing the data set
wine <- read.csv(file.choose())
View(wine)

# Correlation between the variables
cor(wine)
wine1 <- princomp(wine,cor = TRUE, scores = TRUE, covmat = NULL )
wine1$scores
str(wine1)

# Summary of the data 
summary(wine1)
loadings(wine1)
plot(wine1)
biplot(wine1)
wine1$scores[,1:3]
wine <- cbind(wine,wine1$scores[,1:3])
View(wine)
clust_data <- wine[,15:17]

#Normalizing the data
norm_clust <- scale(clust_data)
dist2 <- dist(norm_clust,method = "euclidean")
fit2 <- hclust(dist2,method="complete")
plot(fit2)   #displaying dendogram

groups1 <- cutree(fit2,5)

final11 <- as.matrix(groups1)
View(final11)

final1<-cbind(final11,wine) # binding column wise with orginal data
View(final1)



