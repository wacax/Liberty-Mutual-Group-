#Liberty Mutual Group - Fire Peril Loss Cost
#ver 0.2

#########################
#Init
rm(list=ls(all=TRUE))

#Load/install required libraries
require('ggplot2')
require('leaps')

#Set Working Directory
workingDirectory <- "/home/wacax/Wacax/Kaggle/Liberty Mutual Group/Liberty Mutual Group - Fire Peril Loss Cost"
setwd(workingDirectory)

dataDirectory <- '/home/wacax/Wacax/Kaggle/Liberty Mutual Group/Data/'

#Load external functions


#############################
#Load Data
#Input Data
rows2read <- 'all'
train <- read.csv(paste0(dataDirectory, 'train.csv'), header = TRUE, stringsAsFactors = FALSE, nrows = ifelse(class(rows2read) == 'character', -1, rows2read))
test <- read.csv(paste0(dataDirectory, 'test.csv'), header = TRUE, stringsAsFactors = FALSE, nrows = ifelse(class(rows2read) == 'character' -1, rows2read))

submissionTemplate <- read.csv(paste0(dataDirectory, 'sampleSubmission.csv'), header = TRUE, stringsAsFactors = FALSE)

################################
#Add a new column that indicates whether there was a fire or not
train['fire'] <- ifelse(train$target > 0, 'Fire', 'NoFire')

#Data Transformation
train <- transform(train, var1 = as.factor(var1), var2 = as.factor(var2), var3 = as.factor(var3), 
                   var4 = as.factor(var4), var5 = as.factor(var5), var6 = as.factor(var6), var7 = as.factor(var7), 
                   var8 = as.factor(var8), var9 = as.factor(var9), fire = as.factor(fire))
test <- transform(test, var1 = as.factor(var1), var2 = as.factor(var2), var3 = as.factor(var3), 
                  var4 = as.factor(var4), var5 = as.factor(var5), var6 = as.factor(var6), var7 = as.factor(var7), 
                  var8 = as.factor(var8), var9 = as.factor(var9))

##################################################
#EDA
str(train)
print(table(ifelse(train$target > 0, 1, 0)) / length(ifelse(train$target > 0, 1, 0)))
fireCosts <- as.data.frame(train$target[train$target>0]); names(fireCosts) <- 'Cost'
ggplot(data = train, aes(x = ifelse(train$target > 0, TRUE, FALSE))) +  geom_histogram() 
ggplot(data = fireCosts, aes(x = Cost)) +  geom_density() 
ggplot(data = fireCosts, aes(x = log(Cost))) +  geom_density() 

#NA omit, regsubsets and kmeans are sensitive to NAs
noNAIndices <- which(apply(is.na(train), 1, sum) == 0)

#Predictors Selection
#Fire or not
linearBestModels <- regsubsets(fire ~ ., data = train[noNAIndices, c(seq(3, 19), seq(21,303))], method = 'forward', 
                               nvmax=200, really.big=TRUE)
bestMods <- summary(linearBestModels)
names(bestMods)
plot(bestMods$cp, xlab="Number of Variables", ylab="Cp")
bestNumberOfPredictors <- which.min(bestMods$cp)
points(bestNumberOfPredictors, bestMods$cp[bestNumberOfPredictors],pch=20,col="red")

plot(linearBestModels,scale="Cp") #warning it cannot plot properly
coef(linearBestModels,10)

#Fire damage regression
whichFire <- which(train$target > 0)
linearBestModels <- regsubsets(target ~ ., data = train[intersect(noNAIndices, whichFire), c(seq(2, 19), seq(21,302))], 
                               method = 'forward', nvmax=100, really.big=TRUE)
bestMods <- summary(linearBestModels)
names(bestMods)
plot(bestMods$cp, xlab="Number of Variables", ylab="Cp")
bestNumberOfPredictors <- which.min(bestMods$cp)
points(bestNumberOfPredictors, bestMods$cp[bestNumberOfPredictors],pch=20,col="red")

plot(linearBestModels,scale="Cp") #warning it cannot plot properly
coef(linearBestModels,10)

#regsubsets predict method
predict.regsubsets=function(object,newdata,id,...){
  form=as.formula(object$call[[2]])
  mat=model.matrix(form,newdata)
  coefi=coef(object,id=id)
  mat[,names(coefi)]%*%coefi
}

#10-fold cross-validation
set.seed(101)
folds <- sample(rep(seq(1, 10), length=length(noNAIndices)))
table(folds)
cv.errors <- matrix(NA, 10, 11)
for(k in 1:10){
  bestFit <- regsubsets(target ~ ., data = train[noNAIndices[folds!=k], c(seq(2, 19), seq(21,302))],
                        method = 'forward', nvmax=11, really.big=TRUE)
  for(i in 1:11){
    pred <- predict(bestFit, train[noNAIndices[folds==k], c(seq(2, 19), seq(21,302))], id = i)
    cv.errors[k,i] <- mean((train$target[noNAIndices[folds==k]] - pred)^2)
  }
}
rmse.cv=sqrt(apply(cv.errors,2,mean))
plot(rmse.cv,pch=19,type="b")


#Clustering
#Kmeans (2 groups), The idea is to see if kmeans clustering can help explore the 
#fire vs no fire groups and if they match to some extent to the given labels
derp <- kmeans(train[noNAIndices, c('fire', 'weatherVar32', 'weatherVar33')], 2)

#PCA
derp <- princomp(train[noNAIndices, c('fire', 'weatherVar32', 'weatherVar33')])



