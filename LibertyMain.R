#Liberty Mutual Group - Fire Peril Loss Cost
#ver 0.4

#########################
#Init
rm(list=ls(all=TRUE))

#Load/install required libraries
require('ggplot2')
require('leaps')
require('caret')
require('gbm')
require('parallel')
require('foreach')
require('glmnet')
require('RVowpalWabbit')

#Set Working Directory
workingDirectory <- "/home/wacax/Wacax/Kaggle/Liberty Mutual Group/Liberty Mutual Group - Fire Peril Loss Cost"
setwd(workingDirectory)

dataDirectory <- '/home/wacax/Wacax/Kaggle/Liberty Mutual Group/Data/'

#Load external functions
source(paste0(workingDirectory, 'linearFeatureSelection.R'))

#############################
#Load Data
#Input Data
rows2read <- 20000
train <- read.csv(paste0(dataDirectory, 'train.csv'), header = TRUE, stringsAsFactors = FALSE, nrows = ifelse(class(rows2read) == 'character', -1, rows2read))
test <- read.csv(paste0(dataDirectory, 'test.csv'), header = TRUE, stringsAsFactors = FALSE, nrows = ifelse(class(rows2read) == 'character', -1, rows2read))

submissionTemplate <- read.csv(paste0(dataDirectory, 'sampleSubmission.csv'), header = TRUE, stringsAsFactors = FALSE)

################################
#DATA PREPROCESSING
#extract gini weights
weightsTrain <- train$var11
weightsTest <- test$var11

#Add a new column that indicates whether there was a fire or not
train['fire'] <- ifelse(train$target > 0, 1, 0)

#Data Transformation
train <- transform(train, var1 = as.factor(var1), var2 = as.factor(var2), var3 = as.factor(var3), 
                   var4 = as.factor(var4), var5 = as.factor(var5), var6 = as.factor(var6), var7 = as.factor(var7), 
                   var8 = as.factor(var8), var9 = as.factor(var9))
test <- transform(test, var1 = as.factor(var1), var2 = as.factor(var2), var3 = as.factor(var3), 
                  var4 = as.factor(var4), var5 = as.factor(var5), var6 = as.factor(var6), var7 = as.factor(var7), 
                  var8 = as.factor(var8), var9 = as.factor(var9))

##################################################
#EDA
#Plotting
str(train)
print(table(train$fire) / length(train$fire))
fireCosts <- as.data.frame(train$target[train$target>0]); names(fireCosts) <- 'Cost'
ggplot(data = train, aes(x = ifelse(train$target > 0, TRUE, FALSE))) +  geom_histogram() 
ggplot(data = fireCosts, aes(x = Cost)) +  geom_density() 
ggplot(data = fireCosts, aes(x = log(Cost))) +  geom_density() 

#Predictors Selection
#NA omit, regsubsets and kmeans are sensitive to NAs
noNAIndices <- which(apply(is.na(train), 1, sum) == 0)
#Fire or not
linearBestModels <- regsubsets(fire ~ ., data = train[noNAIndices, c(seq(3, 19), seq(21,303))], 
                               method = 'forward', weights = weightsTrain[noNAIndices], nvmax=200, really.big=TRUE)
bestMods <- summary(linearBestModels)
names(bestMods)
bestNumberOfPredictors <- which.min(bestMods$cp)
#ggplot of optimal number of predictors
ErrorsFireClasif <- as.data.frame(bestMods$cp); names(ErrorsFireClasif) <- 'CPError'
ggplot(data = ErrorsFireClasif, aes(x = seq(1, nrow(ErrorsFireClasif)), y = CPError)) +  geom_line() +  geom_point()
#Regular plot of optimal number of predictors
plot(bestMods$cp, xlab="Number of Variables", ylab="CP Error")
points(bestNumberOfPredictors, bestMods$cp[bestNumberOfPredictors],pch=20,col="red")

plot(linearBestModels,scale="Cp") #warning it cannot plot properly
coef(linearBestModels,10)

#Extract the name of the most predictive columns
predictors1 <- as.data.frame(bestMods$which)
repeatedNames <- sapply(names(predictors1)[2:82], function(stringX){
  return(substr(stringX, 1, 4))
})
originalPredictorNames <- c(names(predictors1)[1], repeatedNames, names(predictors1)[83:length(names(predictors1))])
predictors1 <- sort(apply(predictors1, 2, sum), decreasing = TRUE, index.return = TRUE)
predictors1 <- unique(originalPredictorNames[predictors1$ix[2:bestNumberOfPredictors]])

#Fire damage regression
whichFire <- which(train$target > 0)
linearBestModels <- regsubsets(target ~ ., data = train[intersect(noNAIndices, whichFire), c(seq(2, 19), seq(21,302))],
                               method = 'forward', weights = weightsTrain[intersect(noNAIndices, whichFire)], nvmax=100, really.big=TRUE)
bestMods <- summary(linearBestModels)
names(bestMods)
plot(bestMods$cp, xlab="Number of Variables", ylab="Cp")
bestNumberOfPredictors <- which.min(bestMods$cp)
points(bestNumberOfPredictors, bestMods$cp[bestNumberOfPredictors],pch=20,col="red")

plot(linearBestModels,scale="Cp") #warning it cannot plot properly
coef(linearBestModels,10)

#Extract the name of the most predictive columns
predictorsRegression <- as.data.frame(bestMods$which)
repeatedNames <- sapply(names(predictorsRegression)[2:82], function(stringX){
  return(substr(stringX, 1, 4))
})
originalPredictorNames <- c(names(predictorsRegression)[1], repeatedNames, names(predictorsRegression)[83:length(names(predictorsRegression))])
predictorsRegression <- sort(apply(predictorsRegression, 2, sum), decreasing = TRUE, index.return = TRUE)
predictorsRegression <- unique(originalPredictorNames[predictorsRegression$ix[2:bestNumberOfPredictors]])

#Create a predict Regsubsets Method
predict.regsubsets <- function(object,newdata,id,...){
  #TODO: Add documentation
  form <- as.formula(object$call[[2]])
  mat <- model.matrix(form,newdata)
  coefi <- coef(object,id <- id)
  mat[,names(coefi)]%*%coefi  
}

#10-fold cross-validation
set.seed(101)
folds <- sample(rep(seq(1, 10), length=length(intersect(noNAIndices, whichFire))))
table(folds)
cv.errors <- matrix(NA, 10, 11)
for(k in 1:10){
  bestFit <- regsubsets(target ~ ., data = train[intersect(noNAIndices, whichFire)[folds!=k], c(seq(2, 19), seq(21,302))],
                        method = 'forward', weights = weightsTrain[noNAIndices[folds==k]], nvmax=11, really.big=TRUE)
  for(i in 1:11){
    pred <- predict(bestFit, train[intersect(noNAIndices, whichFire)[folds==k], c(seq(2, 19), seq(21,302))], id = i)
    cv.errors[k,i] <- mean((train$target[intersect(noNAIndices, whichFire)[folds==k]] - pred)^2)
  }
}
rmse.cv=sqrt(apply(cv.errors,2,mean))
plot(rmse.cv,pch=19,type="b")

#All Data Regression
#Fire damage regression
whichFire <- which(train$target > 0)
linearBestModels <- regsubsets(target ~ ., data = train[noNAIndices, c(seq(2, 19), seq(21,302))], 
                               method = 'forward', weights = weightsTrain[noNAIndices], nvmax=100, really.big=TRUE)
bestMods <- summary(linearBestModels)
names(bestMods)
plot(bestMods$cp, xlab="Number of Variables", ylab="Cp")
bestNumberOfPredictors <- which.min(bestMods$cp)
points(bestNumberOfPredictors, bestMods$cp[bestNumberOfPredictors],pch=20,col="red")

plot(linearBestModels,scale="Cp") #warning it cannot plot properly
coef(linearBestModels,10)

#Extract the name of the most predictive columns
predictorsAllData <- as.data.frame(bestMods$which)
repeatedNames <- sapply(names(predictorsAllData)[2:82], function(stringX){
  return(substr(stringX, 1, 4))
})
originalPredictorNames <- c(names(predictorsAllData)[1], repeatedNames, names(predictorsAllData)[83:length(names(predictorsAllData))])
predictorsAllData <- sort(apply(predictorsAllData, 2, sum), decreasing = TRUE, index.return = TRUE)
predictorsAllData <- unique(originalPredictorNames[predictorsAllData$ix[2:bestNumberOfPredictors]])

#10-fold cross-validation
set.seed(101)
folds <- sample(rep(seq(1, 10), length=length(noNAIndices)))
table(folds)
cv.errors <- matrix(NA, 10, 25)
for(k in 1:10){
  bestFit <- regsubsets(target ~ ., data = train[noNAIndices[folds!=k], c(seq(2, 19), seq(21,302))],
                        method = 'forward', weights = weightsTrain[noNAIndices[folds!=k]], nvmax=25, really.big=TRUE)
  for(i in 1:25){
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

##########################################################
#MODELLING
#GBM
#Cross-validation
GBMModel <- gbm.fit(x = train[ , predictors1], y = train$fire, n.trees = 1000,
                    interaction.depth = 4, verbose = TRUE, nTrain = floor(nrow(train) * 0.7))

#Plain Regression Model for Comparison
#without weights
GBMModelRegression <- gbm(formula = target ~ ., data = train[ , c(predictorsAllData, 'target')],
                          distribution = 'gaussian', n.trees = 2000, interaction.depth = 2, 
                          verbose = TRUE, train.fraction = 0.7, cv.folds = 5, n.cores = 3)

#with weights
GBMModelRegression <- gbm(formula = target ~ ., data = train[ , c(predictorsAllData, 'target')],
                          weights = weightsTrain, distribution = 'gaussian', n.trees = 2000, 
                          interaction.depth = 2, verbose = TRUE, train.fraction = 0.7, cv.folds = 5, n.cores = 3)

summary.gbm(GBMModelRegression)
plot.gbm(GBMModelRegression)
pretty.gbm.tree(GBMModelRegression, i.tree = 1000)

#Final Model
GBMModel <- gbm.fit(x = train[ , predictors1], y = train$fire, 
                    n.trees = 2000, interaction.depth = 4, verbose = TRUE)
summary.gbm(GBMModel)
plot.gbm(GBMModel)
pretty.gbm.tree(GBMModel, i.tree = 1000)

#GLM 
#Cross-Validaton

#Final Model
GLMModel <- glm(fire ~ ., data = train[ , c(predictors1, 'fire')], family = 'binomial')

#VOWPAL WABBIT
#call vowpal wabbit function
#?????
#profit

#GLMNET
#Cross-validation
#transform train Dataframe to model matrix as glmnet only accepts matrices as input
train <- model.matrix(~ . , data = train[ , c(predictors1, 'fire')]) 
#cross validate the data, glmnet does it automatially there is no need for the caret package or a custom CV
GLMNETModelCV <- cv.glmnet(x = train[,1:dim(train)[2]-1], y = train[,dim(train)[2]], nfolds = 5, parallel = TRUE, family = 'binomial')
plot(GLMNETModelCV)
coef(GLMNETModelCV)

#Final Model
#this is not recommended by the package authors, use GLMNETModelCV$glmnet.fit instead
GLMNETModel <- glmnet(x = train[,1:dim(train)[2]-1], y = train[,dim(train)[2]], family = 'binomial', lamda = GLMNETModelCV$lambda.min) 
plot(GLMNETModel, xvar="lambda", label=TRUE)
#Recommended use
GLMNETModel <- GLMNETModelCV$glmnet.fit
plot(GLMNETModel, xvar="lambda", label=TRUE)

##########################################################
#PREDICTIONS
#plain GBM Regression
GBMRegressionPrediction <- predict(GBMModelRegression, newdata = test[ , predictors1], n.trees = 1000)
#GBM
GBMPrediction <- predict(GBMModel, newdata = test[ , predictors1], n.trees = 1000, type = 'response')
#GLM
#this removes the "factor var4 has new levels A1, Z" error
GLMModel$xlevels[['var4']] <- union(GLMModel$xlevels[['var4']], levels(test$var4))
#prediction
GLMPrediction <- predict(GLMModel, newdata = test[ , predictors1], type = 'response')
#GLMNET
GLMNETPrediction <- rep(0, 1, nrow(test))
test <- model.matrix(~ . , data = test[ , c('id', predictors1)])
PredictionMatrix <- predict(GLMNETModel, newx = test[ , 2:dim(test)[2]], type = 'response')   #it needs to be fixed, since model matrix deletes data, the test matrix ends up being incomplete
GLMNETPrediction[as.numeric(rownames(test))] <- PredictionMatrix[, match(GLMNETModelCV$lambda.min, GLMNETModel$lambda)]

#Values Regression
#GBM
fireDamageAverage <- mean(train$target[train$target > 0])
fireIndices <- sort(GBMPrediction, decreasing = TRUE, index.return = TRUE)
GBMPrediction[fireIndices$ix[1:floor(length(GBMPrediction) * 0.03)]] <- fireDamageAverage
GBMPrediction[-fireIndices$ix[1:floor(length(GBMPrediction) * 0.03)]] <- 0
#GLM

#GLMNET
fireDamageAverage <- mean(train$target[train$target > 0])
fireIndices <- sort(GLMNETPrediction, decreasing = TRUE, index.return = TRUE)
GLMNETPrediction[fireIndices$ix[1:floor(length(GLMNETPrediction) * 0.03)]] <- fireDamageAverage
GLMNETPrediction[-fireIndices$ix[1:floor(length(GLMNETPrediction) * 0.03)]] <- 0


#########################################################
#Write .csv
submissionTemplate$target <- GBMPrediction
write.csv(submissionTemplate, file = "predictionTest.csv", row.names = FALSE)