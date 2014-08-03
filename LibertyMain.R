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
require('plyr')
require('glmnet')
require('RVowpalWabbit')

#Set Working Directory
workingDirectory <- '/home/wacax/Wacax/Kaggle/Liberty Mutual Group/Liberty Mutual Group - Fire Peril Loss Cost/'
setwd(workingDirectory)

dataDirectory <- '/home/wacax/Wacax/Kaggle/Liberty Mutual Group/Data/'

#Load external functions
source(paste0(workingDirectory, 'linearFeatureSelection.R'))
source(paste0(workingDirectory, 'WeightedGini.R'))

#############################
#Load Data
#Input Data
rows2read <- 10000
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

#NA omit, regsubsets and kmeans are sensitive to NAs
noNAIndices <- which(apply(is.na(train), 1, sum) == 0)

#Clustering
#Kmeans (2 groups), The idea is to see if kmeans clustering can help explore the 
#fire vs no fire groups and if they match to some extent to the given labels
derp <- kmeans(train[noNAIndices, c('fire', 'weatherVar32', 'weatherVar33')], 2)

#PCA
derp <- princomp(train[noNAIndices, c('fire', 'weatherVar32', 'weatherVar33')])

###################################################
#Predictors Selection
#Linear Feature Selection
#Fire or No-Fire Predictors
predictors1 <- linearFeatureSelection(fire ~ ., allPredictorsData = train[, c(seq(3, 19), seq(21,303))])
predictors1 <- predictors1[[1]]
#Predictor selection using trees
treeModel <- gbm.fit(train[, c(seq(3, 19), seq(21,302))], as.factor(train$fire), distribution = 'bernoulli', nTrain = floor(nrow(train) *0.7), n.trees = 2500)
best.iter <- gbm.perf(treeModel, method="test")
GBMClassPredictors <- summary(treeModel)
GBMClassPredictors <- as.character(GBMClassPredictors$var[GBMClassPredictors$rel.inf > 1])
predictors1 <- union(predictors1, GBMClassPredictors)

#Fire damage regression predictor

whichFire <- which(train$target > 0)
predictorsRegression <- linearFeatureSelection(target ~ ., allPredictorsData = train[whichFire, c(seq(2, 19), seq(21,302))], userMax = 100)
predictorsRegression <- predictorsRegression[[1]]
#Predictor selection using trees
treeModel <- gbm.fit(train[whichFire, c(seq(3, 19), seq(21,302))], train$target[whichFire], distribution = 'gaussian', nTrain = floor(length(whichFire) *0.7), n.trees = 4000)
best.iter <- gbm.perf(treeModel, method="test")
GBMGregPredictors <- summary(treeModel)
GBMGregPredictors <- as.character(GBMGregPredictors$var[GBMGregPredictors$rel.inf > 1])
predictorsRegression <- union(predictorsRegression, GBMGregPredictors)
                     
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

#All Data Fire Damage Regression
predictorsAllData <- linearFeatureSelection(target ~ ., allPredictorsData = train[, c(seq(2, 19), seq(21,302))], userMax = 100)
predictorsAllData <- predictorsAllData[[1]]
#Predictor selection using trees
treeModel <- gbm.fit(train[, c(seq(3, 19), seq(21,302))], train$target, distribution = 'gaussian', nTrain = floor(nrow(train) *0.7), n.trees = 500)
best.iter <- gbm.perf(treeModel, method="test")
GBMAllPredictors <- summary(treeModel)
GBMAllPredictors <- as.character(GBMAllPredictors$var[GBMAllPredictors$rel.inf > 1])
predictorsAllData <- union(predictorsAllData, GBMAllPredictors)

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

##########################################################
#MODELLING
#GBM
#Cross-validation
GBMModel <- gbm.fit(x = train[ , predictors1], y = train$fire, n.trees = 1000,
                    interaction.depth = 4, verbose = TRUE, nTrain = floor(nrow(train) * 0.7), distribution = 'bernoulli')
#Competition Scores
NormalizedWeightedGini <- function(solution, weights, submission) {
  WeightedGini(solution, weights, submission) / WeightedGini(solution, weights, solution)
}

#Cross-validation
#Add a new column loss or not as factor
train['lossFactor'] <- as.factor(ifelse(train$target > 0, 1, 0))

GBMControl <- trainControl(method="cv",
                           number=5,
                           summaryFunction = twoClassSummary,
                           verboseIter=TRUE)

gbmGrid <- expand.grid(.distribution =c('bernoulli', 'adaboost'),
                       .interaction.depth = seq(1, 7, 3),
                       .shrinkage = c(0.001, 0.003, 0.01, 0,03, 0.1), 
                       .n.trees = 1000)

gbmMOD <- train(form = lossFactor ~ ., 
                data = train[ , c(predictors1, 'lossFactor')],
                method = "gbm",
                tuneGrid = gbmGrid,
                trControl = GBMControl,
                verbose = TRUE)

#Final Model
#Fire - No Fire Model
GBMModel <- gbm.fit(x = train[ , predictors1], y = train$fire, distribution = 'bernoulli',
                    n.trees = 1000, verbose = TRUE)
summary.gbm(GBMModel)
plot.gbm(GBMModel)
pretty.gbm.tree(GBMModel, i.tree = 1000)

#Value Regression
#5 Fold Cross-Validation + best distribution
GBMControl <- trainControl(method="cv",
                           number=5,
                           summaryFunction = twoClassSummary,
                           verboseIter=TRUE)

gbmGrid <- expand.grid(.interaction.depth = seq(1, 7, 3),
                       .shrinkage = c(0.001, 0.003, 0.01, 0,03, 0.1), 
                       .n.trees = 1000)

gbmMOD <- train(form = target ~ ., 
                data = train[whichFire , c(predictorsRegression, 'target')],
                method = "gbm",
                tuneGrid = gbmGrid,
                trControl = GBMControl,
                distribution = 'gaussian',
                verbose = TRUE)

#Final Model
whichFire <- which(train$target > 0)
GBMModelReg <- gbm.fit(x = train[whichFire , predictorsRegression], y = train$target[whichFire], distribution = 'gaussian',
                       n.trees = 4000, verbose = TRUE)
summary.gbm(GBMModel)
plot.gbm(GBMModel)
pretty.gbm.tree(GBMModel, i.tree = 1000)

#Full Data Value Regression
#5 Fold Cross-Validation + best distribution
GBMControl <- trainControl(method="cv",
                           number=5,
                           summaryFunction = twoClassSummary,
                           verboseIter=TRUE)

gbmGrid <- expand.grid(.interaction.depth = seq(1, 7, 3),
                       .shrinkage = c(0.001, 0.003, 0.01, 0,03, 0.1), 
                       .n.trees = 1000)

gbmMOD <- train(form = target ~ ., 
                data = train[ , c(predictorsAllData, 'target')],
                method = "gbm",
                tuneGrid = gbmGrid,
                trControl = GBMControl,
                distribution = 'gaussian',
                verbose = TRUE)

#Final Model
whichFire <- which(train$target > 0)
GBMModelReg <- gbm.fit(x = train[ , predictorsAllData], y = train$target, distribution = 'gaussian',
                       n.trees = 4000, verbose = TRUE)
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
#Classification fire or no fire

#Cross-validation
#transform train Dataframe to model matrix as glmnet only accepts matrices as input
trainClassMatrix <- model.matrix(~ . , data = train[ , c(predictors1, 'fire')]) 
#cross validate the data, glmnet does it automatially there is no need for the caret package or a custom CV
GLMNETModelCV <- cv.glmnet(x = trainClassMatrix[,1:dim(trainClassMatrix)[2]-1], y = trainClassMatrix[,dim(trainClassMatrix)[2]], nfolds = 5, parallel = TRUE, family = 'binomial')
plot(GLMNETModelCV)
coef(GLMNETModelCV)

#Final Model
#this is not recommended by the package authors, use GLMNETModelCV$glmnet.fit instead
#GLMNETModel <- glmnet(x = trainClassMatrix[,1:dim(trainClassMatrix)[2]-1], y = trainClassMatrix[,dim(trainClassMatrix)[2]], family = 'binomial', lamda = GLMNETModelCV$lambda.min) 
#plot(GLMNETModel, xvar="lambda", label=TRUE)

#Recommended use
GLMNETModel <- GLMNETModelCV$glmnet.fit
plot(GLMNETModel, xvar="lambda", label=TRUE)

#Regression

#Cross-validation
#Find regression targets
whichFire <- which(train$target > 0)
#transform train Dataframe to model matrix as glmnet only accepts matrices as input
trainRegMatrix <- model.matrix(~ . , data = train[whichFire , c(predictorsRegression, 'target')]) 
#cross validate the data, glmnet does it automatially there is no need for the caret package or a custom CV
GLMNETModelCVReg <- cv.glmnet(x = trainRegMatrix[,1:dim(trainRegMatrix)[2]-1], y = trainRegMatrix[,dim(trainRegMatrix)[2]], nfolds = 5, parallel = TRUE, family = 'gaussian')
plot(GLMNETModelCV)
coef(GLMNETModelCV)

#Final Model
#this is not recommended by the package authors, use GLMNETModelCV$glmnet.fit instead
#GLMNETModel <- glmnet(x = train[,1:dim(train)[2]-1], y = train[,dim(train)[2]], family = 'binomial', lamda = GLMNETModelCV$lambda.min) 
#plot(GLMNETModel, xvar="lambda", label=TRUE)

#Recommended use
GLMNETModelReg <- GLMNETModelCVReg$glmnet.fit
plot(GLMNETModelReg, xvar="lambda", label=TRUE)

##########################################################
#PREDICTIONS
#plain GBM Regression
GBMRegressionPrediction <- predict(GBMModelRegression, newdata = test[ , predictors1], n.trees = 1000)
#GBM
#Fire-No Fire Prediction
GBMPrediction <- predict(GBMModel, newdata = test[ , predictors1], n.trees = 1000, type = 'response')
#Value Regression Prediction
GBMPredictionReg <- predict(GBMModelReg, newdata = test[ , predictorsRegression], n.trees = 2800)
GBMPrediction <- GBMPrediction * GBMPredictionReg
#GLM
#this removes the "factor var4 has new levels A1, Z" error
GLMModel$xlevels[['var4']] <- union(GLMModel$xlevels[['var4']], levels(test$var4))
#prediction
GLMPrediction <- predict(GLMModel, newdata = test[ , predictors1], type = 'response')

#GLMNET
#Classification
GLMNETPrediction <- rep(0, 1, nrow(test))
testMatrix <- model.matrix(~ . , data = test[ , c('id', predictors1)])
PredictionMatrix <- predict(GLMNETModel, newx = testMatrix[ , 2:dim(testMatrix)[2]], type = 'response')   #it needs to be fixed, since model matrix deletes data, the test matrix ends up being incomplete
GLMNETPrediction[as.numeric(rownames(testMatrix))] <- PredictionMatrix[, match(GLMNETModelCV$lambda.min, GLMNETModel$lambda)]

#Regresssion

#Values Regression
#GBM
fireDamageAverage <- mean(train$target[train$target > 0])
fireIndices <- sort(GBMPrediction, decreasing = TRUE, index.return = TRUE)
GBMPrediction[fireIndices$ix[1:floor(length(GBMPrediction) * 0.03)]] <- fireDamageAverage
GBMPrediction[-fireIndices$ix[1:floor(length(GBMPrediction) * 0.03)]] <- 0
#GLM



#########################################################
#Write .csv
submissionTemplate$target <- GBMPrediction
write.csv(submissionTemplate, file = "predictionTestII.csv", row.names = FALSE)