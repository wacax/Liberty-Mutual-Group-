#Liberty Mutual Group - Fire Peril Loss Cost
#ver 0.10

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
source(paste0(workingDirectory, 'csv2vw.R'))

#############################
#Load Data
#Input Data
rows2read <- 'all'
train <- read.csv(paste0(dataDirectory, 'train.csv'), header = TRUE, stringsAsFactors = FALSE, nrows = ifelse(class(rows2read) == 'character', -1, rows2read))
test <- read.csv(paste0(dataDirectory, 'test.csv'), header = TRUE, stringsAsFactors = FALSE, nrows = ifelse(class(rows2read) == 'character', -1, rows2read))

submissionTemplate <- read.csv(paste0(dataDirectory, 'sampleSubmission.csv'), header = TRUE, stringsAsFactors = FALSE)

################################
#DATA PREPROCESSING
#remove NAs using mean normalization
print(paste0('There are ', length(which(apply(is.na(train), 1, sum) > 0)), ' NA rows in the data'))
colsWithNAs <- apply(train, 2, function(column){return(sum(is.na(column)))})
#determine numeric features
numericIdx <- names(train)[sort(intersect(union(which(sapply(train, class) == 'numeric'), 
                                         which(sapply(train, class) == 'integer')),
                                   which(colsWithNAs > 0)))]
means2Subtract <- colMeans(train[ , numericIdx], na.rm = TRUE)

#Center the training data and replace NAs with means
for(i in 1:length(numericIdx)){
  naIdxs <- is.na(train[ , numericIdx[i]])
  train[ , numericIdx[i]] <- as.numeric(train[ , numericIdx[i]] - means2Subtract[i])
  train[naIdxs , numericIdx[i]] <- 0
  print(paste0(numericIdx[i], ' Column Normalized'))
}

NAsPerColumn <- apply(train, 2, function(column){return(sum(is.na(column)))})
print(paste0('There are ', length(which(apply(is.na(train), 2, sum) > 0)), ' NA rows after normalization'))

#Center the test data and replace NAs with training data 
for(i in 1:length(numericIdx)){
  naIdxs <- is.na(test[ , numericIdx[i]])
  if(sum(naIdxs == 0)){test[ , numericIdx[i]] <- as.numeric(test[ , numericIdx[i]] - means2Subtract[i])
                       test[naIdxs , numericIdx[i]] <- 0}
  print(paste0(numericIdx[i], ' Column Normalized'))
}

NAsPerColumn <- apply(test, 2, function(column){return(sum(is.na(column)))})
print(paste0('There are ', length(which(apply(is.na(test), 2, sum) > 0)), ' NA rows after normalization'))


#------------------------------------------------
#extract gini weights
weightsTrain <- train$var11
weightsTest <- test$var11

#NA indices, regsubsets and kmeans are sensitive to NAs
noNAIndices <- which(apply(is.na(train), 1, sum) == 0)

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
predictors1 <- linearFeatureSelection(fire ~ ., train[, c(seq(3, 12), seq(14, 19), seq(21,303))])
predictors1 <- predictors1[[1]]
#Predictor selection using trees
treeModel <- gbm(fire ~ ., train[, c(seq(3, 12), seq(14, 19), seq(21,303))], distribution = 'adaboost',
                 train.fraction = 0.7, n.trees = 1000, weights = weightsTrain, verbose = TRUE, 
                 cv.folds = 3, n.cores = 3)
best.iter <- gbm.perf(treeModel, method="test")
GBMClassPredictors <- summary(treeModel)
GBMClassPredictors <- as.character(GBMClassPredictors$var[GBMClassPredictors$rel.inf > 1])
predictors1 <- union(predictors1, GBMClassPredictors)

#Fire damage regression predictor
whichFire <- which(train$target > 0)
noNAIndices <- intersect(noNAIndices, whichFire)
predictorsRegression <- linearFeatureSelection(target ~ ., train[whichFire, c(seq(2, 12), seq(14, 19), seq(21,302))], userMax = 100)
predictorsRegression <- predictorsRegression[[1]]
#Predictor selection using trees
treeModel <- gbm(target ~ ., train[whichFire, c(seq(2, 12), seq(14, 19), seq(21,302))], distribution = 'gaussian', 
                 train.fraction = 0.7, n.trees = 1000, weights = weightsTrain[whichFire], verbose = TRUE, 
                 cv.folds = 3, n.cores = 3)
best.iter <- gbm.perf(treeModel, method="test")
GBMRegPredictors <- summary(treeModel)
GBMRegPredictors <- as.character(GBMRegPredictors$var[GBMRegPredictors$rel.inf > 1])
predictorsRegression <- union(predictorsRegression, GBMRegPredictors)
                     
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
noNAIndices <- which(apply(is.na(train), 1, sum) == 0)
predictorsAllData <- linearFeatureSelection(target ~ ., train[, c(seq(2, 12), seq(14, 19), seq(21,302))], userMax = 100)
predictorsAllData <- predictorsAllData[[1]]
#Predictor selection using trees
treeModel <- gbm(target ~ ., train[, c(seq(2, 12), seq(14, 19), seq(21,302))], distribution = 'gaussian', 
                 train.fraction = 0.7, n.trees = 1000, weights = weightsTrain, verbose = TRUE, 
                 cv.folds = 3, n.cores = 3)
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
#Add a new column loss or not as factor
train['lossFactor'] <- as.factor(ifelse(train$target > 0, 1, 0))

GBMControl <- trainControl(method = "cv",
                           number = 5,
                           verboseIter = TRUE,
                           classProbs = TRUE)

gbmGrid <- expand.grid(.interaction.depth = seq(1, 5, 2),
                       .shrinkage = c(0.001, 0.003), 
                       .n.trees = 2500)

gbmMODClass <- train(form = lossFactor ~ ., 
                     data = train[noNAIndices , c(predictors1, 'lossFactor')],
                     method = "gbm",
                     metric = "ROC",
                     tuneGrid = gbmGrid,
                     trControl = GBMControl,
                     distribution = 'adaboost',
                     weights = weightsTrain[noNAIndices],
                     train.fraction = 0.7,
                     verbose = TRUE)

plot(gbmMODClass)
#Best Number of trees
treesClass <- gbm.perf(gbmMODClass$finalModel, method = 'test')
treesIterated <- max(gbmGrid$.n.trees)
gbmMODClassExpanded <- gbmMODClass$finalModel

while(treesClass >= treesIterated - 20){
  # do another 5000 iterations  
  gbmMODClassExpanded <- gbm.more(gbmMODClassExpanded, max(gbmGrid$.n.trees),
                                  data = train[noNAIndices , c(predictors1, 'lossFactor')],
                                  weights = weightsTrain[noNAIndices],
                                  verbose=TRUE)
  treesClass <- gbm.perf(gbmMODClassExpanded, method = 'test')
  treesIterated <- treesIterated + max(gbmGrid$.n.trees)
    
  if(treesIterated >= 50000){break}  
}

gbmMODClass$finalModel <- gbmMODClassExpanded

#Final Model
#Add a new column loss or not as factor
train['lossFactor'] <- as.factor(ifelse(train$target > 0, 1, 0))
#Loss - No Loss Model
GBMModel <- gbm(lossFactor ~ ., data = train[noNAIndices , c(predictors1, 'lossFactor')], distribution = 'adaboost',
                weights = weightsTrain[noNAIndices],
                interaction.depth = gbmMODClass$bestTune[2], shrinkage = gbmMODClass$bestTune[3], 
                n.trees = treesClass, verbose = TRUE)
summary.gbm(GBMModel)
plot.gbm(GBMModel)
pretty.gbm.tree(GBMModel, i.tree = treesClass)

#Value Regression
#5 Fold Cross-Validation 
whichFire <- which(train$target > 0)
GBMControl <- trainControl(method="cv",
                           number=5,
                           verboseIter=TRUE)

gbmGrid <- expand.grid(.interaction.depth = seq(1, 16, 3),
                       .shrinkage = c(0.001, 0.003), 
                       .n.trees = 4000)

gbmMODReg <- train(form = target ~ ., 
                   data = train[whichFire , c(predictorsRegression, 'target')],
                   method = "gbm",
                   tuneGrid = gbmGrid,
                   trControl = GBMControl,
                   distribution = 'gaussian',
                   weights = weightsTrain[whichFire],
                   train.fraction = 0.7,
                   verbose = TRUE)

#Best Number of trees
treesReg <- gbm.perf(gbmMODReg$finalModel, method = 'test')
#Final Model
GBMModelReg <- gbm(target ~ ., data = train[whichFire, c(predictorsRegression, 'target')],
                   distribution = 'gaussian',
                   weights = weightsTrain[whichFire],
                   interaction.depth = gbmMODReg$bestTune[2], shrinkage = gbmMODReg$bestTune[3], 
                   n.trees = treesReg, verbose = TRUE)
summary.gbm(GBMModelReg)
plot.gbm(GBMModelReg)
pretty.gbm.tree(GBMModelReg, i.tree = treesReg)

#Full Data Value Regression
#5 Fold Cross-Validation + best distribution
GBMControl <- trainControl(method="cv",
                           number=5,
                           verboseIter=TRUE)

gbmGrid <- expand.grid(.interaction.depth = seq(1, 7, 3),
                       .shrinkage = c(0.001, 0.003, 0.01), 
                       .n.trees = 2500)

gbmMODAll <- train(form = target ~ ., 
                   data = train[noNAIndices , c(predictorsAllData, 'target')],
                   method = "gbm",
                   tuneGrid = gbmGrid,
                   trControl = GBMControl,
                   distribution = 'gaussian',
                   weights = weightsTrain[noNAIndices],
                   train.fraction = 0.7,
                   verbose = TRUE)

#Best Number of trees
treesAll <- gbm.perf(gbmMODAll$finalModel, method = 'test')

#Final Model
GBMModelAll <- gbm.fit(x = train[noNAIndices , predictorsAllData], y = train$target, 
                       distribution = 'gaussian', weights = weightsTrain[noNAIndices],
                       interaction.depth = gbmMODAll$bestTune[2],
                       shrinkage = gbmMODAll$bestTune[3], n.trees = treesAll, verbose = TRUE)
summary.gbm(GBMModelAll)
plot.gbm(GBMModelAll)
pretty.gbm.tree(GBMModelAll, i.tree = treesAll)

#Ensemble of Models
trainGBMClass <- predict(GBMModel, newdata = train[ , predictors1], n.trees = treesClass, type = 'response')
trainGBMReg <- predict(GBMModelReg, newdata = train[ , predictorsRegression], n.trees = treesReg)
trainGBMAll <- predict(GBMModelAll, newdata = train[ , predictorsAllData], n.trees = treesAll)

#train responses data frame
GBMTrainPredictionsTwoPredictors <- data.frame(trainGBMClass, trainGBMReg, train$target)
names(GBMTrainPredictionsTwoPredictors)[3] <- 'target'
GBMTrainPredictionsFull <- data.frame(trainGBMClass, trainGBMReg, trainGBMAll, train$target)

#Build Ensembles
GBMControl <- trainControl(method="cv",
                           number=5,
                           verboseIter=TRUE)

gbmGrid <- expand.grid(.interaction.depth = seq(1, 7, 3),
                       .shrinkage = c(0.001, 0.003, 0.01), 
                       .n.trees = 3500)

gbmMODEnsembleTwo <- train(form = target ~ ., 
                           data = GBMTrainPredictionsTwoPredictors,
                           method = "gbm",
                           tuneGrid = gbmGrid,
                           trControl = GBMControl,
                           distribution = 'gaussian',
                           train.fraction = 0.7,
                           verbose = TRUE)

#Best Number of trees
treesEnsembleTwo <- gbm.perf(gbmMODEnsembleTwo$finalModel, method = 'test')

GBMEnsembleTwo <- gbm.fit(x = GBMTrainPredictionsTwoPredictors[, c('trainGBMClass', 'trainGBMReg')],
                          y = GBMTrainPredictionsTwoPredictors$target,
                          distribution = 'gaussian', interaction.depth = gbmMODEnsembleTwo$bestTune[2],
                          shrinkage = gbmMODEnsembleTwo$bestTune[3],
                          n.trees = treesEnsembleTwo, verbose = TRUE)

GBMEnsembleFull <- gbm.fit(x = GBMTrainPredictionsFull[, c('trainGBMClass', 'trainGBMReg', 'trainGBMAll')],
                           y = GBMTrainPredictionsFull$target, distribution = 'gaussian', 
                           interaction.depth = 1, shrinkage = 0.001, n.trees = 3000, verbose = TRUE)

#Competition Scores GBM
NormalizedWeightedGini <- function(solution, weights, submission) {
  WeightedGini(solution, weights, submission) / WeightedGini(solution, weights, solution)
}

#GLM 
#Cross-Validaton

#Final Model
GLMModel <- glm(fire ~ ., data = train[ , c(predictors1, 'fire')], family = 'binomial')

#VOWPAL WABBIT
#transform csv data into vw readable data
vwOutputDir <- paste0(dataDirectory)
csv2vw(train, 'target', 'weight', 'Id', outputFileDir = paste0(vwOutputDir, 'trainvw.txt'))

#?????
## change to directory of data we just created
setwd(vwOutputDir)

# Test 3: without -d, training only
# {VW} train-sets/0002.dat    -f models/0002.model
test3 <- c("-t", "train-sets/0002.dat",
           "-f", "models/0002.model")

res <- vw(test3)
res

#profit

#GLMNET
#Classification loss or no loss due to fire

#Cross-validation
#transform train Dataframe to model matrix as glmnet only accepts matrices as input
trainClassMatrix <- model.matrix(~ . , data = train[ , c(predictors1, 'fire', 'var11')]) 
#cross validate the data, glmnet does it automatially there is no need for the caret package or a custom CV
GLMNETModelCV <- cv.glmnet(x = trainClassMatrix[,1:(dim(trainClassMatrix)[2]-2)], 
                           y = trainClassMatrix[,dim(trainClassMatrix)[2]-1], 
                           nfolds = 5, parallel = TRUE, family = 'binomial', 
                           weights = trainClassMatrix[,ncol(trainClassMatrix)])
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
trainRegMatrix <- model.matrix(~ . , data = train[whichFire , c(predictorsRegression, 'target', 'var11')]) 
#cross validate the data, glmnet does it automatially there is no need for the caret package or a custom CV
GLMNETModelCVReg <- cv.glmnet(x = trainRegMatrix[,1:(dim(trainRegMatrix)[2]-2)], 
                              y = trainRegMatrix[,dim(trainRegMatrix)[2]-1], nfolds = 5,
                              parallel = TRUE, family = 'gaussian', 
                              weights = trainRegMatrix[,ncol(trainRegMatrix)])
plot(GLMNETModelCVReg)
coef(GLMNETModelCVReg)

#Final Model
#this is not recommended by the package authors, use GLMNETModelCV$glmnet.fit instead
#GLMNETModel <- glmnet(x = train[,1:dim(train)[2]-1], y = train[,dim(train)[2]], family = 'binomial', lamda = GLMNETModelCV$lambda.min) 
#plot(GLMNETModel, xvar="lambda", label=TRUE)

#Recommended use
GLMNETModelReg <- GLMNETModelCVReg$glmnet.fit
plot(GLMNETModelReg, xvar="lambda", label=TRUE)

#All Data
#Cross-validation
#transform train Dataframe to model matrix as glmnet only accepts matrices as input
trainAllMatrix <- model.matrix(~ . , data = train[ , c(predictorsAllData, 'target', 'var11')]) 
#cross validate the data, glmnet does it automatially there is no need for the caret package or a custom CV
GLMNETModelCVAll <- cv.glmnet(x = trainAllMatrix[,1:(dim(trainAllMatrix)[2]-2)],
                              y = trainAllMatrix[,dim(trainAllMatrix)[2]-1], nfolds = 5,
                              parallel = TRUE, family = 'gaussian', 
                              weights = trainAllMatrix[, ncol(trainAllMatrix)])
plot(GLMNETModelCVAll)
coef(GLMNETModelCVAll)

#Final Model
#this is not recommended by the package authors, use GLMNETModelCVAll$glmnet.fit instead
#GLMNETModel <- glmnet(x = train[,1:dim(train)[2]-1], y = train[,dim(train)[2]], family = 'binomial', lamda = GLMNETModelCV$lambda.min) 
#plot(GLMNETModel, xvar="lambda", label=TRUE)

#Recommended use
GLMNETModelAll <- GLMNETModelCVAll$glmnet.fit
plot(GLMNETModelAll, xvar="lambda", label=TRUE)

##########################################################
#PREDICTIONS
#GBM
#Loss - no Loss to Fire Prediction
GBMPrediction <- predict(object = GBMModel, newdata = test[ , predictors1], n.trees = treesClass)
#Value Regression Prediction
GBMPredictionReg <- predict(GBMModelReg, newdata = test[ , predictorsRegression], n.trees = treesReg)
#All Data Regression Prediction
GBMPredictionAll <- predict(GBMModelAll, newdata = test[ , predictorsAllData], n.trees = treesAll)

#2-3 models Ensembles
#Simple combinations
#Classification - Regression
GBMPredReg <- GBMPrediction * GBMPredictionReg
#Classification - All
GBMPredAll <- GBMPrediction * GBMPredictionAll
#Regression - All
GBMPredAll <- GBMPredictionReg * GBMPredictionAll

#GBM of ensemble of Models, the final result will be compared to the targets columns in the train dataframe data
GBMTestPredictions <- data.frame(cbind(GBMPrediction, GBMPredictionReg, GBMPredictionAll))

ensamblePredictions <- predict(GBMEnsemble, newdata = GBMTestPredictions, n.trees = 3000)



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

#Regression
GLMNETPredictionReg <- rep(0, 1, nrow(test))
testMatrixReg <- model.matrix(~ . , data = test[ , c('id', predictorsRegression)])
PredictionMatrix <- predict(GLMNETModelReg, newx = testMatrixReg[ , 2:dim(testMatrixReg)[2]])   #it needs to be fixed, since model matrix deletes data, the test matrix ends up being incomplete
GLMNETPredictionReg[as.numeric(rownames(testMatrixReg))] <- PredictionMatrix[, match(GLMNETModelCVReg$lambda.min, GLMNETModelCVReg$lambda)]

#All Data
GLMNETPredictionAll <- rep(0, 1, nrow(test))
testMatrixAll <- model.matrix(~ . , data = test[ , c('id', predictorsAllData)])
PredictionMatrix <- predict(GLMNETModelAll, newx = testMatrixAll[ , 2:dim(testMatrixAll)[2]])   #it needs to be fixed, since model matrix deletes data, the test matrix ends up being incomplete
GLMNETPredictionAll[as.numeric(rownames(testMatrixAll))] <- PredictionMatrix[, match(GLMNETModelCVAll$lambda.min, GLMNETModelCVAll$lambda)]

#Values Regression
#GBM
fireDamageAverage <- mean(train$target[train$target > 0])
fireIndices <- sort(GBMPrediction, decreasing = TRUE, index.return = TRUE)
GBMPrediction[fireIndices$ix[1:floor(length(GBMPrediction) * 0.03)]] <- fireDamageAverage
GBMPrediction[-fireIndices$ix[1:floor(length(GBMPrediction) * 0.03)]] <- 0
#GLM



#########################################################
#Write .csv multiplication
submissionTemplate$target <- GBMPredReg
write.csv(submissionTemplate, file = "predictionII.csv", row.names = FALSE)

#Write .csv ensemble
submissionTemplate$target <- ensamblePredictions
write.csv(submissionTemplate, file = "predictionTestII.csv", row.names = FALSE)