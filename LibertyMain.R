#Liberty Mutual Group - Fire Peril Loss Cost
#ver 0.10

#########################
#Init
rm(list=ls(all=TRUE))

#Load/install required libraries
require('data.table')
require('ggplot2')
require('leaps')
require('rpart')
require('caret')
require('gbm')
require('parallel')
require('foreach')
require('plyr')
require('doParallel')
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
noBootstrapingSapmles <- 5
## "pre-allocate" an empty list of length 5
linearClassPredictorsDF <- vector("list", noBootstrapingSapmles)
#Loss or No-Loss Predictors
for (i in 1:noBootstrapingSapmles){
  randomSubset <- sample.int(nrow(train), 150000)
  linearClassPredictors <- linearFeatureSelection(fire ~ ., train[randomSubset, c(seq(3, 19), seq(21,303))])
  linearClassPredictorsDF[i] <- list(linearClassPredictors[[1]])[1:floor(length(linearClassPredictors[[1]]) * 0.7)]
}
linearClassPredictors <- character()
for(i in 1:length(linearClassPredictorsDF)){
  linearClassPredictors <- union(linearClassPredictors, linearClassPredictorsDF[[i]])
}
#Predictor selection using trees
randomSubset <- sample.int(nrow(train), nrow(train)) #use this to use full data
treeModel <- gbm.fit(x = train[randomSubset, c(seq(3, 19), seq(21,302))], y = train$fire[randomSubset],
                     distribution = 'adaboost', nTrain = floor(nrow(train) * 0.7), n.trees = 500, verbose = TRUE)
best.iter <- gbm.perf(treeModel, method = "test")
GBMClassPredictors <- summary(treeModel)
GBMClassPredictors <- as.character(GBMClassPredictors$var[GBMClassPredictors$rel.inf > 1])
allClassPredictors <- union(linearClassPredictors, GBMClassPredictors)

#Fire damage regression predictor
whichLoss <- which(train$target > 0)
linearRegPredictors <- linearFeatureSelection(target ~ ., train[whichLoss, c(seq(2, 19), seq(21,302))], userMax = 100)
linearRegPredictors <- linearRegPredictors[[1]]
#Predictor selection using trees
treeModel <- gbm.fit(x = train[whichLoss, c(seq(3, 19), seq(21,302))], y = train$target[whichLoss],
                     distribution = 'gaussian', nTrain = floor(nrow(train) * 0.7), n.trees = 1500, verbose = TRUE)
best.iter <- gbm.perf(treeModel, method="test")
GBMRegPredictors <- summary(treeModel)
GBMRegPredictors <- as.character(GBMRegPredictors$var[GBMRegPredictors$rel.inf > 0.5])
allPredictorsReg <- union(linearRegPredictors, GBMRegPredictors)

#All Data Fire Damage Regression
noBootstrapingSapmles <- 5
## "pre-allocate" an empty list of length 5
linearFullPredictorsDF <- vector("list", noBootstrapingSapmles)
#Loss value regression all data
for (i in 1:noBootstrapingSapmles){
  randomSubset <- sample.int(nrow(train), 150000)
  linearPredictorsFull <- linearFeatureSelection(target ~ ., train[randomSubset, c(seq(2, 19), seq(21,302))])
  linearFullPredictorsDF[i] <- list(linearPredictorsFull[[1]])[1:floor(length(linearPredictorsFull[[1]]) * 0.7)]
}
linearPredictorsFull <- character()
for(i in 1:length(linearFullPredictorsDF)){
  linearPredictorsFull <- union(linearPredictorsFull, linearFullPredictorsDF[[i]])
}
#Predictor selection using trees
randomSubset <- sample.int(nrow(train), nrow(train)) #use this to use full data
treeModel <- gbm.fit(x = train[randomSubset, c(seq(3, 19), seq(21,302))], y = train$target[randomSubset],
                     distribution = 'gaussian', nTrain = floor(nrow(train) * 0.7), n.trees = 500, verbose = TRUE)
best.iter <- gbm.perf(treeModel, method="test")
GBMAllPredictors <- summary(treeModel)
GBMAllPredictors <- as.character(GBMAllPredictors$var[GBMAllPredictors$rel.inf > 0.5])
allPredictorsFull <- union(linearPredictorsFull, GBMAllPredictors)

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
                       .n.trees = 500)

randomSubset <- sample.int(nrow(train), nrow(train)) #use this to use full data
gbmMODClass <- train(form = lossFactor ~ ., 
                     data = train[randomSubset , c(GBMClassPredictors, 'lossFactor')],
                     method = "gbm",
                     metric = "ROC",
                     tuneGrid = gbmGrid,
                     trControl = GBMControl,
                     distribution = 'adaboost',
                     train.fraction = 0.7,
                     verbose = TRUE)

#Plot Cross Validation
ggplot(gbmMODClass)  + theme(legend.position = "top") 
confusionMatrix(gbmPred, churnTest$churn)
#Best Number of trees
treesClass <- gbm.perf(gbmMODClass$finalModel, method = 'test')
treesIterated <- max(gbmGrid$.n.trees)
gbmMODClassExpanded <- gbmMODClass$finalModel

while(treesClass >= treesIterated - 20){
  # do another 5000 iterations  
  gbmMODClassExpanded <- gbm.more(gbmMODClassExpanded, max(gbmGrid$.n.trees),
                                  data = train[randomSubset , c(GBMClassPredictors, 'lossFactor')],
                                  verbose=TRUE)
  treesClass <- gbm.perf(gbmMODClassExpanded, method = 'test')
  treesIterated <- treesIterated + max(gbmGrid$.n.trees)
  
  if(treesIterated >= 50000){break}  
}

gbmMODClass$finalModel <- gbmMODClassExpanded

#Final Model
#Loss - No Loss Model
GBMModel <- gbm.fit(x = train[randomSubset , GBMClassPredictors], y = train$lossFactor,
                    distribution = 'adaboost',
                    interaction.depth = gbmMODClass$bestTune[2], shrinkage = gbmMODClass$bestTune[3], 
                    n.trees = treesClass, verbose = TRUE)
summary.gbm(GBMModel)
plot.gbm(GBMModel)
pretty.gbm.tree(GBMModel, i.tree = treesClass)

#Value Regression
#5 Fold Cross-Validation 
whichFire <- which(train$target > 0)
GBMControl <- trainControl(method = "cv",
                           number = 5,
                           verboseIter = TRUE)

gbmGrid <- expand.grid(.interaction.depth = seq(1, 16, 3),
                       .shrinkage = c(0.001, 0.003), 
                       .n.trees = 4000)

gbmMODReg <- train(form = target ~ ., 
                   data = train[whichFire , c(GBMRegPredictors, 'target')],
                   method = "gbm",
                   tuneGrid = gbmGrid,
                   trControl = GBMControl,
                   distribution = 'gaussian',
                   train.fraction = 0.7,
                   verbose = TRUE)

#Best Number of trees
treesReg <- gbm.perf(gbmMODReg$finalModel, method = 'test')
#Final Model
GBMModelReg <- gbm.fit(x = train[whichFire, GBMRegPredictors], y = train[, 'target'],
                       distribution = 'gaussian',
                       interaction.depth = gbmMODReg$bestTune[2], shrinkage = gbmMODReg$bestTune[3], 
                       n.trees = treesReg, verbose = TRUE)
summary.gbm(GBMModelReg)
plot.gbm(GBMModelReg)
pretty.gbm.tree(GBMModelReg, i.tree = treesReg)

#Full Data Value Regression
#5 Fold Cross-Validation + best distribution
GBMControl <- trainControl(method = "cv",
                           number = 5,
                           verboseIter = TRUE)

gbmGrid <- expand.grid(.interaction.depth = seq(1, 7, 3),
                       .shrinkage = c(0.001, 0.003, 0.01), 
                       .n.trees = 500)

randomSubset <- sample.int(nrow(train), nrow(train)) #use this to use full data
gbmMODAll <- train(form = target ~ ., 
                   data = train[randomSubset , c(GBMAllPredictors, 'target')],
                   method = "gbm",
                   tuneGrid = gbmGrid,
                   trControl = GBMControl,
                   distribution = 'gaussian',
                   weights = weightsTrain[randomSubset],
                   train.fraction = 0.7,
                   verbose = TRUE)

#Best Number of trees
treesAll <- gbm.perf(gbmMODAll$finalModel, method = 'test')
#Final Model
GBMModelAll <- gbm.fit(x = train[randomSubset , GBMAllPredictors], y = train$target[randomSubset], 
                       distribution = 'gaussian', interaction.depth = gbmMODAll$bestTune[2],
                       shrinkage = gbmMODAll$bestTune[3], n.trees = treesAll, verbose = TRUE)
summary.gbm(GBMModelAll)
plot.gbm(GBMModelAll)
pretty.gbm.tree(GBMModelAll, i.tree = treesAll)

#Ensemble of Models
trainGBMClass <- predict(GBMModel, newdata = train[ , GBMClassPredictors], n.trees = treesClass, type = 'response')
trainGBMReg <- predict(GBMModelReg, newdata = train[ , GBMRegPredictors], n.trees = treesReg)
trainGBMAll <- predict(GBMModelAll, newdata = train[ , GBMAllPredictors], n.trees = treesAll)

#GLMNET Extra features
trainClassReg <- trainGBMClass * trainGBMReg
trainClassAll <- trainGBMClass * trainGBMAll
trainRegAll <- trainGBMReg * trainGBMAll

#train responses data frame
GBMTrainPredictions <- data.frame(trainGBMClass, trainGBMReg, trainGBMAll, 
                                  trainClassReg, trainClassAll, trainRegAll)
names(GBMTrainPredictions) <- c('GBMClass', 'GBMReg', 'GBMAll',
                                'ClassReg', 'ClassAll', 'RegAll')

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
randomSubset <- sample.int(nrow(train), nrow(train)) #use this to use full data
#transform train Dataframe to model matrix as glmnet only accepts matrices as input
trainClassMatrix <- model.matrix(~ . , data = train[randomSubset , c(linearClassPredictors, 'lossFactor')]) 
#cross validate the data, glmnet does it automatially there is no need for the caret package or a custom CV
registerDoParallel(detectCores() - 1)
GLMNETModelCV <- cv.glmnet(x = trainClassMatrix[,1:(dim(trainClassMatrix)[2]-1)], 
                           y = trainClassMatrix[,dim(trainClassMatrix)[2]], 
                           nfolds = 5, parallel = TRUE, family = 'binomial')
plot(GLMNETModelCV)
coef(GLMNETModelCV)
rm(trainClassMatrix)
#Recommended use
GLMNETModel <- GLMNETModelCV$glmnet.fit
plot(GLMNETModel, xvar="lambda", label=TRUE)

#Regression
#Cross-validation
#Find regression targets
whichFire <- which(train$target > 0)
#transform train Dataframe to model matrix as glmnet only accepts matrices as input
trainRegMatrix <- model.matrix(~ . , data = train[whichFire , c(linearRegPredictors, 'target')]) 
#cross validate the data, glmnet does it automatially there is no need for the caret package or a custom CV
registerDoParallel(detectCores() - 1)
GLMNETModelCVReg <- cv.glmnet(x = trainRegMatrix[,1:(dim(trainRegMatrix)[2]-1)], 
                              y = trainRegMatrix[,dim(trainRegMatrix)[2]], nfolds = 5,
                              parallel = TRUE, family = 'gaussian')
plot(GLMNETModelCVReg)
coef(GLMNETModelCVReg)
rm(trainRegMatrix)
#Recommended use
GLMNETModelReg <- GLMNETModelCVReg$glmnet.fit
plot(GLMNETModelReg, xvar = "lambda", label=TRUE)

#All Data
#Cross-validation
randomSubset <- sample.int(nrow(train), nrow(train)) #use this to use full data
#transform train Dataframe to model matrix as glmnet only accepts matrices as input
trainAllMatrix <- model.matrix(~ . , data = train[randomSubset , c(linearPredictorsFull, 'target')]) 
#cross validate the data, glmnet does it automatially there is no need for the caret package or a custom CV
registerDoParallel(detectCores() - 1)
GLMNETModelCVAll <- cv.glmnet(x = trainAllMatrix[,1:(dim(trainAllMatrix)[2]-1)],
                              y = trainAllMatrix[,dim(trainAllMatrix)[2]], nfolds = 5,
                              parallel = TRUE, family = 'gaussian')
plot(GLMNETModelCVAll)
coef(GLMNETModelCVAll)
rm(trainAllMatrix)
#Recommended use
GLMNETModelAll <- GLMNETModelCVAll$glmnet.fit
plot(GLMNETModelAll, xvar="lambda", label=TRUE)

#---------------------------------------------------------
#GLMNETS' predictions
trainClassMatrix <- model.matrix(~ . , data = train[ , linearClassPredictors]) 
trainGLMNETClass <- exp(predict(GLMNETModelCV, newx = trainClassMatrix))
rm(trainClassMatrix)

trainRegMatrix <- model.matrix(~ . , data = train[ , linearRegPredictors])
trainGLMNETReg <- predict(GLMNETModelCVReg, newx = trainRegMatrix)
rm(trainRegMatrix)

trainAllMatrix <- model.matrix(~ . , data = train[ , linearPredictorsFull) 
trainGLMNETAll <- predict(GLMNETModelCVAll, newx = trainAllMatrix)
rm(trainAllMatrix)

#GLMNET Extra features
trainClassRegGLMNET <- trainGLMNETClass * trainGLMNETReg
trainClassAllGLMNET <- trainGLMNETClass * trainGLMNETAll
trainRegAllGLMNET <- trainGLMNETReg * trainGLMNETAll
  
#train responses data frame
GLMNETTrainPredictions <- data.frame(trainGLMNETClass, trainGLMNETReg, trainGLMNETAll, trainClassRegGLMNET,
                                    trainClassAllGLMNET, trainRegAllGLMNET, train$target)
names(GLMNETTrainPredictions) <- c('GLMNETClass', 'GLMNETReg', 'GLMNETAll', 'ClassRegGLMNET',
                                   'ClassAllGLMNET', 'RegAllGLMNET', 'target')

trainPredictions <- cbind(GBMTrainPredictions, GLMNETTrainPredictions)

#Feature Selection for ensemble
ensembleControl <- rfeControl(functions = lmFuncs,
                              method = "cv",
                              repeats = 5,
                              verbose = TRUE)

randomSubset <- sample.int(nrow(train), nrow(train)) #use this to use full data
ensembleFeatures <- rfe(x = trainPredictions[randomSubset, 1:12], 
                        y = trainPredictions$target[randomSubset],
                        rfeControl = ensembleControl)

ensembleFeatures <- predictor(ensembleFeatures)

#Build Ensembles
ensembleControl <- trainControl(method="cv",
                                number=5,
                                verbose = TRUE)

ensembleGrid <- expand.grid(.interaction.depth = seq(1, 7, 3),
                            .shrinkage = c(0.001, 0.003, 0.01), 
                            .n.trees = 1000)

randomSubset <- sample.int(nrow(train), nrow(train)) #use this to use full data
MODEnsemble <- train(form = target ~ ., 
                     data = trainPredictions[randomSubset , c(ensembleFeatures, 'target')],
                     method = "gbm",
                     tuneGrid = ensembleGrid,
                     trControl = ensembleControl,
                     distribution = 'gaussian',
                     train.fraction = 0.7,
                     verbose = TRUE)

#Best Number of trees
ensembleTrees <- gbm.perf(MODEnsemble$finalModel, method = 'test')
treesIterated <- max(ensembleGrid$.n.trees)
gbmEnsembleExpanded <- MODEnsemble$finalModel

while(ensembleTrees >= treesIterated - 100){
  # do another 1000 iterations  
  gbmEnsembleExpanded <- gbm.more(gbmEnsembleExpanded, max(ensembleGrid$.n.trees),
                                  data = trainPredictions[randomSubset , c(ensembleFeatures, 'target')],
                                  verbose=TRUE)
  ensembleTrees <- gbm.perf(gbmEnsembleExpanded, method = 'test')
  treesIterated <- treesIterated + max(ensembleGrid$.n.trees)
  
  if(treesIterated >= 50000){break}  
}

#Full Model
randomSubset <- sample.int(nrow(train), nrow(train)) #use this to use full data
EnsembleModel <- gbm.fit(x = trainPredictions[randomSubset , ensembleFeatures],
                         y = trainPredictions$target[randomSubset], distribution = 'gaussian', 
                         interaction.depth = MODEnsemble$bestTune[2], shrinkage = MODEnsemble$bestTune[3],
                         n.trees = ensembleTrees, verbose = TRUE)

##########################################################
#PREDICTIONS
#GBM
#Loss - no Loss to Fire Prediction
GBMPrediction <- predict(GBMModel, newdata = test[ , GBMClassPredictors], n.trees = treesClass,, type = 'response')
#Value Regression Prediction
GBMPredictionReg <- predict(GBMModelReg, newdata = test[ , GBMRegPredictors], n.trees = treesReg)
#All Data Regression Prediction
GBMPredictionAll <- predict(GBMModelAll, newdata = test[ , GBMAllPredictors], n.trees = treesAll)

#2-3 models Ensembles
#Simple combinations
#Classification - Regression
GBMPredReg <- GBMPrediction * GBMPredictionReg
#Classification - All
GBMPredAll <- GBMPrediction * GBMPredictionAll
#Regression - All
GBMPredAll <- GBMPredictionReg * GBMPredictionAll

#Test responses data frame
GBMTestPredictions <- data.frame(GBMPrediction, GBMPredictionReg, GBMPredictionAll, 
                                 GBMPredReg, GBMPredAll, GBMPredAll)
names(GBMTestPredictions) <- c('GBMClass', 'GBMReg', 'GBMAll',
                               'ClassReg', 'ClassAll', 'RegAll')

#GLMNET
#Classification
testMatrix <- model.matrix(~ . , data = test[ , linearClassPredictors])
GLMNETPrediction <- exp(predict(GLMNETModelCV, newx = testMatrix))   #it needs to be fixed, since model matrix deletes data, the test matrix ends up being incomplete
rm(testMatrix)
#Regression
testMatrixReg <- model.matrix(~ . , data = test[ , c(linearRegPredictors])
GLMNETPredictionReg <- predict(GLMNETModelCVReg, newx = testMatrixReg)   #it needs to be fixed, since model matrix deletes data, the test matrix ends up being incomplete
rm(testMatrixReg)
#All Data
testMatrixAll <- model.matrix(~ . , data = test[ , linearPredictorsFull])
GLMNETPredictionAll <- predict(GLMNETModelCVAll, newx = testMatrixAll)   #it needs to be fixed, since model matrix deletes data, the test matrix ends up being incomplete
rm(testMatrixAll)

#Classification - Regression
GLMNETPredReg <- GLMNETPrediction * GLMNETPredictionReg
#Classification - All
GLMNETPredAll <- GLMNETPrediction * GLMNETPredictionAll
#Regression - All
GLMNETRegAll <- GLMNETPredictionReg * GLMNETPredictionAll

#test responses data frame
GLMNETTestPredictions <- data.frame(GLMNETPrediction, GLMNETPredictionReg, GLMNETPredictionAll,
                                    GLMNETPredReg, GLMNETPredAll, GLMNETRegAll)
names(GLMNETTestPredictions) <- c('GLMNETClass', 'GLMNETReg', 'GLMNETAll', 
                                   'ClassRegGLMNET', 'ClassAllGLMNET', 'RegAllGLMNET')

testPredictions <- cbind(GBMTestPredictions, GLMNETTestPredictions)

ensamblePredictions <- predict(EnsembleModel,
                               newdata = testPredictions,
                               t.trees = ensembleTrees)

#########################################################
#Write .csv multiplication
submissionTemplate$target <- GBMPredReg
write.csv(submissionTemplate, file = "predictionII.csv", row.names = FALSE)

#Write .csv ensemble
submissionTemplate$target <- ensamblePredictions
write.csv(submissionTemplate, file = "predictionTestII.csv", row.names = FALSE)

#Write .csv GLMNET ensemble
submissionTemplate$target <- ensambleGLMNETPredictions
write.csv(submissionTemplate, file = "predictionTestIII.csv", row.names = FALSE)

#Write .csv GLMNET full ensemble
submissionTemplate$target <- fullensemblePredictions
write.csv(submissionTemplate, file = "predictionTestIV.csv", row.names = FALSE)

#Write .csv GLMNET pred reg
submissionTemplate$target <- GLMNETPredReg
write.csv(submissionTemplate, file = "predictionV.csv", row.names = FALSE)
