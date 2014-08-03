linearFeatureSelection <- function(uformula, allPredictorsData, userMethod = 'forward', 
                                   plotIt = TRUE, userMax = 200){
  #Linear Feature Selection Using the Leaps Package
  #TODO: Add documentation
  
  #NA omit, regsubsets and kmeans are sensitive to NAs
  noNAIndices <- which(apply(is.na(allPredictorsData), 1, sum) == 0)
  
  #Fire or not
  linearBestModels <- regsubsets(uformula, data = allPredictorsData[noNAIndices, ], method = userMethod, 
                                 nvmax = userMax, really.big = ifelse(userMethod == 'exhaustive' & ncol(allPredictorsData) > 50, TRUE, FALSE))
  
  bestMods <- summary(linearBestModels)
  names(bestMods)
  bestNumberOfPredictors <- which.min(bestMods$cp)
  
  if(plotIt == TRUE){
    #ggplot of optimal number of predictors
    #ErrorsFireClasif <- as.data.frame(bestMods$cp); names(ErrorsFireClasif) <- 'CPError'
    #print(ggplot(data = ErrorsFireClasif, aes(x = seq(1, nrow(ErrorsFireClasif)), y = CPError)) +  geom_line() +  geom_point())    
    
    #Regular plot of optimal number of predictors
    plot(bestMods$cp, xlab="Number of Variables", ylab="CP Error")
    points(bestNumberOfPredictors, bestMods$cp[bestNumberOfPredictors],pch=20,col="red")
  }

  #Extract the name of the most predictive columns' names
  #TODO: change the hard coded [2:82] that defines factors, find another way to pick factors' names
  predictors1 <- as.data.frame(bestMods$which)
  repeatedNames <- sapply(names(predictors1)[2:82], function(stringX){
    return(substr(stringX, 1, 4))
  })
  originalPredictorNames <- c(names(predictors1)[1], repeatedNames, names(predictors1)[83:length(names(predictors1))])
  predictors1 <- sort(apply(predictors1, 2, sum), decreasing = TRUE, index.return = TRUE)
  predictors1 <- unique(originalPredictorNames[predictors1$ix[2:bestNumberOfPredictors + 1]])
  return(list(predictors1, bestNumberOfPredictors))
}