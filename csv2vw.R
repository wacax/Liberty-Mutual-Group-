csv2vw <- function(csvFile, Label, Importance, Tag,
                   commaSeparated = TRUE, fullTextInput = TRUE, fromConnection = FALSE, outputFileDir = NULL){
  

#  Features have to be in the form of:
#  [Label] [Importance [Tag]]|Namespace Features |Namespace Features ... |Namespace Features

  
  labelIdx <- which(names(csvFile) == Label)
  ImportanceIdx <- which(names(csvFile) == Importance)
  TagIdx <- which(names(csvFile) == Tag)
  
  #determine column classes
  dataClasses <- sapply(csvFile, class)
  #determine numeric features
  numericIdx <- which(dataClasses == 'numeric')
  #determine categorical features
  categoricalIdx <- which(dataClasses == 'factor')
  #Vector with all columns names
  dataNames <- names(csvFile)
  
  vwText <- apply(csvFile, 1, function(csvLine, l, i, t, numIdx, catIdx, namesVec){
    #numeric features    
    numericVector <- apply(cbind(namesVec[numIdx], csvLine[numIdx]), 2, function(lin){
      return(paste0(lin[1], ':', lin[2]))
    })
    
    #categorical features    
    catVector <- apply(cbind(namesVec[catIdx], csvLine[catIdx]), 2, function(lin){
      return(paste0(lin[1], lin[2], ':', 1))
    })
    
    vwLine <- paste0(      
      csvLine[l], ' ',
      ifelse(i == NULL, '1.0', csvLine[i]), 
      ifelse(t == NULL, ' ', paste0(' ', csvLine[t])), 
      '|', 
      numericVector, 
      catVector
    )
    
  }, labelIdx, ImportanceIdx = NULL, TagIdx = NULL, numericIdx, categoricalIdx, dataNames)
  
  return(vwText)
}
