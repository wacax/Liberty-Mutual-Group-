WeightedGini <- function(solution, weights, submission){
  
  #Scoring for the competition hosted at Kaggle Liberty Mutual Group found on
  #https://www.kaggle.com/c/liberty-mutual-fire-peril/forums/t/9880/update-on-the-evaluation-metric
  
  df = data.frame(solution = solution, weights = weights, submission = submission)
  df <- df[order(df$submission, decreasing = TRUE),]
  df$random = cumsum((df$weights/sum(df$weights)))
  totalPositive <- sum(df$solution * df$weights)
  df$cumPosFound <- cumsum(df$solution * df$weights)
  df$Lorentz <- df$cumPosFound / totalPositive
  n <- nrow(df)
  gini <- sum(df$Lorentz[-1]*df$random[-n]) - sum(df$Lorentz[-n]*df$random[-1])
  return(gini)
}
