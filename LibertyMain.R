#Liberty Mutual Group - Fire Peril Loss Cost
#ver 0.1

#########################
#Init
rm(list=ls(all=TRUE))

#Load/install required libraries
require('ggplot2')

#Set Working Directory
workingDirectory <- "/home/wacax/Wacax/Kaggle/Liberty Mutual Group/Liberty Mutual Group - Fire Peril Loss Cost"
setwd(workingDirectory)

dataDirectory <- '/home/wacax/Wacax/Kaggle/Liberty Mutual Group/Data/'

#Load functions


#############################
#Load Data
#Input Data
rows2read <- 20000
train <- read.csv(paste0(dataDirectory, 'train.csv'), header = TRUE, stringsAsFactors = FALSE, nrows = rows2read)
test <- read.csv(paste0(dataDirectory, 'test.csv'), header = TRUE, stringsAsFactors = FALSE, nrows = rows2read)

submissionTemplate <- read.csv(paste0(dataDirectory, 'sampleSubmission.csv'), header = TRUE, stringsAsFactors = FALSE)

###############################
#EDA
str(train)
fireProb <- table(ifelse(train$target > 0, 1, 0)) / length(ifelse(train$target > 0, 1, 0))
fireCosts <- as.data.frame(train$target[train$target>0]); names(fireCosts) <- 'Cost'
ggplot(data = train, aes(x = ifelse(train$target > 0, TRUE, FALSE))) +  geom_histogram() 
ggplot(data = fireCosts, aes(x = Cost)) +  geom_density() 
ggplot(data = fireCosts, aes(x = log(Cost))) +  geom_density() 
