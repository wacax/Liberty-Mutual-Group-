#Liberty Mutual Group - Fire Peril Loss Cost
#ver 0.1

#########################
#Init
rm(list=ls(all=TRUE))

#Load/install libraries


#Set Working Directory
workingDirectory <- "/home/wacax/Wacax/Kaggle/Liberty Mutual Group/Liberty Mutual Group - Fire Peril Loss Cost"
setwd(workingDirectory)

dataDirectory <- '/home/wacax/Wacax/Kaggle/Liberty Mutual Group/Data/'

#Load functions


#############################
#Load Data
#Input Data
train <- read.csv(paste0(dataDirectory, 'train.csv'), header = TRUE, stringsAsFactors = FALSE)
test <- read.csv(paste0(dataDirectory, 'test.csv'), header = TRUE, stringsAsFactors = FALSE)

submissionTemplate <- read.csv(paste0(dataDirectory, 'sampleSubmission.csv'), header = TRUE, stringsAsFactors = FALSE)

###############################
#EDA
names(train)