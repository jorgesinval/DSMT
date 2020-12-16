###################################################################################

# Semana 6 - MTCD, 19 de Outubro de 2020

# Mestrado em Ciência de Dados, ISCTE-IUL

# Examplo Classificação com Feedforward ANN (Diabetes Data)


###################################################################################

#Mining Diabetes Data

# The dataset-Diabetes-contains information about 768 females, 
# 268 of whom tested positive for diabetes. The data include eight numeric 
# input attributes and a categorical output attribute indicating 
# the outcome of a test for diabetes

# Given the input attributes, can we build a model to accurately determine 
# if an individual in the dataset tested positive for diabetes? 
# As a secondary goal, the model should error on the side of reporting 
# false positives rather than false negatives.

library(neuralnet)
# Build a model for the Diabetes data set

Diabetes <- read.csv("C:/Users/admin/Desktop/ANN/JustEnough/Diabetes.csv")
# scale the data
sca.dia <- scale(Diabetes[-9])

# Add back the outcome variable
sca.dia <- cbind(Diabetes[9],sca.dia)

# Randomize and split the data for training and testing 
set.seed(1000)
index <- sample(1:nrow(sca.dia), 2/3*nrow(sca.dia))

my.Train <- sca.dia[index,]
my.Test <- sca.dia[-index, ]

# Build and plot the network model.

my.nnetc <- neuralnet(Diabetes ~ .,
                      data=my.Train,hidden=6,
                      act.fct = 'logistic', linear.output = FALSE)
plot(my.nnetc)

# Make predictions on the test data.
my.pred<- predict(my.nnetc,my.Test)

# Make the table needed to create the confusion matrix.
my.results <-data.frame(Predictions =my.pred)

# ifelse converts probabilites to factor values. 
# Use column 1 of my.results for the ifelse statement.
# Column 1 values are probilites for tested_negative
  
my.predList<- ifelse(my.results[1] > 0.5,1,2) #  >.5="tested_negative" 

# Structure the confusion matrix
my.predList <- factor(my.predList,labels=c("Neg","Pos"))
my.conf <- table(my.Test$Diabetes,my.predList,dnn=c("Actual","Predicted"))
my.conf

source("C:/Users/admin/Desktop/ANN/JustEnough_R_Supporting_Materials/Scripts for Just Enough R/Chapter 8 Scripts/confusionP.R")
# Output accuracy
confusionP(my.conf)

