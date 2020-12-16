
###################################################################################

# Semana 6 - MTCD, 19 de Outubro de 2020

# Mestrado em Ciência de Dados, ISCTE-IUL

# Examplo Classificação com Feedforward ANN (Satellite Image data)


###################################################################################


# Classifying Satellite Image Data
# The satellite image dataset represents a digitized satellite 
# image of a portion of the earth's surface. The training and 
# test data consist of 300 pixels for which ground truth has been established. 
# Ground truth of a satellite image is established by having a person on 
# the ground measuring the same thing the satellite is trying to measure 
# (at the same time). The answers are then compared to help evaluate how well 
# the satellite instrument is doing its job



library(neuralnet)
# Use set.seed for a consistent result
set.seed(1000)
Sonar <- read.csv("C:/Users/admin/Desktop/ANN/JustEnough/Sonar.csv")
Sonar2<- Sonar

# Our goal is to build a neural network that can be used to monitor land
# cover changes in the region defined by the dataset. Once a network 
# architecture is determined to be acceptable, the model can be used 
# to monitor for significant changes in the specified region.


# scale the data
Sonar2 <- scale(Sonar2[-7])

# Add back the outcome variable
Sonar2 <- cbind(Sonar[7],Sonar2)

# call neuralnet to create a single hidden layer of 3 nodes.
Sonar2.train <- Sonar2[1:150, ]
Sonar2.test <- Sonar2[151:300, ]
my.nnet <- neuralnet(class ~ .,
                     data=Sonar2.train,hidden=10,act.fct = 'logistic', linear.output = F)
plot(my.nnet)
# my.pred represents the network's predicted values 
my.pred<- predict(my.nnet,Sonar2.test)

# Place the results in a data frame.
my.results <-data.frame(Predicted=my.pred)
# my.results

pred.subs<- function(x)
{
  # This function examines each row of data frame x and determines the column 
  # number of the largest numerical value within each row. Each successive
  # column number is added to a growing list of winning values.
  # The list of winners is then returned to the caller.
  # This function can be used as one of two arguments to the table
  # function that creates the confusion matrix
  
  y<- c()
  for (i in 1:nrow(x))
  {
    largest <- x[i,1]
    largSub = 1
    
    for (j in 1:ncol(x))
    {
      if (x[i,j] > largest)
      {largest = x[i,j]
      largSub =j}} 
    
    y <- append(y,largSub)} # end for i
  return(y)
} # pred.subs


# Call pred.subs to create the table needed to make
#the confusion matrix.
my.predList <- pred.subs(my.results)
# my.predList

# Structure the confusion matrix
my.conf <- table(Sonar2.test$class,my.predList)

# Add a column of matching class numbers 
cbind(1:15,my.conf)

confusionP <- function (x)
  # This function uses confusion matrix x to determine
  # model accuracy. 
  
{correct=0
wrong =0
y<- nrow(x)
z<- ncol(x)
for (i in 1:y) 
{
  for (j in 1:z)
    
    if(i==j) 
      correct = correct + x[i,j]
    else 
      wrong = wrong + x[i,j]
}
pc <-(round(correct/(correct + wrong)*100,2))
cat("  Correct=", correct," ")
cat("Incorrect=", wrong,"\n")

cat("Accuracy =",pc,"%","\n") }


# Print % correct
confusionP(my.conf)


