

###################################################################################

# Semana 6 - MTCD, 19 de Outubro de 2020

# Mestrado em Ciência de Dados, ISCTE-IUL

# Examplo Classificação com Feedforward ANN (dividend data)


###################################################################################

# Main goal: develop a neural network to determine if 
# a stock pays a dividend or not (classification problem).
# 1 - if the stock pays a dividend
# 0 - if the stock does not pay a dividend

# Independent variables
# fcfps: Free cash flow per share (in $)
# earnings_growth: Earnings growth in the past year (in %)
# de: Debt to Equity ratio
# mcap: Market Capitalization of the stock
# current_ratio: Current Ratio (or Current Assets/Current Liabilities)

###############################################################

library(neuralnet)

mydata <- read.csv("C:/Users/admin/Desktop/dianaWork/teach_MTCD/R_programms/Diana_Rfiles/dividendinfo.csv")

# scaled normalization using function "scale"
scaleddata<-scale(mydata)

# Max-Min normalization, we define a function 
# to obtain the max-min scaling
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# we use lapply to run the function across our data
# and saved it into a data frame titled maxmindf
maxmindf <- as.data.frame(lapply(mydata, normalize))

# Training and Test Data
# training data (trainset) on 80% of the observations
trainset <- maxmindf[1:160, ]
testset <- maxmindf[161:200, ]

#Neural Network

# Observe that we are:
# Using neuralnet to "regress" the dependent "dividend" 
# variable against the other independent variables
# Setting the number of hidden layers to (2,1) based on the 
# hidden=(2,1) formula
# The linear.output variable is set to FALSE, given the 
# impact of the independent variables on the dependent 
# variable (dividend) is assumed to be non-linear
# The threshold is set to 0.01, meaning that if the change in 
# error during an iteration is less than 1%, then no further 
# optimization will be carried out by the model
# Deciding on the number of hidden layers in a neural network 
# is not an exact science. In fact, there are instances where 
# accuracy will likely be higher without any hidden layers. 
# Therefore, trial and error plays a significant role in this 
# process.
# One possibility is to compare how the accuracy of the predictions change
# as we modify the number of hidden layers.
# For instance, using a (2,1) configuration ultimately 
# yielded 92.5% classification accuracy for this example.''''

nn <- neuralnet(dividend ~ fcfps + earnings_growth + de + mcap + current_ratio, data=trainset, hidden=c(2,1), linear.output=FALSE, threshold=0.01)
plot(nn)

# We now generate the error of the neural network model, 
# along with the weights between the inputs, hidden layers, 
# and outputs:
nn$result.matrix

#Test the resulting output
# The "subset" function is used to eliminate the dependent 
# variable from the test data
# The "compute" function then creates the prediction variable
# A "results" variable then compares the predicted data with 
# the actual data
# A confusion matrix is then created with the table 
# function to compare the number of true/false positives
# and negatives

temp_test <- subset(testset, select = c("fcfps","earnings_growth", "de", "mcap", "current_ratio"))
head(temp_test)
nn.results <- predict(nn, temp_test)
results <- data.frame(actual = testset[,1], prediction = nn.results[,1])

# The predicted results are compared to the actual results:
results

# confusion matrix: we round up our results using sapply and 
# create a confusion matrix to compare the number of 
# true/false positives and negatives:
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
table(actual,prediction)

# The model generates 17 true negatives (0's), 
# 20 true positives (1's), while there are 3 false negatives.

