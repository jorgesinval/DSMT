###################################################################################

# Semana 6 - MTCD, 19 de Outubro de 2020

# Mestrado em Ciência de Dados, ISCTE-IUL

# Examplo de Feedforward neural network com backpropagation para 

# o conjunto de dados "Boston" (de library(MASS))


###################################################################################



library(neuralnet)
library(MASS) 
library(caTools)
library(neuralnet)
library(ggplot2)
library(knitr)


# CRIM - per capita crime rate by town
# ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
# INDUS - proportion of non-retail business acres per town.
# CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# NOX - nitric oxides concentration (parts per 10 million)
# RM - average number of rooms per dwelling
# AGE - proportion of owner-occupied units built prior to 1940
# DIS - weighted distances to five Boston employment centres
# RAD - index of accessibility to radial highways
# TAX - full-value property-tax rate per $10,000
# PTRATIO - pupil-teacher ratio by town
# B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# LSTAT - % lower status of the population
# MEDV - Median value of owner-occupied homes in $1000's



# ler os dados
set.seed(1)
data = Boston

# cabeçario e estrutura dos dados
head(Boston)
str(Boston)

# pré-procesamento dos dados Min-Max_scaling
# calcular o valor máximo e o valor mínimo em cada coluna (variável)
max_data <- apply(data, 2, max)
min_data <- apply(data, 2, min)

# transformar os dados
data_scaled <- scale(data,center = min_data, scale = max_data - min_data)

# separar a amostra em conjunto de treino e conjunto de teste (80%-20%)
index = sample(1:nrow(data),round(0.80*nrow(data)))
train_data <- as.data.frame(data_scaled[index,])
test_data <- as.data.frame(data_scaled[-index,])

#n = names(data)
#f = as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))

# fórmula a usar na rede (target~features)
f = as.formula(paste"medv ~.")

# definir a rede (1 camada oculta com 10 neurónios)                              
net_data = neuralnet(f,data=train_data,hidden=10,linear.output=T)

# visualizar a rede neuronal
par(mfrow=c(1,1))
plot(net_data)

# prever sobre o conjunto de teste
predict_net_test <- compute(net_data,test_data[,1:13])

# voltar para a escala original das variáveis (re-inverter a transformação)
predict_net_test_start <- predict_net_test$net.result*(max(data$medv)-min(data$medv))+min(data$medv)
test_start <- as.data.frame((test_data$medv)*(max(data$medv)-min(data$medv))+min(data$medv))

# calcular o MSE      
MSE.net_data <- sum((test_start - predict_net_test_start)^2)/nrow(test_start)
print(MSE.net_data)


# To do: "afinar a rede" no sentido de obter um erro de previsão menor
# isto é: alterar número de neurónios na camada oculta
# inserir mais uma camada oculta
# alterar a função de activação
# retirar variáveis independentes (features)
# alterar o learning rate
# Fazer um pequeno ciclo /função que selecciona o melhor modelo de rede neuronal (só em função de 1 ou 2 dos hiperparâmetros)

#######################################################################

# correr um modelo de regressão linear (com todas as variáveis independentes ~.)

Regression_Model <- lm(medv~., data=data)

# output do modelo de regressão linear
summary(Regression_Model)

# dividir a amostra em conjunto de treino e teste
test <- data[-index,]

# fazer a previssão sobre o conjunto de teste (usando a regressão definida)
predict_lm <- predict(Regression_Model,test)

# calcular MSE da regressão linear 
MSE.lm <- sum((predict_lm - test$medv)^2)/nrow(test)
MSE.lm

# relembrar o MSE da ANN
MSE.net_data


# analisar com mais cuidado a regressão linear

fit1=lm(medv~lstat,data=Boston)
#output
summary(fit1)

# reta de regressão
#dev.off()
plot(medv~lstat,Boston)
abline(fit1,col="red")

# observar os resíduos

par(mfrow=c(2,2))
plot(fit1)

# fazer a previssão sobre o conjunto de teste (usando a regressão definida)
predict1 <- predict(fit1,test)

# calcular MSE da regressão linear 
MSE1.lm <- sum((predict1 - test$medv)^2)/nrow(test)
MSE1.lm

# tentar um modelo de regressão polinomial (não-linear)

fit6=lm(medv~lstat +I(lstat^2),Boston)
summary(fit6)
plot(fit6)

par(mfrow=c(1,1))
plot(medv~lstat, Boston)
points(lstat,fitted(fit6), col="red",pch=20)

# fazer a previssão sobre o conjunto de teste (usando a regressão definida)
predict6 <- predict(fit6,test)

# calcular MSE da regressão linear 
MSE6.lm <- sum((predict6 - test$medv)^2)/nrow(test)
MSE6.lm

# tentar um modelo de regressão com interacção entre as variáveis
fit5=lm(medv~lstat*age,Boston)
summary(fit5)

# fazer a previssão sobre o conjunto de teste (usando a regressão definida)
predict5 <- predict(fit5,test)

# calcular MSE da regressão linear 
MSE5.lm <- sum((predict5 - test$medv)^2)/nrow(test)
MSE5.lm


############################################################################################

# vamos fazer mais uma tentativa de previsão com uma rede ANN com 2 camadas intermédias
# rename the dataset
data <- Boston

# 1. Standardize data
# Normalize : min-max scale, z-normalization etc

# grab min and max value per column using the apply function.
maxs <- apply(data, MARGIN=2, max) 
#margin 2 means we want to apply this function to column. help(apply) for more info.
maxs #(max values of each of the columns.)

mins <- apply(data, MARGIN=2, min)
mins

#help(scale) scale is going to return numeric matrix -> will need to change back to df
scaled.data <- scale(data, center = mins, scale = maxs - mins) # This means, each data value will be subtracted by the mins, then divided by max-mins. so get data, subtract mins and divide by max-mins.
scaled <- as.data.frame(scaled.data) # turn the matrix into frame.


# 2. now split train / test by using library(caTools)

split <- sample.split(scaled$medv, SplitRatio = 0.9)
train <- subset(scaled, split == T)
test <- subset(scaled, split == F)

# 3. train the model


# The format for neuralnet: y ~ col1 + col2 .
# since that's a bit redundant...we will do this way
n <- names(train)
n

f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))
f


## medv ~ crim + zn + indus + chas + nox + rm + age + dis + rad + 
##     tax + ptratio + black + lstat


nn <- neuralnet(f, data = train, hidden = c(5,3), linear.output = TRUE) 
# hidden: vector of integers, specifying number of hidden neurons in each layers. (first hidden layer of 5 neurons, second hidden layer of 3 neurons)
# linear.output: Whether this is continuous value, so in our case it's true. but if you are performing classification, this should be false.
plot(nn)


# create prediction with model
predicted.nn.values <- compute(nn, test[1:13]) # for neuralnet, we use compute function instead of predict. pass in the neural net model, and pass in test data without the labels. ( there are 14 features in the data, we just need 13.)


# this is list of neurons and net.result. and what we want is net result.
# but we scaled the data earlier for the training model.
# So we need to undo the operation in order to obtain the true predictions!

true.predictions <- predicted.nn.values$net.result * 
  (max(data$medv) - min(data$medv)) + min(data$medv)
# we were subtracting from the center value and then dividing by that scale value to perform our normalization operation.
# So for true.predictions, we are inverting this.

#convert the test data mean squared error
test.r <- (test$medv) * max(data$medv) - min(data$medv) + min(data$medv)
MSE.nn <- sum((test.r - true.predictions)^2)/nrow(test)
MSE.nn


# we can visualize error by graphically showing the true predictions plotted by the test values.
error.df <- data.frame(test.r, true.predictions)
head(error.df)

# scatter plot for target and predicted values
plot(error.df)



library(mlbench)
library(caret)
library(nnet)

# Data Boston Housing
data(BostonHousing)

# check range of medv (for scaling)
summary(BostonHousing$medv)

# Data partition training+testing
inTrain <- createDataPartition(BostonHousing$medv, p = 0.75, list=FALSE)
train.set <- BostonHousing[inTrain,]
test.set <- BostonHousing[-inTrain,]

# Build the model
netmodel <- nnet(medv/50~., size=5, data=train.set, maxit=500)

#  Testing model
medv.predict <- predict(netmodel,test.set[,-14])*50

# Counting Mean Squared Error
mean((medv.predict - test.set$medv)^2)

# plotting chart
plot(test.set$medv, medv.predict, main="Neural Network Prediction Actual", xlab="Actual")

abline(lm(medv.predict~test.set$medv),col="red")

