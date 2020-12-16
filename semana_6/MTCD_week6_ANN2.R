###################################################################################

# Semana 6 - MTCD, 19 de Outubro de 2020

# Mestrado em Ciência de Dados, ISCTE-IUL

# Examplo de Feedforward neural network com backpropagation para o conjunto de dados "Auto"


###################################################################################

# Auto data set is a data frame with 392 observations on the following 9 variables.
# mpg - miles per gallon
# cylinders - Number of cylinders between 4 and 8
# displacement - Engine displacement (cu. inches)
# horsepower - Engine horsepower
# weight - Vehicle weight (lbs.)
# acceleration - Time to accelerate from 0 to 60 mph (sec.)
# year - Model year (modulo 100)
# origin - Origin of car (1. American, 2. European, 3. Japanese)
# name - Vehicle name

# Auto can be imported from "ISLR" library.
##################################################################################

# importar bibliotecas

library("neuralnet")
library("ISLR")

# ler/importar os dados da biblioteca ISLR 
data = Auto

# ver os dados (abre uma nova janela com os dados)
View(data)

# ver o cabeçario dos dados (as primeiras 5 linhas)
head(data)

# verificar se existem missing data com a função "is.na"
# como temos várias colunas/variáveis definimos uma função que procura os
# missing values por coluna e faz a soma deles por coluna

apply(data,2,function(x) sum(is.na(x)))

# não existem missing values

# representação gráfica (scatter plot) entre as variáveis "weight" e "mp"
# assinalamos uma variável da base de dados "data" usando "$" seguido do nome da variável
par(mfrow=c(1,1))
plot(data$weight, data$mp, pch=data$origin,cex=2)

# representação gráfica (scatter plot) entre várias pares de variáveis
# usamos um gráfico dividido em 4 subgráficos (2,2)
par(mfrow=c(2,2))
plot(data$cylinders, data$mp, pch=data$origin,cex=1)
plot(data$dis, data$mp, pch=data$origin,cex=1, col='blue')
plot(data$horse, data$mp, pch=data$origin,cex=1, col='green')
plot(data$acceleration, data$mp, pch=data$origin,cex=1, col='red')

# scaling data / transformar os dados (normalizar)
# calcular a média de cada uma das primeiras 6 variáveis numéricas/colunas
mean_data <- apply(data[1:6], 2, mean)

# calcular o devio padrão de cada uma das primeiras 6 variáveis
sd_data <- apply(data[1:6], 2, sd)

# normalizar os dados
data_scaled <- as.data.frame(scale(data[,1:6],center = mean_data, scale = sd_data))

# ver os dados normalizados (as primeiras 20 linhas)
head(data_scaled, n=20)

# Fazer a partição da amostra em conjunto de treino e conjunto de teste
# a partição é aleatória (usamos "sample") e 70% dos dados definem o conjunto de treino
index = sample(1:nrow(data),round(0.70*nrow(data)))

# conjunto de treino
train_data <- as.data.frame(data_scaled[index,])

# conjunto de teste
test_data <- as.data.frame(data_scaled[-index,])

# definir a função f como fórmula (target~features)
#n = names(data_scaled)
#f = as.formula(paste("mpg ~", paste(n[!n %in% "mpg"], collapse = " + ")))

f = as.formula("mpg ~.")

# treino da rede neuronal
# arquitectura: input =f, camada intermédia com 3 neurónios, 
# não aplicar a função de activação definida para o output
net = neuralnet(f,data=train_data,hidden=3,linear.output=TRUE)

# Visualisar o modelo ANN (com os pesos)
plot(net)

# Fazer previsão
predict_net_test <- compute(net,test_data[,2:6])

# calcular o erro de previsão (MSE - means square error)
MSE.net <- sum((test_data$mpg - predict_net_test$net.result)^2)/nrow(test_data)

########################################################################

# vamos fazer a mesma previsão usando um modelo de regressão linear

# Ajustar o modelo linear usando a função "lm"
Lm_Mod <- lm(mpg~., data=train_data)
summary(Lm_Mod)

# Prever os dados usando lm
predict_lm <- predict(Lm_Mod,test_data)

# Calcular o  MSE para o conjunto de teste
MSE.lm <- sum((predict_lm - test_data$mpg)^2)/nrow(test_data)

plot.new()

# Representar graficamente os valores preditos (pelos 2 modelos)
par(mfrow=c(1,2))
plot(test_data$mp,predict_net_test$net.result,col='red',main='Real vs predicted for neural network',pch=2,cex=1)
abline(0,1,lwd=5)

plot(test_data$mpg,predict_lm,col='red',main='Real vs predicted for linear regression',pch=2,cex=1)
abline(0,1,lwd=5)

plot.new()
par(mfrow=c(1,1))
# Comparar as previsões num mesmo gráfico
plot(test_data$mp,predict_net_test$net.result,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
points(test_data$mpg,predict_lm,col='blue',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend=c('NN','LM'),pch=18,col=c('red','blue'))





# neuralnet(formula, data, hidden = 1, threshold = 0.01,stepmax = 1e+05, rep = 1, startweights = NULL,
#    learningrate.limit = NULL, learningrate.factor = list(minus = 0.5,plus = 1.2), 
#    learningrate = NULL, lifesign = "none",lifesign.step = 1000, algorithm = "rprop+", err.fct = "sse",
#    act.fct = "logistic", linear.output = TRUE, exclude = NULL,constant.weights = NULL, likelihood = FALSE)