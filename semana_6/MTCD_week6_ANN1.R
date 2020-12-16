###################################################################################

# Semana 6 - MTCD, 19 de Outubro de 2020

# Mestrado em Ci?ncia de Dados, ISCTE-IUL

# Examplo simples de rede neuronal de tipo

# Feedforward com backpropagation para um conjunto de dados gerado para aprendizagem


###################################################################################

# Vamos usar a biblioteca "neuralnet", em particular as fun??es "neuralnet" e "compute/predict"
# compute is a Deprecated function, usar predict

# load library
# require(neuralnet)

library(neuralnet)

# definir os conjuntos de treino para as vari?veis (independentes (features) e dependente (target))
Train_X=c(20,10,30,20,80,30)
Train_Y=c(90,20,40,50,50,80)
Placed=c(1,0,0,0,1,1)  # target como vari?vel bin?ria (classifica??o)

# Colocar todas as vari?veis num data.frame (tabela)
df=data.frame(Train_X,Train_Y,Placed)
head(df)

# usar a fun??o "neuralnet" para definir e ajustar a rede
# a rede definine-se pela f?rmula: target~feature1+feature2+...
# hidden=3 - uma camada interm?dia com 3 neur?nios
# act.fct = "logistic" - fun??o de activa??o log?stica (sigmoid) na camada oculta
# linear.output = FALSE - ent?o a fun??o de activa??o aplica-se a camada de sa?da
# ainda podemos acrescentar: algorithm = "rprop+", (backprop ? por default),
# err.fct = "sse", learningrate, etc....

nn=neuralnet(Placed~Train_X+Train_Y,data=df, hidden=3,act.fct = "logistic",
             linear.output = FALSE)

# representar a arquitetura da rede neuronal
plot(nn)

# definir os conjuntos de teste (apenas para as vari?veis independentes)
Teste_X=c(30,40,85)
Teste_Y=c(85,50,40)
test=data.frame(Teste_X,Teste_Y)

# Fazer a previs?o para a rede definida (usar a fun??o antiga "compute")
Predict1=compute(nn,test)
Predict1$net.result

# Fazer a previs?o para a rede definida (usar a fun??o nova "predict")
# aten??o: a forma em chamar o resultado ? diferente, pois o output de cada
# uma das fun??es compute e predict tem uma estrutura diferente

Predict2=predict(nn,test)
Predict2
