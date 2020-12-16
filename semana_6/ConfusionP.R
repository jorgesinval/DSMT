
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
