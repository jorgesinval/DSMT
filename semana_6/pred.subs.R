
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
