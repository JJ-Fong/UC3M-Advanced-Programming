 
library(rstudioapi)
setwd(dirname(getActiveDocumentContext()$path)) 
library(Rcpp)
library(FNN)
library(microbenchmark)
library(assignment1pckg)
library("matrixStats")
library(dplyr)


sourceCpp("cpp_functions.cpp")
my_knn_cpp(X,X0,y)
FNN::knn(X, matrix(X0, nrow = 1), y, k=1)



microbenchmark(
  my_knn_R(X, X0, y)
  ,my_knn_cpp(X,X0,y)
  ,FNN::knn(X, matrix(X0, nrow = 1), y, k=1)
)




my_knn_R = function(X, X0, y){
  # X data matrix with input attributes
  # y response variable values of instances in X  
  # X0 vector of input attributes for prediction
  
  nrows = nrow(X)
  ncols = ncol(X)
  
  # One of the instances is going to be the closest one:
  #   closest_distance: it is the distance , min_output
  closest_distance = 99999999
  closest_output = -1
  closest_neighbor = -1
  
  for(i in 1:nrows){
    
    distance = 0
    for(j in 1:ncols){
      difference = X[i,j]-X0[j]
      distance = distance + difference * difference
    }
    
    distance = sqrt(distance)
    
    if(distance < closest_distance){
      closest_distance = distance
      closest_output = y[i]
      closest_neighbor = i
    }
  }
  closest_output
}



# X contains the inputs as a matrix of real numbers
data("iris")
# X contains the input attributes (excluding the class)
X <- iris[,-5]
# y contains the response variable (named medv, a numeric value)
y <- iris[,5]

# From dataframe to matrix
X <- as.matrix(X)
# From factor to integer
y <- as.integer(y)

# This is the point we want to predict
X0 <- c(5.80, 3.00, 4.35, 1.30)


my_knn_cpp(X,X0,y)
FNN::knn(X, matrix(X0, nrow = 1), y, k=1)

# Using my_knn and FNN:knn to predict point X0
# Using the same number of neighbors, it should be similar (k=1)
microbenchmark(
  my_knn_R(X, X0, y)
  ,my_knn_cpp(X,X0,y)
  ,FNN::knn(X, matrix(X0, nrow = 1), y, k=1)
)

#KNN with Euclidean Distance
my_knn_cpp(X,X0,y)
#KNN with Minkowsky Distance (p = 2, should be the same as the Euclidean)
my_knn_mink_cpp(X,X0,y,2)
#KNN with Minkowsky Distance (p = 4)
my_knn_mink_cpp(X,X0,y,4)
#KNN with Minkowsky Distance (p = -2, which uses a different formula)
my_knn_mink_cpp(X,X0,y,-2)


#Call of my_knn_mink_scaled_cpp with s = 1 (Normalized data)
newx = my_knn_mink_scaled_cpp(X,X0,y,4,1)
newx %>% head()
#If the data is correctly normalized, all the values should be between 0-1
summary(newx)

#Call of my_knn_mink_scaled_cpp with s = 0 (Standarized data)
newx = my_knn_mink_scaled_cpp(X,X0,y,4,0)
newx %>% head()
#If the data is correctly standardized, all the means should be 0 and all the sd should be 1
round(colMeans(newx),12)
colSds(newx)






