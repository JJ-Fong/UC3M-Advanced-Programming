#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double my_knn_cpp (NumericMatrix X, NumericVector X0, NumericVector y) {
  int nrows = X.nrow(); 
  int ncols = X.ncol(); 
  
  double closest_distance = 99999999; 
  int closest_output = -1, closest_neighbor = -1; 
  double distance = 0, difference = 0; 
  
  for (int i = 0; i < nrows; i++) {
    distance = 0; 
    for (int j = 0; j < ncols; j++) {
      difference = X(i,j) - X0[j]; 
      distance += (difference * difference); 
    }
    
    distance = sqrt(distance); 
    
    if (distance < closest_distance) {
      closest_distance = distance; 
      closest_output = y[i]; 
      closest_neighbor = i; 
    }
  }
  return closest_output; 
}

