#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
int my_knn_cpp_pckg (NumericMatrix X, NumericVector X0, NumericVector y) {
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

// [[Rcpp::export]]
double my_knn_mink_cpp(NumericMatrix X, NumericVector X0, NumericVector y, double p) {
  int nrows = X.nrow(); 
  int ncols = X.ncol(); 
  
  double closest_distance = 99999999; 
  int closest_output = -1, closest_neighbor = -1; 
  double distance = 0, difference = 0; 
  
  for (int i = 0; i < nrows; i++) {
    distance = 0; 
    for (int j = 0; j < ncols; j++) {
      difference = abs(X(i,j) - X0[j]); 
      if (p > 0) { 
        distance = distance + pow(difference, p);
      } else {
        if (difference > distance) {
          distance = difference; 
        }}}
    if (p > 0) {
      distance = pow(distance, 1/p);
    }
    
    if (distance < closest_distance) {
      closest_distance = distance; 
      closest_output = y[i]; 
      closest_neighbor = i; 
    }
  }
  
  return closest_output;
}



// [[Rcpp::export]]
NumericMatrix my_knn_mink_scaled_cpp(NumericMatrix X, NumericVector X0, NumericVector y, double p, double s) {
  NumericMatrix newX = X;
  NumericVector newX0 = X0;
  
  int nrows = X.nrow(); 
  int ncols = X.ncol(); 
  
  if (s == 0) {
    for (int i = 0; i < ncols; i++) {
      double mean_val = mean(X(_,i)); 
      double sd_val = sd(X(_,i));
      for (int j = 0; j < nrows; j++) {
        newX(j,i) = (newX(j,i) - mean_val)/sd_val; 
      }  
      newX0[i] = (newX0[i] - mean_val)/ sd_val; 
    }
  } else if (s == 1) {
    for (int i = 0; i < ncols; i++) {
      double max_val = max(X(_,i)); 
      double min_val = min(X(_,i));
      for (int j = 0; j < nrows; j++) {
        newX(j,i) = (newX(j,i) - min_val)/(max_val - min_val); 
      }  
      newX0[i] = (newX0[i] - min_val)/(max_val - min_val); 
    }
  }
  return(my_knn_mink_cpp(newX, newX0, y, p));
}  