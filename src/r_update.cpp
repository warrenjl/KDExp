#include "RcppArmadillo.h"
#include "KDExp.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

int r_update(arma::vec y,
             arma::mat x,
             arma::vec off_set,
             int n,
             int a_r,
             int b_r,
             arma::vec beta_old,
             double theta_old,
             arma::vec z_old){

arma::vec mu = off_set +
               x*beta_old +
               z_old*theta_old;
  
arma::vec prob = 1.00/(1.00 + exp(-mu));
  
arma::vec r_log_val(b_r - a_r + 1); r_log_val.fill(0.00);  
int counter = 0;
for(int j = (a_r - 1); j < b_r; ++j){

   for(int k = 0; k < n; ++k){
      r_log_val(counter) = r_log_val(counter) +
                           R::dnbinom(y(k),
                                      (j + 1),
                                      (1.00 - prob(k)),
                                      TRUE);
      }
   counter = counter +
             1;
  
   }
  
arma::vec r_prob(b_r - a_r + 1); r_prob.fill(0.00);
for(int j = 0; j < (b_r - a_r + 1); ++j){
   r_prob(j) = 1.00/sum(exp(r_log_val - r_log_val(j)));
   }

IntegerVector sample_set = seq(a_r, b_r);
int r = sampleRcpp(wrap(sample_set), 
                   1, 
                   TRUE, 
                   wrap(r_prob))(0);
    
return(r);

}





