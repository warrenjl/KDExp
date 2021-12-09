#include "RcppArmadillo.h"
#include "KDExp.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

double neg_two_loglike_update(arma::vec y,
                              arma::mat x,
                              arma::vec off_set,
                              int likelihood_indicator,
                              int n,
                              int r,
                              double sigma2_epsilon,
                              arma::vec beta,
                              double theta,
                              arma::vec z){

double dens = 0.00;

arma::vec mu = off_set +
               x*beta + 
               z*theta;

arma::vec prob(n); prob.fill(0.00);

if(likelihood_indicator == 0){
  
  prob = 1.00/(1.00 + exp(-mu));
  dens = sum(y%log(prob) +
             (1.00 - y)%log(1.00 - prob));
    
  }

if(likelihood_indicator == 1){
  dens = -0.50*n*log(2*datum::pi*sigma2_epsilon) -
          0.50*dot((y - mu), (y - mu))/sigma2_epsilon;
  }

if(likelihood_indicator == 2){
  
  prob = 1.00/(1.00 + exp(-mu));
  for(int j = 0; j < n; ++j){
     dens = dens + 
            R::dnbinom(y(j), 
                       r, 
                       (1.00 - prob(j)),        
                       TRUE);
     }
  
  }

double neg_two_loglike = -2.00*dens;

return neg_two_loglike;

}

























































