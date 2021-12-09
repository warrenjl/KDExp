#include "RcppArmadillo.h"
#include "KDExp.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::vec z_MKDE_update(arma::mat x,
                        arma::mat z_ppd,
                        arma::mat H_inv,
                        arma::vec off_set,
                        int n,
                        int m,
                        arma::vec w,
                        arma::vec gamma,
                        arma::vec beta,
                        double theta){
  
arma::mat cov_z = inv_sympd(pow(theta,2)*arma::diagmat(w) + H_inv);
   
arma::mat log_c(1,m); log_c.fill(0.00);
arma::vec piece(n); piece.fill(0.00);
arma::vec temp(m); temp.fill(0.00);
for(int k = 0; k < m; ++k){
   
   piece = theta*(w%(gamma - off_set - x*beta)) + 
           H_inv*trans(z_ppd.row(k));
   log_c.col(k) = -0.50*z_ppd.row(k)*(H_inv*trans(z_ppd.row(k))) +
                   0.50*trans(piece)*(cov_z*piece);
   temp(k) = log_c(0,k);
      
   }

arma::vec probs(m); probs.fill(0.00);
for(int k = 0; k < m; ++k){
   probs(k) = 1.00/sum(exp(temp - temp(k)));
   }
             
IntegerVector sample_set = seq(0, (m - 1));
double index = sampleRcpp(wrap(sample_set), 
                          1, 
                          TRUE, 
                          wrap(probs))(0);

arma::vec mu_z = cov_z*(theta*(w%(gamma - off_set - x*beta)) + H_inv*trans(z_ppd.row(index)));
   
arma::mat ind_norms = arma::randn(1,n);
arma::vec z = mu_z + 
              trans(ind_norms*arma::chol(cov_z));
    
return(z);

}
