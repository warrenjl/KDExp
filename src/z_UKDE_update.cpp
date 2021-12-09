#include "RcppArmadillo.h"
#include "KDExp.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

arma::vec z_UKDE_update(arma::mat x,
                        arma::mat z_ppd,
                        arma::vec h,
                        arma::vec off_set,
                        int n,
                        int m,
                        arma::vec w,
                        arma::vec gamma,
                        arma::vec beta,
                        double theta){
  
arma::vec var_z = pow(h,2)/(pow(theta,2)*(pow(h,2))%w + 1.00);
arma::mat log_c(m,n); log_c.fill(0.00);
for(int k = 0; k < m; ++k){
   log_c.row(k) = trans(-0.50*(pow((gamma - off_set - x*beta),2)%pow(h,2)%w + pow(trans(z_ppd.row(k)),2))/pow(h,2) +
                         0.50*(pow(theta*(gamma - off_set - x*beta)%pow(h,2)%w + trans(z_ppd.row(k)),2))/(pow(h,2)%(pow(theta,2)*(pow(h,2)%w) + 1.00)));
   }

//arma::mat temp_mat(m,n); temp_mat.fill(0.00);  
//arma::mat probs(m,n); probs.fill(0.00);
//for(int k = 0; k < m; ++k){
//  
//   temp_mat.each_row() = log_c.row(k); 
//   probs.row(k) = 1.00/sum(exp(log_c - temp_mat), 0);
//  
//   }
  
arma::mat temp_mat(m,n); temp_mat.fill(0.00);
temp_mat.each_row() = sum(exp(log_c), 0);
arma::mat probs = exp(log_c)/temp_mat;
             
double mu_z = 0.00;
IntegerVector sample_set = seq(0, (m - 1));
double index = 0.00;
arma::vec z(n); z.fill(0.00);
arma::vec mu_temp = off_set + 
                    x*beta;
for(int k = 0; k < n; ++k){
  
   index = sampleRcpp(wrap(sample_set), 
                      1, 
                      TRUE, 
                      wrap(probs.col(k)))(0);
      
   mu_z = (theta*(gamma(k) - mu_temp(k))*pow(h(k),2)*w(k) + z_ppd(index, k))/(pow(theta,2)*pow(h(k),2)*w(k) + 1.00);

   z(k) = R::rnorm(mu_z, 
                   sqrt(var_z(k)));  
      
   }
    
return(z);

}
