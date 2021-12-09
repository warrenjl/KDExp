#include "RcppArmadillo.h"
#include "KDExp.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List delta_update(arma::mat x,
                        arma::vec off_set,
                        int n,
                        int p_x,
                        double sigma2_regress,
                        arma::vec w,
                        arma::vec gamma,
                        arma::vec z_old){

arma::mat w_mat(n, (p_x + 1));
for(int j = 0; j < (p_x + 1); ++j){
   w_mat.col(j) = w;
   }

arma::mat z_temp(n,1); z_temp.col(0) = z_old;
arma::mat x_full = join_horiz(x, 
                              z_temp);

arma::mat x_full_trans = trans(x_full);

arma::mat cov_delta = inv_sympd(x_full_trans*(w_mat%x_full) + 
                               (1.00/sigma2_regress)*eye((p_x + 1), (p_x + 1)));

arma::vec mean_delta = cov_delta*(x_full_trans*(w%(gamma - off_set)));

arma::mat ind_norms = arma::randn(1, (p_x + 1));
arma::vec delta = mean_delta + 
                  trans(ind_norms*arma::chol(cov_delta));

arma::vec beta = delta.subvec(0, (p_x - 1));
double theta = delta(p_x);

return Rcpp::List::create(Rcpp::Named("beta") = beta,
                          Rcpp::Named("theta") = theta);

}



