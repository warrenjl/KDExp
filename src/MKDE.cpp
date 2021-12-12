#include "RcppArmadillo.h"
#include "KDExp.h"
using namespace arma;
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]

Rcpp::List MKDE(int mcmc_samples,
                arma::vec y,
                arma::mat x,
                arma::mat z_ppd,
                arma::mat H,
                int likelihood_indicator,
                Rcpp::Nullable<Rcpp::NumericVector> offset = R_NilValue,
                Rcpp::Nullable<double> a_r_prior = R_NilValue,
                Rcpp::Nullable<double> b_r_prior = R_NilValue,
                Rcpp::Nullable<double> a_sigma2_epsilon_prior = R_NilValue,
                Rcpp::Nullable<double> b_sigma2_epsilon_prior = R_NilValue,
                Rcpp::Nullable<double> sigma2_regress_prior = R_NilValue,
                Rcpp::Nullable<double> r_init = R_NilValue,
                Rcpp::Nullable<double> sigma2_epsilon_init = R_NilValue,
                Rcpp::Nullable<Rcpp::NumericVector> beta_init = R_NilValue,
                Rcpp::Nullable<double> theta_init = R_NilValue){

//Defining Parameters and Quantities of Interest
int p_x = x.n_cols;
int n = z_ppd.n_cols;
int m = z_ppd.n_rows;
arma::mat H_inv = inv_sympd(H);

arma::vec r(mcmc_samples); r.fill(0.00);
arma::vec sigma2_epsilon(mcmc_samples); sigma2_epsilon.fill(0.00);
arma::mat beta(p_x, mcmc_samples); beta.fill(0.00);
arma::vec theta(mcmc_samples); theta.fill(0.00);
arma::vec z(n); z.fill(0.00);
arma::vec neg_two_loglike(mcmc_samples); neg_two_loglike.fill(0.00);

arma::vec off_set(n); off_set.fill(0.00);
if(offset.isNotNull()){
  off_set = Rcpp::as<arma::vec>(offset);
  }

//Prior Information
int a_r = 1;
if(a_r_prior.isNotNull()){
  a_r = Rcpp::as<int>(a_r_prior);
  }

int b_r = 100;
if(b_r_prior.isNotNull()){
  b_r = Rcpp::as<int>(b_r_prior);
  }

double a_sigma2_epsilon = 0.01;
if(a_sigma2_epsilon_prior.isNotNull()){
  a_sigma2_epsilon = Rcpp::as<double>(a_sigma2_epsilon_prior);
  }

double b_sigma2_epsilon = 0.01;
if(b_sigma2_epsilon_prior.isNotNull()){
  b_sigma2_epsilon = Rcpp::as<double>(b_sigma2_epsilon_prior);
  }

double sigma2_regress = 10000.00;
if(sigma2_regress_prior.isNotNull()){
  sigma2_regress = Rcpp::as<double>(sigma2_regress_prior);
  }

//Initial Values
r(0) = b_r;
if(r_init.isNotNull()){
  r(0) = Rcpp::as<int>(r_init);
  }

sigma2_epsilon(0) = var(y);
if(sigma2_epsilon_init.isNotNull()){
  sigma2_epsilon(0) = Rcpp::as<double>(sigma2_epsilon_init);
  }

beta.col(0).fill(0.00);
if(beta_init.isNotNull()){
  beta.col(0) = Rcpp::as<arma::vec>(beta_init);
  }

theta(0) = 0.00;
if(theta_init.isNotNull()){
  theta(0) = Rcpp::as<double>(theta_init);
  }

z = trans(z_ppd.row(0));

neg_two_loglike(0) = neg_two_loglike_update(y,
                                            x,
                                            off_set,
                                            likelihood_indicator,
                                            n,
                                            r(0),
                                            sigma2_epsilon(0),
                                            beta.col(0),
                                            theta(0),
                                            z);

//Main Sampling Loop
arma::vec w(n); w.fill(0.00);
arma::vec gamma = y;
if(likelihood_indicator == 2){
  
  //w Update
  Rcpp::List w_output = w_update(y,
                                 x,
                                 off_set,
                                 likelihood_indicator,
                                 n,
                                 r(0),
                                 beta.col(0),
                                 theta(0),
                                 z);
  w = Rcpp::as<arma::vec>(w_output[0]);
  gamma = Rcpp::as<arma::vec>(w_output[1]);
  
  }

for(int j = 1; j < mcmc_samples; ++j){
  
   if(likelihood_indicator == 1){
  
     //sigma2_epsilon Update
     sigma2_epsilon(j) = sigma2_epsilon_update(y,
                                               x,
                                               n,
                                               a_sigma2_epsilon,
                                               b_sigma2_epsilon,
                                               beta.col(j-1),
                                               theta(j-1),
                                               z);
     w.fill(1.00/sigma2_epsilon(j));
    
     }
    
   if(likelihood_indicator == 0){
    
     //w Update
     Rcpp::List w_output = w_update(y,
                                    x,
                                    off_set,
                                    likelihood_indicator,
                                    n,
                                    r(j-1),
                                    beta.col(j-1),
                                    theta(j-1),
                                    z);
     w = Rcpp::as<arma::vec>(w_output[0]);
     gamma = Rcpp::as<arma::vec>(w_output[1]);
    
     }
  
   //beta, theta Update
   Rcpp::List delta_output = delta_update(x, 
                                          off_set,
                                          n,
                                          p_x,
                                          sigma2_regress,
                                          w,
                                          gamma,
                                          z);
  
   beta.col(j) = Rcpp::as<arma::vec>(delta_output[0]);
   theta(j) = Rcpp::as<double>(delta_output[1]);
   
   //z Update
   z = z_MKDE_update(x,
                     z_ppd,
                     H_inv,
                     off_set,
                     n,
                     m,
                     w,
                     gamma,
                     beta.col(j),
                     theta(j));
   
   if(likelihood_indicator == 2){
     
     //r Update
     r(j) = r_update(y,
                     x,
                     off_set,
                     n,
                     a_r,
                     b_r,
                     beta.col(j),
                     theta(j),
                     z);
     
     //w Update
     Rcpp::List w_output = w_update(y,
                                    x,
                                    off_set,
                                    likelihood_indicator,
                                    n,
                                    r(j),
                                    beta.col(j),
                                    theta(j),
                                    z);
     w = Rcpp::as<arma::vec>(w_output[0]);
     gamma = Rcpp::as<arma::vec>(w_output[1]);
     
     }
   
   //neg_two_loglike Update
   neg_two_loglike(j) = neg_two_loglike_update(y,
                                               x,
                                               off_set,
                                               likelihood_indicator,
                                               n,
                                               r(j),
                                               sigma2_epsilon(j),
                                               beta.col(j),
                                               theta(j),
                                               z);
  
   //Progress
   if((j + 1) % 10 == 0){ 
     Rcpp::checkUserInterrupt();
     }
  
   if(((j + 1) % int(round(mcmc_samples*0.10)) == 0)){
    
     double completion = round(100*((j + 1)/(double)mcmc_samples));
     Rcpp::Rcout << "Progress: " << completion << "%" << std::endl;
     Rcpp::Rcout << "**************" << std::endl;
    
     }
  
   }
                                  
return Rcpp::List::create(Rcpp::Named("sigma2_epsilon") = sigma2_epsilon,
                          Rcpp::Named("beta") = beta,
                          Rcpp::Named("theta") = theta,
                          Rcpp::Named("r") = r,
                          Rcpp::Named("neg_two_loglike") = neg_two_loglike);

}

