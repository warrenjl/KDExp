#ifndef __KDExp__
#define __KDExp__

arma::vec rcpp_pgdraw(arma::vec b, 
                      arma::vec c);

Rcpp::NumericVector sampleRcpp(Rcpp::NumericVector x,
                               int size,
                               bool replace,
                               Rcpp::NumericVector prob = Rcpp::NumericVector::create());

int r_update(arma::vec y,
             arma::mat x,
             arma::vec off_set,
             int n,
             int a_r,
             int b_r,
             arma::vec beta_old,
             double theta_old,
             arma::vec z_old);
  
double sigma2_epsilon_update(arma::vec y,
                             arma::mat x,
                             int n,
                             double a_sigma2_epsilon,
                             double b_sigma2_epsilon,
                             arma::vec beta_old,
                             double theta_old,
                             arma::vec z_old);

Rcpp::List w_update(arma::vec y,
                    arma::mat x,
                    arma::vec off_set,
                    arma::vec tri_als,
                    int likelihood_indicator,
                    int n,
                    int r,
                    arma::vec beta_old,
                    double theta_old,
                    arma::vec z_old);

Rcpp::List delta_update(arma::mat x,
                        arma::vec off_set,
                        int n,
                        int p_x,
                        double sigma2_regress,
                        arma::vec w,
                        arma::vec gamma,
                        arma::vec z_old);

arma::vec z_UKDE_update(arma::mat x,
                        arma::mat z_ppd,
                        arma::vec h,
                        arma::vec off_set,
                        int n,
                        int m,
                        arma::vec w,
                        arma::vec gamma,
                        arma::vec beta,
                        double theta);

double neg_two_loglike_update(arma::vec y,
                              arma::mat x,
                              arma::vec off_set,
                              int likelihood_indicator,
                              int n,
                              int r,
                              double sigma2_epsilon,
                              arma::vec beta,
                              double theta,
                              arma::vec z);

Rcpp::List UKDE(int mcmc_samples,
                arma::vec y,
                arma::mat x,
                arma::mat z_ppd,
                arma::vec h,
                int likelihood_indicator,
                Rcpp::Nullable<Rcpp::NumericVector> offset,
                Rcpp::Nullable<Rcpp::NumericVector> trials,
                Rcpp::Nullable<double> a_r_prior,
                Rcpp::Nullable<double> b_r_prior,
                Rcpp::Nullable<double> a_sigma2_epsilon_prior,
                Rcpp::Nullable<double> b_sigma2_epsilon_prior,
                Rcpp::Nullable<double> sigma2_regress_prior,
                Rcpp::Nullable<double> r_init,
                Rcpp::Nullable<double> sigma2_epsilon_init,
                Rcpp::Nullable<Rcpp::NumericVector> beta_init,
                Rcpp::Nullable<double> theta_init); 

arma::vec z_MKDE_update(arma::mat x,
                        arma::mat z_ppd,
                        arma::mat H_inv,
                        arma::vec off_set,
                        int n,
                        int m,
                        arma::vec w,
                        arma::vec gamma,
                        arma::vec beta,
                        double theta);

Rcpp::List MKDE(int mcmc_samples,
                arma::vec y,
                arma::mat x,
                arma::mat z_ppd,
                arma::mat H,
                int likelihood_indicator,
                Rcpp::Nullable<Rcpp::NumericVector> offset,
                Rcpp::Nullable<Rcpp::NumericVector> trials,
                Rcpp::Nullable<double> a_r_prior,
                Rcpp::Nullable<double> b_r_prior,
                Rcpp::Nullable<double> a_sigma2_epsilon_prior,
                Rcpp::Nullable<double> b_sigma2_epsilon_prior,
                Rcpp::Nullable<double> sigma2_regress_prior,
                Rcpp::Nullable<double> r_init,
                Rcpp::Nullable<double> sigma2_epsilon_init,
                Rcpp::Nullable<Rcpp::NumericVector> beta_init,
                Rcpp::Nullable<double> theta_init);

#endif // __KDExp__
