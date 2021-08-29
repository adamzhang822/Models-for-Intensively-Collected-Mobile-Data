data {
int<lower=1> K;                       // number of groups
int<lower=1> nsubj;                   // number of data points
int<lower=1> nt;                      // length of timeseries
matrix[nsubj,nt] y;                   // observations
}

parameters {
real<lower=0,upper=1> p0 ;                   // initial prob grp 1
vector<lower=0,upper=1>[K] tp ;              // transition probs of staying in group
ordered[K] mu;                               // location parameter of mixture components
vector[K] sig_e;                             // scale parameter of mixture componets
vector<lower=0>[2] sig_ls;                   // SD (scales) of mixture components
cholesky_factor_corr[2] chol_ls;             // Cholesky factors of location-scale effect covariance matrix
vector[2] theta[nsubj];                      // standardized REs (location and scale)
}

transformed parameters {
matrix<lower=0,upper=1>[nsubj,nt] pred;       // one-step filter prediction of probabililty of group membership
vector[2] nu[nsubj];                          // RE array (location and scale)
real nu_loc[nsubj];                           // RE location
real nu_scale[nsubj];                         // RE scale
vector[2] Sig_ls;                             // Variance of random effects 
real cor_ls;                                  // Correlation of random effects 
{
  matrix[nsubj,nt] F;                         //filtered belief states
  real like1;    
  real like2;
  real p1;
  real p2;
  
  Sig_ls = sqrt(diagonal(crossprod(diag_pre_multiply(sig_ls, chol_ls))));
  cor_ls = crossprod(chol_ls)[1,2];

  //Forwards algorithm
  for (n in 1:nsubj){ 
    F[n, 1] = p0; 
    pred[n, 1] = F[n, 1];
    nu[n] = (diag_pre_multiply(sig_ls, chol_ls)) * theta[n];
    nu_loc[n] = nu[n][1];
    nu_scale[n] = nu[n][2];
    }
  for (t in 1:nt){
    for (n in 1:nsubj) {
      //update prior using data
      like1 = exp(normal_lpdf(y[n, t] | mu[1] + nu_loc[n], sig_e[1] * sqrt(exp(nu_scale[n])))); // local evidence for class 1
      like2 = exp(normal_lpdf(y[n, t] | mu[2] + nu_loc[n], sig_e[2] * sqrt(exp(nu_scale[n])))); // local evidence for class 2
      p1 = F[n, t] * like1; // joint for class 1 
      p2 = (1 - F[n, t]) * like2; // joint for class 2 (used for normalizing constant)
      F[n, t] = p1 / (p1 + p2); // update filtered belief state
      
      
      //predict forward one timestep
      if (t != nt) {
        p1 = F[n, t] * tp[1] + (1 - F[n, t]) * (1 - tp[2]); // one-step ahead predictive prob for class 1
        p2 = F[n, t] * (1 - tp[1]) + (1 - F[n, t]) * tp[2]; // one-step ahead predictive prob for class 2
        F[n, t+1] = p1 / (p1 + p2); // prep for computing update next cycle 
        pred[n,t+1] = F[n,t+1]; // prep for likelihood computation 
        }
      }
    }
  }
}

model {
// declare temp for log component densities
real ps;    
// Priors
p0 ~ uniform(0, 1);
tp ~ uniform(0, 1);
mu ~ normal(0, 100);
chol_ls ~ lkj_corr_cholesky(1);
for(n in 1 : nsubj){
  theta[n] ~ normal(0, 1);
}
// Likelihood
for (n in 1 : nsubj){
  for (t in 1:nt) {
    ps = pred[n, t] * exp(normal_lpdf(y[n, t] | mu[1] + nu_loc[n], sig_e[1] * sqrt(exp(nu_scale[n])))) +
         (1 - pred[n, t]) * exp(normal_lpdf(y[n, t] | mu[2] + nu_loc[n], sig_e[2] * sqrt(exp(nu_scale[n]))));
    target += log(ps);
    }
  }
}
