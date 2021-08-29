data {
  int<lower=1> N; //number of data points
  int<lower=1> nsubj; //number of subjects
  int<lower=1> nobs; // number of obs per subject 
  int<lower=1, upper=nsubj> subject[N]; //indicator for subjects
  row_vector[3] X_mean[nobs,nsubj]; //design matrix for fixed effect
  row_vector[3] X_var[nobs,nsubj]; //design matrix for fixed effect
  vector[nobs] y[nsubj]; //outcome
  matrix<lower=0>[nobs,nobs] ar1_pow; // AR1 matrix skeleton
}

parameters {
  vector[3] beta; //fixed effect for mean
  real<lower=0> sigma_subj_loc; //fixed effect for variance
  vector[3] tau; // fixed efect for subject level location RE variance
  real tau_ell; // fixed linear effect of subject-level location effect on within-subject variance 
  real<lower=0> sigma_subj_scale; // scale RE sd at subject level
  real z_subj_loc[nsubj]; // standardized random location effect  at subject level
  real z_subj_scale[nsubj]; // standardized random scale effect at subject level
  real<lower=0,upper=1> rho; // AR(1) param
}

transformed parameters{
  matrix<lower=0>[nobs,nobs] cov_mats[nsubj];
  matrix<lower=0>[nobs,nobs] cov_mat;
  vector<lower=0>[nobs] sds;
  matrix<lower=0,upper=1>[nobs,nobs] ar1;
  vector[nobs] mus[nsubj];
  vector[nobs] mu;
  
  for(i in 1:nobs){
    for(j in 1:nobs){
      ar1[i,j] = pow(rho,ar1_pow[i,j]);
    }
  }
  
  for(i in 1:nsubj){
    for(j in 1:nobs){
      mu[j] = X_mean[j,i] * beta + sigma_subj_loc * z_subj_loc[subject[i]];
    }
    mus[i] = mu;
  }
  
  for(i in 1:nsubj){
    for(j in 1:nobs){
      sds[j] = X_var[j,i] * tau + tau_ell * z_subj_loc[subject[i]] + sigma_subj_scale * z_subj_scale[subject[i]];
    }
    cov_mat = sds * sds';
    cov_mats[i] = ar1 .* cov_mat;
  }
}

model {
  //REs
  int begin;
  int end;
  
  rho ~ uniform(0,1);
  
  for(i in 1:nsubj){
    z_subj_loc[i] ~ normal(0,1);
    z_subj_scale[i] ~ normal(0,1);
  }
  // likelihood
  for(i in 1:nsubj){
    begin = (nobs*(i-1)) + 1;
    end = nobs * i;
    y[i] ~ multi_normal(mus[i],cov_mats[i]);
  }
}
