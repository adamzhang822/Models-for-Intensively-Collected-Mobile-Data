data {
  int<lower=1> N; //number of data points
  int<lower=1> nsubj; //number of subjects
  int<lower=1, upper=nsubj> subject[N]; //indicator for subjects
  row_vector[3] X_mean[N]; //design matrix for fixed effect
  row_vector[3] X_var[N]; //design matrix for fixed effect
  real y[N]; //outcome
}

parameters {
  vector[3] beta; //fixed effect for mean
  real<lower=0> sigma_subj_loc; //fixed effect for variance
  vector[3] tau; // fixed efect for subject level location RE variance
  real tau_ell; // fixed linear effect of subject-level location effect on within-subject variance 
  real<lower=0> sigma_subj_scale; // scale RE sd at subject level
  real z_subj_loc[nsubj]; // standardized random location effect  at subject level
  real z_subj_scale[nsubj]; // standardized random scale effect at subject level
}

model {
  //REs
  for(i in 1:nsubj){
    z_subj_loc[i] ~ normal(0,1);
    z_subj_scale[i] ~ normal(0,1);
  }
  
  // likelihood
  for (i in 1 : N) {
    y[i] ~ normal(X_mean[i] * beta + sigma_subj_loc * z_subj_loc[subject[i]],
    sqrt(exp(X_var[i] * tau + tau_ell * z_subj_loc[subject[i]] + sigma_subj_scale * z_subj_scale[subject[i]])));
    }
}
