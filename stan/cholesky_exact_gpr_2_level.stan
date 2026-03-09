data {
  int<lower=1> N_years;
  int<lower=1> N_countries;
  
  // ARI
  int<lower=1> N_ari;
  vector[N_ari] log_ari;
  vector[N_ari] ari_se;
  array[N_ari] int ari_year;
  array[N_ari] int ari_country;
  
  // Prevalence
  int<lower=1> N_prev;
  vector[N_prev] log_prev;
  vector[N_prev] prev_se;
  array[N_prev] int prev_year;
  array[N_prev] int prev_country;
  
  // Distance matrix (T x T)
  matrix[N_years, N_years] distance_matrix;
  
  // 1 = prior only, 0 = evaluate likelihood
  int<lower=0, upper=1> prior_only;
}

parameters {
  // GP Hyperparameters
  real<lower=0> sigma_K;
  real<lower=0> l;
  real alpha; 
  
  // T x C matrix
  matrix[N_years, N_countries] eta; 
  
  // Hierarchical ratio parameters
  real mu_ratio;
  real<lower=0> sigma_ratio;
  vector[N_countries] log_ratio;
}

transformed parameters {
  // T x C matrix
  matrix[N_years, N_countries] k; 
  
  // Prevent saving covariance matrix to memory
  {
    matrix[N_years, N_years] SIGMA;
    matrix[N_years, N_years] L_K;
    
    // Build squared exponential covariance matrix
    // Note: distance_matrix is already squared in R
    for (i in 1:(N_years-1)) {
      for (j in (i+1):N_years) {
        SIGMA[i, j] = square(sigma_K) * exp(-distance_matrix[i, j] / (2.0 * square(l)));
        SIGMA[j, i] = SIGMA[i, j]; 
      }
    }
    
    // Fill diagonal with variance + numerical nugget for Cholesky stability
    for (i in 1:N_years) {
      SIGMA[i, i] = square(sigma_K) + 1e-6; 
    }
    
    L_K = cholesky_decompose(SIGMA);
    
    // Matrix multiplication: (T x T) * (T x C) = (T x C)
    k = L_K * eta; 
  }
}

model {
  // Priors
  alpha ~ normal(-4.0, 1.5); 
  sigma_K ~ lognormal(log(0.5), 2); 
  l ~ lognormal(log(2), 2); 
  
  to_vector(eta) ~ std_normal(); 
  
  mu_ratio ~ normal(1.678, 1.0); 
  sigma_ratio ~ exponential(1.0); 
  log_ratio ~ normal(mu_ratio, sigma_ratio); 

// Likelihoods
  if (prior_only == 0) {
    // Map the matrix k to observation vectors
    for (n in 1:N_ari) {
      log_ari[n] ~ normal(alpha + k[ari_year[n], ari_country[n]], ari_se[n]);
    }
    
    for (n in 1:N_prev) {
      log_prev[n] ~ normal(alpha + k[prev_year[n], prev_country[n]] - log_ratio[prev_country[n]], prev_se[n]);
    }
  }
}

generated quantities {
  // Vectors to hold simulated data
  vector[N_ari] log_ari_rep;
  vector[N_prev] log_prev_rep;
  
  // Simulate data
  for (n in 1:N_ari) {
    log_ari_rep[n] = normal_rng(alpha + k[ari_year[n], ari_country[n]], ari_se[n]);
  }
  
  // Simulate prevalence data points using shifted posterior estimates
  for (n in 1:N_prev) {
    log_prev_rep[n] = normal_rng(alpha + k[prev_year[n], prev_country[n]] - log_ratio[prev_country[n]], prev_se[n]);
  }
}
