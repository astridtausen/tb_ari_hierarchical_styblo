data {
  int<lower=1> N_years;
  array[N_years] real x_time; // Replaces distance matrix with vector to use gp_exp_quad_cov() instead
  
  int<lower=1> N_regions;
  int<lower=1> N_subregions;
  int<lower=1> N_countries;
  
  array[N_countries] int<lower=1, upper=N_subregions> country_to_subregion;
  array[N_subregions] int<lower=1, upper=N_regions> subregion_to_region;
  
  // ARI data
  int<lower=1> N_ari;
  vector[N_ari] log_ari;
  vector[N_ari] ari_se;
  array[N_ari] int ari_year;
  array[N_ari] int ari_country;
  
  // Prevalence data
  int<lower=1> N_prev;
  vector[N_prev] log_prev;
  vector[N_prev] prev_se;
  array[N_prev] int prev_year;
  array[N_prev] int prev_country;
  
  // 1 = prior only, 0 = posterior too
  int<lower=0, upper=1> prior_only;
}

parameters {
  // GP hyperparameters
  real<lower=0> sigma_K;
  real<lower=0> l;
  real alpha; 
  
  // Non-centered GP latent variables: T x C matrix, the standard normal variables that we multiply the cholesky factor by
  matrix[N_years, N_countries] eta; 
  
  // Non-centered hierarchical Styblo ratio
  real mu_global_ratio;
  
  real<lower=0> sigma_region;
  vector[N_regions] region_offset_raw; // Standard normal
  
  real<lower=0> sigma_subregion;
  vector[N_subregions] subregion_offset_raw; // Standard normal
  
  real<lower=0> sigma_country;
  vector[N_countries] country_offset_raw; // Standard normal
}

transformed parameters {
  matrix[N_years, N_countries] k; 
  
  vector[N_regions] region_ratio;
  vector[N_subregions] subregion_ratio;
  vector[N_countries] log_ratio;
  
  // Styblo ratios
  region_ratio = mu_global_ratio + region_offset_raw * sigma_region;
  
  for (s in 1:N_subregions) {
    subregion_ratio[s] = region_ratio[subregion_to_region[s]] + subregion_offset_raw[s] * sigma_subregion;
  }
  
  for (c in 1:N_countries) {
    log_ratio[c] = subregion_ratio[country_to_subregion[c]] + country_offset_raw[c] * sigma_country;
  }
  
  // GP
  { // Define covariance matrix
    matrix[N_years, N_years] SIGMA = gp_exp_quad_cov(x_time, sigma_K, l);
    
    // Cholesky decomposition of covariance matrix to Cholesky factor L_K
    matrix[N_years, N_years] L_K = cholesky_decompose(add_diag(SIGMA, 1e-6));
    
    // k is the matrix of latent function values, ie. "true" (zero-centered) ARI trajectories!  
    // which is the product of the cholesky factor and independent noise (standard normal)
    k = L_K * eta; 
  }
}

model {
  // Priors
  alpha ~ normal(-4.0, 1.5); 
  sigma_K ~ lognormal(log(0.5), 2); 
  l ~ lognormal(log(2), 2); 
  to_vector(eta) ~ std_normal(); 
  
  // Styblo (mu=1.678, sigma=0.371)
  mu_global_ratio ~ normal(1.678, 0.371);
  
  sigma_region ~ exponential(1.0);
  region_offset_raw ~ std_normal();
  
  sigma_subregion ~ exponential(1.0);
  subregion_offset_raw ~ std_normal();
  
  sigma_country ~ exponential(1.0);
  country_offset_raw ~ std_normal();
  
  // Likelihoods
  if (prior_only == 0) {
    for (n in 1:N_ari) {
      log_ari[n] ~ normal(alpha + k[ari_year[n], ari_country[n]], ari_se[n]);
    }
    
    for (n in 1:N_prev) {
      log_prev[n] ~ normal(alpha + k[prev_year[n], prev_country[n]] - log_ratio[prev_country[n]], prev_se[n]);
    }
  }
}

generated quantities {
  vector[N_ari] log_ari_rep;
  vector[N_prev] log_prev_rep;
  
  for (n in 1:N_ari) {
    log_ari_rep[n] = normal_rng(alpha + k[ari_year[n], ari_country[n]], ari_se[n]);
  }
  
  for (n in 1:N_prev) {
    log_prev_rep[n] = normal_rng(alpha + k[prev_year[n], prev_country[n]] - log_ratio[prev_country[n]], prev_se[n]);
  }
}
