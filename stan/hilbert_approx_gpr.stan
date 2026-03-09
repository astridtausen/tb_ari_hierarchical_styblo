data {
  int<lower=1> N_years;
  vector[N_years] x_time; 
  
  // Hilbert decisions (number of basis functions and defined boundaries)
  int<lower=1> M; 
  real<lower=1.0> c; 
  
  // Hierarchy of countries and UN regions
  int<lower=1> N_regions;
  int<lower=1> N_subregions;
  int<lower=1> N_countries;
  
  array[N_countries] int<lower=1, upper=N_subregions> country_to_subregion;
  array[N_subregions] int<lower=1, upper=N_regions> subregion_to_region;
  
  // Priors on hyperparameters
  real prior_alpha_mean;
  real prior_alpha_sd;
  real prior_beta_slope_mean;
  real prior_beta_slope_sd;
  real prior_sigma_K_mu;
  real prior_sigma_K_sigma;
  real prior_l_mu;
  real prior_l_sigma;
  real prior_mu_global_ratio_mean;
  real prior_mu_global_ratio_sd;
  real prior_hier_sd;
  
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
  
  // Prior only vs. posterior, linear vs. constant mean trend
  int<lower=0, upper=1> prior_only;
  int<lower=0, upper=1> linear;
}

transformed data {
  // Define the boundary L
  real L = c * max(x_time); 
  
  // Eigendecomposition of Laplace operator
  vector[M] omega;
  matrix[N_years, M] PHI;
  
  for (m in 1:M) {
    omega[m] = (m * pi()) / (2.0 * L); // Eigenvalues (frequencies)
    for (t in 1:N_years) {
      // Eigenfunctions (orthogonal basis mapping)
      PHI[t, m] = (1.0 / sqrt(L)) * sin(omega[m] * (x_time[t] + L));
    }
  }
}

parameters {
  // GP Hyperparameters
  real alpha; 
  real beta_slope;
  real<lower=1e-6> sigma_K; 
  real<lower=1e-6> l;
  
  // Standard normal basis weights (M x C projection parameters)
  matrix[M, N_countries] beta; 
  
  // Non-centered hierarchical offsets
  real mu_global_ratio;
  real<lower=0> sigma_region;
  vector[N_regions] region_offset_raw;
  real<lower=0> sigma_subregion;
  vector[N_subregions] subregion_offset_raw;
  real<lower=0> sigma_country;
  vector[N_countries] country_offset_raw;
}

transformed parameters {
  vector[N_regions] region_ratio;
  vector[N_subregions] subregion_ratio;
  vector[N_countries] log_ratio;
  matrix[N_years, N_countries] k; 
  
  // Affine transformations for ultrametric distances
  region_ratio = mu_global_ratio + region_offset_raw * sigma_region;
  subregion_ratio = region_ratio[subregion_to_region] + subregion_offset_raw * sigma_subregion;
  log_ratio = subregion_ratio[country_to_subregion] + country_offset_raw * sigma_country;
  
  // Functional projection to L^2([-L, L])
  {
    vector[M] SPD;
    matrix[M, N_countries] diag_beta;
    
    for (m in 1:M) {
      // Evaluate the spectral density measure for the exponentiated quadratic kernel
      SPD[m] = square(sigma_K) * sqrt(2.0 * pi()) * l * exp(-0.5 * square(l * omega[m]));
      
      // Scale standard normals by the deterministic spectral amplitude
      diag_beta[m, ] = sqrt(SPD[m]) * beta[m, ];
    }
    
    // Linear projection: k maps time to function space via basis functions
    k = PHI * diag_beta; 
  }
}

model {
  // Priors
  alpha ~ normal(prior_alpha_mean, prior_alpha_sd);
  beta_slope ~ normal(prior_beta_slope_mean, prior_beta_slope_sd);
  sigma_K ~ lognormal(prior_sigma_K_mu, prior_sigma_K_sigma); 
  l ~ lognormal(prior_l_mu, prior_l_sigma); 
  
  to_vector(beta) ~ std_normal(); 
  
  mu_global_ratio ~ normal(prior_mu_global_ratio_mean, prior_mu_global_ratio_sd);
  
  // Regularizing half-normals for structural shrinkage
  sigma_region ~ normal(0, prior_hier_sd);
  region_offset_raw ~ std_normal();
  
  sigma_subregion ~ normal(0, prior_hier_sd);
  subregion_offset_raw ~ std_normal();
  
  sigma_country ~ normal(0, prior_hier_sd);
  country_offset_raw ~ std_normal();
  
  // Likelihoods
  if (prior_only == 0) {
    vector[N_ari] ari_mu;
    for (n in 1:N_ari) {
      ari_mu[n] = alpha 
                  + (linear * beta_slope * x_time[ari_year[n]])
                  + k[ari_year[n], ari_country[n]];
    }
    log_ari ~ normal(ari_mu, ari_se);
    
    vector[N_prev] prev_mu;
    for (n in 1:N_prev) {
      prev_mu[n] = alpha 
                   + (linear * beta_slope * x_time[prev_year[n]])
                   + k[prev_year[n], prev_country[n]] 
                   - log_ratio[prev_country[n]];
    }
    log_prev ~ normal(prev_mu, prev_se);
  }
}

generated quantities {
  vector[N_ari] log_ari_rep;
  vector[N_prev] log_prev_rep;
  
  // Vectors for cross validation
  vector[N_ari] log_lik_ari;
  vector[N_prev] log_lik_prev;
  
  for (n in 1:N_ari) {
    real mu_a = alpha 
                  + (linear * beta_slope * x_time[ari_year[n]])
                  + k[ari_year[n], ari_country[n]];
    log_ari_rep[n] = normal_rng(mu_a, ari_se[n]);
    log_lik_ari[n] = normal_lpdf(log_ari[n] | mu_a, ari_se[n]);
  }
  
  for (n in 1:N_prev) {
    real mu_p = alpha 
                   + (linear * beta_slope * x_time[prev_year[n]])
                   + k[prev_year[n], prev_country[n]] 
                   - log_ratio[prev_country[n]];
    log_prev_rep[n] = normal_rng(mu_p, prev_se[n]);
    log_lik_prev[n] = normal_lpdf(log_prev[n] | mu_p, prev_se[n]);
  }
}
