import nest_asyncio
nest_asyncio.apply()
import pystan
import arviz as az
import torch
import pandas as pd

def _mcmc():

    mccode= """
    data {
    int<lower=0> N; // number of individuals
    int<lower=0> M; // number of items
    vector[M] y[N]; //Y matrix of M items for N individuadls
    real lam_mean; //prior mean for lambda
    real<lower=0> lam_sig2; //prior variance for lambda
    real nu_mean; //prior mean for nu 
    real<lower=0> nu_sig2; //prior variance for nu 
    real<lower=0> sig2_shape; 
    real<lower=0> sig2_rate; 
    real<lower=0> psi_shape;  
    real<lower=0> psi_rate;
    }

    parameters {
    vector[N] eta_norm; // normalized eta for each individual
    vector[M] nu; // int for item m
    vector<lower=0>[M-1] lam; // loading item m, fixing the first to be 1
    real <lower=0> sig2; // var of the factor
    vector<lower=0>[M] psi; // sd of error
    }
    transformed parameters{
    vector[N] eta;
    real sigma;
    sigma = sqrt(sig2);
    eta = sigma*eta_norm; 
    }

    model{
    vector[M] mu[N];
    matrix[M,M] Sigma;
    
    real cond_sd_lambda[M-1];
    vector[M] lambda;
    
    eta_norm ~ normal(0,1) ;
    lambda[1] = 1;
    lambda[2:M] = lam;

    sig2 ~ inv_gamma(sig2_shape, sig2_rate);
    
    for(m in 1:M){
        psi[m]~ inv_gamma(psi_shape, psi_rate);
        nu[m] ~ normal(0,sqrt(nu_sig2));    
    }
    
    for(m in 1:(M-1) ){
        cond_sd_lambda[m] = sqrt(lam_sig2*psi[m+1]);
        lam[m] ~ normal(lam_mean,cond_sd_lambda[m]);
    }
    
    for(i in 1:N){   
        mu[i] = nu + lambda*eta[i];    
    }
    
    Sigma =  diag_matrix(psi);
    
    y ~ multi_normal(mu,Sigma); 
    }
        """
    #Build posterior 
    model = pystan.StanModel(model_code = mccode) 
    return model

def single_factor_cfa_mcmc(y_data: torch.tensor, hyper_params) -> pd.DataFrame:
    data = {
        "y": y_data.numpy(), 
        "N": y_data.size(0), 
        "M": y_data.size(1), 
        **hyper_params
    }

    # h = {var:param.item() for var,param in hyper.items()}

    model = _mcmc()
    #Sample from posterior 
    fit = model.sampling(
        data = data, 
        chains = 4, 
        warmup = 7500, 
        iter = 15000, 
        thin = 1 # no thinning
        )
    return fit.to_dataframe()

    
    