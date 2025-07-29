from typing import Dict
import pandas as pd 
import torch
from torch.distributions import MultivariateNormal as mvn 

def sample_from_distribution(dist_by_var: Dict[str, torch.distributions.Distribution], n: int = 60_000, latent_only = True) -> pd.DataFrame:
    '''Sample n times from optimised variational distributions, either for all variables or only non-latent variables'''
    samples_by_scalar = {}
    for var, dist in dist_by_var.items(): 
        if var == 'eta' and latent_only: 
            continue
        if dist:
            sample = dist.rsample(torch.Size([n])).detach().numpy()
            if sample.ndim>1: 
                #If multidimensional
                for i in range(sample.shape[1]): 
                    if var!='lam':
                        #Label lambda values correctly, taking into account that lambda.1 is fixed to 1 for model identifiability purposes 
                        samples_by_scalar[var+"."+ str(i+1)] = sample[:,i]
                    else: 
                        samples_by_scalar[var+"."+ str(i+2)] = sample[:,i]
            else: 
                samples_by_scalar[var] = sample
    return pd.DataFrame(samples_by_scalar)
