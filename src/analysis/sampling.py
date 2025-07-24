from dataclasses import dataclass, asdict
from typing import Dict
import pandas as pd 
import numpy as np
import torch
from torch.distributions import MultivariateNormal as mvn 

def sample_from_distribution(dist_by_var: Dict[str, torch.distributions.Distribution], n: int = 60_000) -> pd.DataFrame:
    '''Sample n times from optimised variational distributions'''
    samples_by_scalar = {}
    for var, dist in dist_by_var.items(): 
        if dist:
            sample = dist.rsample(torch.Size([n])).detach().numpy()
            if sample.ndim>1: 
                #If multidimensional
                for i in range(sample.shape[1]): 
                    samples_by_scalar[var+"."+ str(i+1)] = sample[:,i]
            else: 
                samples_by_scalar[var] = sample
    return pd.DataFrame(samples_by_scalar)