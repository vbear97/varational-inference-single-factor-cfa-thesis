''''Functions to enable sampling from optimised distributions of any given VI model.'''
from dataclasses import dataclass, asdict
import pandas as pd 
import numpy as np
import torch
from torch.distributions import MultivariateNormal as mvn 

from ..pdfs import InverseGamma

@dataclass 
class SingleCFAVariationalParameters:
    nu: mvn 
    lam: mvn 
    psi: InverseGamma
    sig2: InverseGamma

def create_sample_from_qvar(qvar: SingleCFAVariationalParameters, n: int = 60000) -> pd.DataFrame: 
    '''Sample n times from optimised variational distributions'''
    samples_by_scalar = {}
    qvar = asdict(qvar)
    for param, distribution in qvar.items(): 
        sample = distribution.rsample(torch.Size([n])).detach().numpy()
        if sample.shape[1] >1: 
            for i in range(sample.shape[1]): 
                samples_by_scalar[param+"."+ str(i+1)] = sample[:,i]
        else: 
            samples_by_scalar[param] = sample[:,0]
    
    return pd.DataFrame(samples_by_scalar)