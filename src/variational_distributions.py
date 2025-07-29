from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch 
from torch import tensor as tensor
from .pdfs import InverseGamma
import pandas as pd

class VariationalDistribution(ABC): 
    '''Template class for variational distribution, essentially a wrapper around a torch distribution object, that implements reparameterisation gradient sampling.'''

    def __init__(self, var_params: tensor):
        self.var_params = var_params
        self.var_params.requires_grad = True
    
    @abstractmethod
    def dist(self) -> Optional[torch.distributions.Distribution]:
        '''Invoke the underlying torch distribution object'''
        pass

    def log_prob(self, x) -> torch.tensor:
        '''Calculate log pdf, assuming independent '''
        return self.dist().log_prob(x).sum()

    def rsample(self, n = torch.Size([])) -> torch.tensor:
        '''Sample using reparameterisation'''
        return self.dist().rsample(n)

class Normal(VariationalDistribution):
    def __init__(self, size: int = 1, mu=None, log_s=None):
        # Take log_standard deviation to allow for unconstrained optimisation.
        if mu is None:
            mu = torch.randn(size)
        if log_s is None:
            log_s = torch.randn(size) 
        # Variational parameters
        super().__init__(torch.stack([mu, log_s]))
        self.size = size 

    def dist(self):
        return torch.distributions.Normal(self.var_params[0], self.var_params[1].exp())

class InvGamma(VariationalDistribution): 
    randomised_offset = 1.0 
    def __init__(self, size: int = 1, log_alpha=None, log_beta=None):
        if log_alpha is None:
            log_alpha = torch.rand(size) + self.randomised_offset 
        if log_beta is None:
            log_beta = torch.rand(size) + self.randomised_offset #log_beta
        self.size = size 
        # Variational parameters
        super().__init__(torch.stack([log_alpha, log_beta]))
    
    def dist(self):
        return InverseGamma(concentration= torch.exp(self.var_params[0]), rate= torch.exp(self.var_params[1]))

class Degenerate(VariationalDistribution): 
    def __init__(self, fixed_value: tensor):
        self.var_params = fixed_value
        #Since the variational parameter is fixed to a point value, we do NOT require gradients.
        self.var_params.requires_grad = False 

    def dist(self): 
        return None 

    def rsample(self): 
        '''Theta sample should return the same degenerated fixed point values'''
        return self.var_params.clone()

    def log_prob(self, x): 
        return torch.tensor([0.0])
    

