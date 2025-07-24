import torch
from torch.distributions import Gamma

class InverseGamma():
    r'''
    Creates a one dimensional Inv-Gamma Distribution parameterised by concentration and rate, where: 

    X ~ Gamma(concentration, rate)
    Y = 1/X ~  InvGamma(concentration, rate)

    Args: 
    concentration, rate (float or Tensor): concentration, rate of the Gamma distribution
    '''
    def __init__(self, concentration, rate, validate_args = None): 
        self.base_dist = Gamma(concentration = concentration, rate = rate, validate_args=None)
        self.concentration = concentration
        self.rate = rate
    
    def log_prob(self,y):
        #1/0 not a problem here, since log_prob will only be evaluated on theta_sample
        abs_dj = torch.square(torch.reciprocal(y))
        y_rec = torch.reciprocal(y)
        return self.base_dist.log_prob(y_rec) + torch.log(abs_dj)
    
    def rsample(self, n= torch.Size([])): 
        base_sample = self.base_dist.rsample(n)
        return torch.reciprocal(base_sample)