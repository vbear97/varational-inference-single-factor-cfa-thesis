from abc import ABC, abstractmethod
from typing import Dict
import torch
from torch.distributions import MultivariateNormal, Normal

from .variational_distributions import InverseGamma

class BayesianStatisticalModel(ABC): 
    """Template class for bayesian statistical models."""

    def __init__(self, y_data, hyper_params, degenerates: list[str] = None):
        super().__init__()
        self.y_data = y_data.clone().detach()
        self.hyper_params = hyper_params
        self.degenerates = degenerates or []

    @abstractmethod
    def log_likelihood(self, params): 
        '''Compute the log-likelihood of data given model params.'''
        pass 

    @abstractmethod
    def log_prior(self, params): 
        '''Compute log prior density of model params.'''
        pass 

class SingleFactorCFA(BayesianStatisticalModel): 
    '''Single Factor Confirmatory Factor Analysis Model with lambda_1 set to 1 for identifiability'''
    
    #Set lambda_1 to 1 for model identifiability purposes
    lam1= torch.tensor([1.0])

    def __init__(self, y_data, hyper_params, degenerates):
        super().__init__(y_data, hyper_params, degenerates)

    def log_likelihood(self, params: Dict[str, torch.tensor]):
        #TODO: Impose an expected shape on the parameters and hyper-parameters
        '''Calculate log-likelihood according to multivariate normal'''
        #Covariance matrix 
        covariance = torch.diag(params['psi'])
        
        #Lambda_1 is fixed to 1 for model explainability 
        lam_full = torch.cat((self.lam1, params['lam']))
        
        #Means 
        like_dist_means = params['nu'] + torch.matmul(params['eta'].unsqueeze(1), lam_full.unsqueeze(0))
    
        return MultivariateNormal(like_dist_means, covariance_matrix= covariance).log_prob(self.y_data).sum()
    
    def log_prior(self, params: Dict[str, torch.tensor]):
        priors = {'nu': Normal(loc = self.hyper_params['nu_mean'], scale = torch.sqrt(self.hyper_params['nu_sig2'])), \
            
        'sig2': InverseGamma(concentration = self.hyper_params['sig2_shape'], rate = self.hyper_params['sig2_rate']),\

        'psi': InverseGamma(concentration = self.hyper_params['psi_shape'], rate= self.hyper_params['psi_rate']),\
           
        'eta': Normal(loc = 0, scale = torch.sqrt(params['sig2'])),\

        'lam': Normal(loc = self.hyper_params['lam_mean'], \
            scale = torch.sqrt(self.hyper_params['lam_sig2']*(params['psi'][1:])))
        }
        log_priors = {var: priors[var].log_prob(params[var]).sum() for var in priors if var not in self.degenerates}

        return sum(log_priors.values())
    

