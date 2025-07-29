from abc import ABC, abstractmethod
from typing import Dict, Literal
import torch

from .custom_types import SINGLE_CFA_VARIABLES_LITERAL
from .variational_distributions import Degenerate, InvGamma, Normal, VariationalDistribution

class MeanFieldVariationalFamily(ABC): 
    '''Template class for variational family, assuming that each model parameter is independently distributed (mean field assumption)'''
    def __init__(self, qvar: Dict[str, VariationalDistribution]):
        '''Instantiate variational family'''
        self.qvar_by_var = qvar 

    def generate_theta_sample(self) -> Dict[str, torch.tensor]:
        '''Generate single sample of Bayesian model parameters given current variational distribution, assuming independence'''
        return {var: qvar.rsample() for (var,qvar) in self.qvar_by_var.items()}
    
    def entropy(self, theta_sample) -> torch.tensor:
        qvar_prob = {var: self.qvar_by_var[var].log_prob(sample) for (var,sample) in theta_sample.items()}
        return sum(qvar_prob.values())
    
    def scalar_param_values(self) -> Dict[str, torch.tensor]:
        '''Extract parameter values without gradient information'''
        return {key: self.qvar_by_var[key].var_params.clone().detach() for key in self.qvar_by_var}
    
    @abstractmethod 
    def natural_param_values(self) -> Dict[str, float]:
        '''Extract natural variational parameter values'''
        pass

class SingleCFAVariationalFamily(MeanFieldVariationalFamily): 
    '''Single CFA Variational Family with default starting point'''
    def __init__(
        self, 
        m: int, 
        n: int, 
        degenerates: Dict[SINGLE_CFA_VARIABLES_LITERAL, torch.tensor] = None,
        qvar: Dict[SINGLE_CFA_VARIABLES_LITERAL, VariationalDistribution] = None
        ):
        if not qvar: 
            qvar = {
                'nu': Normal(m, mu= torch.zeros(m), log_s = torch.zeros(m)), 
                'lam': Normal(m-1, mu = torch.ones(m-1), log_s = torch.zeros(m-1)), 
                'eta': Normal(n, mu = torch.zeros(n), log_s = torch.ones(n)), 
                'psi': InvGamma(m, log_alpha = torch.ones(m), log_beta = torch.ones(m)), 
                'sig2': InvGamma(1, log_alpha = torch.tensor(1.00), log_beta = torch.tensor(1.00))
            }
        self.degenerates = degenerates or {}
        for param_name, degenerate_value in degenerates.items(): 
            qvar[param_name] = Degenerate(fixed_value = degenerate_value)

        super().__init__(qvar)
    
    def natural_param_values(self) -> Dict[str, float]:
        scalars = {}
        # Handle nu parameters (if not degenerate)
        if 'nu' not in self.degenerates:
            for i in range(len(self.qvar_by_var['nu'].var_params[0])):
                scalars[f'nu{i+1}_mean'] = self.qvar_by_var['nu'].var_params[0][i].item() 
                scalars[f'nu{i+1}_sig'] = self.qvar_by_var['nu'].var_params[1][i].exp().item()
        
        # Handle lambda parameters (if not degenerate)
        if 'lam' not in self.degenerates:
            for i in range(len(self.qvar_by_var['lam'].var_params[0])):
                scalars[f'lambda{i+2}_mean'] = self.qvar_by_var['lam'].var_params[0][i].item()
                scalars[f'lambda{i+2}_sig'] = self.qvar_by_var['lam'].var_params[1][i].exp().item()
        
        # Handle psi parameters (if not degenerate)
        if 'psi' not in self.degenerates:
            for i in range(len(self.qvar_by_var['psi'].var_params[0])):
                scalars[f'psi_{i+1}_alpha'] = self.qvar_by_var['psi'].var_params[0][i].exp().item()
                scalars[f'psi_{i+1}_beta'] = self.qvar_by_var['psi'].var_params[1][i].exp().item()
        
        # Handle sig2 parameters (if not degenerate)
        if 'sig2' not in self.degenerates:
            scalars['sig2_alpha'] = self.qvar_by_var['sig2'].var_params[0].exp().item()
            scalars['sig2_beta'] = self.qvar_by_var['sig2'].var_params[1].exp().item()
        
        # Handle eta parameters (if not degenerate)
        if 'eta' not in self.degenerates:
            # Add eta parameters if they exist and follow similar pattern
            # TODO: Add in option 
            pass
            
        return scalars  
