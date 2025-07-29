from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Literal, TypedDict
import torch
from dataclasses import asdict, dataclass
from tqdm import trange
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter

from .analysis.sampling import sample_from_distribution

from .variational_family import MeanFieldVariationalFamily, SingleCFAVariationalFamily
from .statistical_model import BayesianStatisticalModel, SingleFactorCFA

@dataclass 
class VIOptimisationParameters: 
    num_iterations: float = 20000
    relative_error_threshold: float = 10e-4
    patience: int = 100

@dataclass 
class ConvergenceData: 
    relative_errors: list[float] = None 
    convergence_time: float = None 
    num_iterations: float = None

@dataclass 
class VIOptimisationResults: 
    opt_params: VIOptimisationParameters
    convergence_data: ConvergenceData

class SchedulerParams(TypedDict, total = False): 
    """Parameters for ReducedLROnPlateau scheduler"""
    mode: str
    threshold_mode: str
    factor: float
    patience: int
    threshold: float
    cooldown: int
    min_lr: float
    eps: float

class VIModel(ABC):
    '''Template class for performing mean-field variational inference, given a Bayesian statistical model and given a mean field variational family'''

    DEFAULT_SCHEDULER_PARAMS = {
        'mode': 'min',
        'threshold_mode': 'rel',
        'factor': 0.1,
        'patience': 1000,
        'threshold': 0.0001,
        'cooldown': 0,
        'min_lr': 0.0,
        'eps': 1e-08,
        }

    def __init__(self, model: BayesianStatisticalModel, qvar: MeanFieldVariationalFamily):
        super().__init__()
        self.model = model 
        self.qvar = qvar 
        self.is_fitted = False 
        self.results: VIOptimisationResults = None

    def _elbo(self): 
        '''Calculate the evidence lower bound'''
        theta_sample = self.qvar.generate_theta_sample()
        return self.model.log_likelihood(theta_sample) + self.model.log_prior(theta_sample) - self.qvar.entropy(theta_sample)
    
    def _elbo_multi(self, K: int = 10): 
        '''Calculate evidence lower bound by averaging over K realisations from qvar'''
        elbos = torch.stack([self._elbo() for k in range(K)])
        return elbos.mean()

    def vr_bound(self, K:int = 10, alpha: float = 0.5): 
        '''
        Calculate the generalised VR bound (generalisation of ELBO), wich is used to perform Renyi Divergence Variational Inference. 
        Technically, only values larger than 0 correspond to valid Reny divergences. 

        However, the algorithm is totally feasible using alpha < 0, even though it does not corrrespond to a valid divergence. 
        '''
        if alpha <1: 
            logw = torch.stack([self._elbo() for k in range(K)])
            logw = (1-alpha)*logw
            c = logw.max()
            logw_correct = logw - c
            lse = c + torch.logsumexp(logw_correct, dim = 0) #equal to logsumexp(logw), via offsetting
            lse_av = lse - torch.log(torch.tensor(K)) #divide by K to take average.
            vr_bound = lse_av/(1-alpha)
            return vr_bound
        
        elif (alpha== 1): 
            #alpha = 1 is equivalent to KL Divergence Variational Inference
            return self._elbo_multi(K = K)
            
        else: 
            raise ValueError('Alpha must be < 1')
            
    @abstractmethod
    def create_optimizer(self): 
        '''Customise optimizer object for each variational family'''
        pass 

    # @property 
    # @abstractmethod
    # def sample_qvar(self, **kwargs): 
    #     '''Sample each statistical model parameter from its optimised variational distribution'''
    #     pass 

    def _rel_error(self, prev, next): 
        rel = {key: ((next[key] - prev[key])/prev[key]).abs().max() for key in prev}
        return max(rel.values())

    def optimize(self, optimisation_parameters: VIOptimisationParameters, filename: str = 'OptimiseVIModel', K: int = 10, alpha: float = 0.5, optimizer: torch.optim.Optimizer = None, scheduler_params:    SchedulerParams = None):
        #Initialise optimizer objects
        rel_error = []
        if not optimizer: 
            optimizer = self.create_optimizer()

        if not scheduler_params: 
            scheduler_params = self.DEFAULT_SCHEDULER_PARAMS

        count_small_errors = 0
        prev = None
        iters = trange(optimisation_parameters.num_iterations, mininterval=1)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            **scheduler_params
        )
        
        filename = filename + "_K_" + str(K) + "_alpha" + str(alpha)
        writer = SummaryWriter(filename)    
    
        #Optimisation parameters
        start_time = timer()
        for t in iters:
            #Update gradients
            optimizer.zero_grad()
            loss = -self.vr_bound(K = K, alpha = alpha)
            #Backpropagation
            loss.backward()
            optimizer.step()

            #Record values to tensorboard 
            writer.add_scalars("Natural Variational Parameters", self.qvar.natural_param_values(), global_step = t)
            writer.add_scalar('VR Bound', scalar_value = loss.item(), global_step = t)

            #Tune learning rate by validation loss
            scheduler.step(loss.item())

            #Assess convergence - get the scalar parameter values 
            next = self.qvar.scalar_param_values()
            if (t>1):
                error = self._rel_error(prev, next)
                rel_error.append(error)
                if(error<=optimisation_parameters.relative_error_threshold): 
                    count_small_errors+=1
                    if (count_small_errors >= optimisation_parameters.patience): 
                        print("VI converged at step t = ", t)
                        break 
                else: 
                    #Reset the counter - we need a string of consecutively small errors
                    count_small_errors = 0
            prev=next

        #Update results
        end_time = timer() 
        writer.close()
        self.is_fitted = True
        self.results = VIOptimisationResults(
            opt_params = optimisation_parameters, 
            convergence_data=ConvergenceData(
            relative_errors = rel_error, 
            convergence_time = end_time - start_time, 
            num_iterations=t
            )
        )
    
class SingleCFAVIModel(VIModel): 
    '''VI Model class for single factor confirmatory factor analysis model. '''
        
    def __init__(self, y_data, hyper_params, degenerates: Dict[Literal['nu', 'lam',  'eta', 'psi', 'sig2'], torch.tensor] = None):
        stats_model = SingleFactorCFA(y_data = y_data, hyper_params=hyper_params, degenerates = degenerates.keys())
        qvar = SingleCFAVariationalFamily(m = y_data.shape[1], n = y_data.shape[0], degenerates = degenerates)
        super().__init__(stats_model, qvar)
    
    def create_optimizer(self): 
        optimizer = torch.optim.Adam([{'params': [self.qvar.qvar_by_var['nu'].var_params, self.qvar.qvar_by_var['lam'].var_params], 'lr': 0.01},\
     {'params': [self.qvar.qvar_by_var['psi'].var_params, self.qvar.qvar_by_var['sig2'].var_params], 'lr': 0.1},\
         {'params':[self.qvar.qvar_by_var['eta'].var_params], 'lr': 0.1} 
         ]
         )
        return optimizer 