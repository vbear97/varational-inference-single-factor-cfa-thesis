from dataclasses import dataclass
from typing import Optional
from torch.distributions import MultivariateNormal as mvn 
from .pdfs import InverseGamma

@dataclass 
class SingleCFAVariationalParameters:
    nu: mvn 
    lam: mvn 
    psi: InverseGamma
    sig2: InverseGamma
    eta: Optional[mvn] = None