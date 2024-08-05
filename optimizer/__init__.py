from torch.optim import SGD
from .sam import SAM


def get_optimizer(
    net,
    opt_name='sam',
    opt_hyperparameter={}):
    if opt_name == 'sam':
        return SAM(
            net.parameters(), 
            **opt_hyperparameter
        )
    elif opt_name == 'sgd':
        return SGD(
            net.parameters(), 
            **opt_hyperparameter
        )
    else:
        raise ValueError("Invalid optimizer!!!")