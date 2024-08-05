from .cifar100 import get_cifar100


# Data
def get_dataloader(
    data_name='cifar10',
    batch_size=256,
    num_workers=4,
    split=(0.9, 0.1)    
):
    print('==> Preparing data..')

    if data_name == "cifar100":
        return get_cifar100(batch_size, num_workers, split)

