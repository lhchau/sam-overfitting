import wandb
import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import *
from utils import *
from dataloader import *
from optimizer import *

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

################################
#### 0. SETUP CONFIGURATION
################################
cfg = exec_configurator()
initialize(cfg['trainer']['seed'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc, start_epoch, logging_dict = 0, 0, {}

# Total number of training epochs
EPOCHS = cfg['trainer']['epochs'] 

print('==> Initialize Logging Framework..')
logging_name = get_logging_name(cfg)
logging_name += ('_' + current_time)

framework_name = cfg['logging']['framework_name']
if framework_name == 'wandb':
    wandb.init(project=cfg['logging']['project_name'], name=logging_name, config=cfg)

################################
#### 1. BUILD THE DATASET
################################
train_dataloader, val_dataloader, test_dataloader, num_classes = get_dataloader(**cfg['dataloader'])

################################
#### 2. BUILD THE NEURAL NETWORK
################################
net = get_model(**cfg['model'], num_classes=num_classes)
net = net.to(device)

total_params = sum(p.numel() for p in net.parameters())
print(f'==> Number of parameters in {cfg["model"]}: {total_params}')

################################
#### 3.a OPTIMIZING MODEL PARAMETERS
################################
criterion = nn.CrossEntropyLoss().to(device)
sch = cfg['trainer'].get('sch', None)
optimizer = get_optimizer(net, **cfg['optimizer'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

################################
#### 3.b Training 
################################
if __name__ == "__main__":
    for epoch in range(start_epoch, start_epoch+EPOCHS):
        print('\nEpoch: %d' % epoch)
        loop_one_epoch(
            dataloader=train_dataloader,
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            logging_dict=logging_dict,
            epoch=epoch,
            loop_type='train',
            logging_name=logging_name)
        best_acc = loop_one_epoch(
            dataloader=val_dataloader,
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            logging_dict=logging_dict,
            epoch=epoch,
            loop_type='val',
            logging_name=logging_name,
            best_acc=best_acc)
        scheduler.step()
        
        if framework_name == 'wandb':
            wandb.log(logging_dict)
    
    logging_dict = {}
    loop_one_epoch(
        dataloader=test_dataloader,
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        logging_dict=logging_dict,
        epoch=epoch,
        loop_type='test',
        logging_name=logging_name)
    
    if framework_name == 'wandb':
        wandb.log(logging_dict)
        
        mini_hessian_batch_size = 128
        cfg['dataloader']['batch_size'] = mini_hessian_batch_size
        train_dataloader, _, _, _ = get_dataloader(**cfg['dataloader'])
        figure = get_eigen_hessian_plot(
            name=logging_name, 
            net=net,
            criterion=criterion,
            dataloader=train_dataloader,
            hessian_batch_size=128*20,
            mini_hessian_batch_size=mini_hessian_batch_size
        )
        wandb.log({'train/top5_eigenvalue_density': wandb.Image(figure)})