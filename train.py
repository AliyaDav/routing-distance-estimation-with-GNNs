import numpy as np
import time
from tqdm import tqdm
import os

import torch
import torch.optim as optim
from options import get_options
import wandb
import json

import torch.nn.functional as F 
from torch.nn import Linear, BatchNorm1d, ModuleList, MSELoss
from torch_geometric.nn import TransformerConv, TopKPooling, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from utils.pytorchtools import EarlyStopping
from VRP_dataset import VRPDataset
from torch_geometric.loader import DataLoader

from utils.functions import move_to, calc_loss, get_inner_model, torch_load_cpu

from models.GNN import GNN
from models.GATNar import GATnar
from models.mlp import MLPdecoder

import warnings
warnings.filterwarnings("ignore", message="indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead.")


opts=get_options()

torch.manual_seed(opts.seed)
np.random.seed(opts.seed)

wandb.init(project="Train-mixed")
early_stopping = EarlyStopping(patience=opts.patience, verbose=True)

# opts.device = str(torch.device("cuda:0" if opts.use_cuda else "mps"))
if torch.cuda.is_available():
    opts.device = torch.device('cuda:0')
else:
    opts.device = torch.device('cpu')

if not os.path.exists(opts.save_dir):
    os.makedirs(opts.save_dir)

train_dataset = VRPDataset('vrp_data_large/train')
val_dataset = VRPDataset('vrp_data_large/validation')

opts.epoch_size = len(train_dataset)

train_dataloader = DataLoader(train_dataset, follow_batch = ['x', 'y'],
                                batch_size=opts.batch_size, 
                                num_workers=opts.num_workers,
                                shuffle=True, 
                                drop_last=True, 
                                generator=torch.Generator().manual_seed(1234)
                                )

valid_dataloader = DataLoader(val_dataset, follow_batch = ['x', 'y'],
                                batch_size=opts.batch_size, 
                                num_workers=opts.num_workers,
                                shuffle=True, 
                                drop_last=True, 
                                generator=torch.Generator().manual_seed(1234)
                                )


model=GATnar(opts)
# model=MLPdecoder(20, 128, 'batch')

# load data from checkpoint
# load_data = {}
# load_path = '/Users/aliyadavletshina/Desktop/thesis/routing-distance-estimation-with-GNNs/outputs/tsp_20221031T180004/epoch-21.pt'
# if load_path is not None:
#     print('\nLoading data from {}'.format(load_path))
# load_data =torch_load_cpu(load_path)

# # # Overwrite model parameters by parameters to load
# model_ = get_inner_model(model)
# model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

wandb.config.update(opts)

# with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
#     json.dump(vars(opts), f, indent=True)

def validate(model, valid_dataloader):

    print(f'\nValidating...')
    model.eval()
    opt_gap = []
    rmse_val = []

    rmse = torch.nn.MSELoss()

    for idx, bat in enumerate(valid_dataloader):
        with torch.no_grad():
            targets = bat.y.float()

            graph_pred = model(bat.x.float(),
                    bat.edge_index, 
                    bat.edge_attr.float(),
                    bat.batch
                    )
            
            opt_gap_batch = calc_loss(graph_pred.squeeze(), targets)
            opt_gap.append(opt_gap_batch)

            rmse_batch = rmse(graph_pred.squeeze(), targets)
            rmse_val.append(rmse_batch)

        wandb.log({f'Validation targets': targets, 'Validation preds': graph_pred.squeeze()})

    return np.array(opt_gap).mean(), np.array(rmse_val).mean()

def train_epoch(
        epoch, 
        model, 
        train_dataloader,
        valid_dataloader,
        optimizer, 
        lr_scheduler,
        opts
    ):
    print("\nStart train epoch {}".format(epoch))
    start_time = time.time()    
    step = epoch * (opts.epoch_size // opts.batch_size)
    
    model.train()
    optimizer.zero_grad()

    for batch_id, batch in enumerate(train_dataloader):

        graph_pred = model(batch.x.float(), 
                    batch.edge_index, 
                    batch.edge_attr.float(),
                    batch.batch
                    ) 

        targets = batch.y.float()

        loss = calc_loss(graph_pred.squeeze(), targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        step += 1

        if step % int(opts.log_step) == 0:
            wandb.log({'Step MAPE': loss})

    lr_scheduler.step(epoch)
    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        saving_path = os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        print(f'Saving model and state to {saving_path}...')
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all()
            },
            saving_path
        )

    epoch_mape, epoch_rmse = validate(model, valid_dataloader)
        
    # wandb.log({'Epoch validation MAPE': epoch_mape, 'Epoch validation RMSE': epoch_rmse})
    return epoch_mape, epoch_rmse

optimizer = optim.Adam([{'params': model.parameters(), 'lr': opts.lr_model}])
lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

wandb.watch(model)
print(model)
nb_param = 0
for param in model.parameters():
    nb_param += np.prod(list(param.data.size()))
print('Number of parameters: ', nb_param)

for epoch in range(opts.n_epochs):
    epoch_mape, epoch_rmse = train_epoch(
        epoch, 
        model, 
        train_dataloader,
        valid_dataloader,
        optimizer, 
        lr_scheduler,
        opts
    )
    early_stopping(epoch_mape, model)

    wandb.log({'Epoch validation MAPE': epoch_mape, 'Epoch validation RMSE': epoch_rmse})

    if early_stopping.early_stop:
        print("Early stopping")
        break
