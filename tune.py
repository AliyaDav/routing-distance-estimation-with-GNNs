import numpy as np
import time
import json
import torch
import torch.optim as optim
from options import get_options
import wandb

from VRP_dataset import VRPDataset
from torch_geometric.loader import DataLoader
from utils.pytorchtools import EarlyStopping
from models.GATNar import GATnar

import warnings
warnings.filterwarnings("ignore", message="indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead.")

# from train import validate
from utils.functions import calc_loss

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

        # wandb.log({f'Batch validation MAPE': opt_gap_batch, 'rmse_batch': rmse_batch})

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

        # wandb.log({'loss': loss})

    lr_scheduler.step(epoch)
    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    epoch_mape, epoch_rmse = validate(model, valid_dataloader)

    return epoch_mape, epoch_rmse

def run():

    opts=get_options()
    early_stopping = EarlyStopping(patience=opts.patience, verbose=True)
    
    # initialize W&B for sweep
    wandb.init(project="Tuning_mixed_data", config=sweep_configuration)
    # wandb.init()
    opts.lr_model = wandb.config.lr_model
    opts.batch_size=wandb.config.batch_size
    opts.embedding_dim=wandb.config.embedding_dim
    opts.n_decode_layers=wandb.config.n_decode_layers
    opts.n_encode_layers=wandb.config.n_encode_layers
    # opts.n_epochs=10

    wandb.config.update(opts)

    print(vars(opts))
    # load data
    train_dataset = VRPDataset('vrp_data_large/train')
    val_dataset = VRPDataset('vrp_data_large/validation')
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
    
    opts.epoch_size = len(train_dataset)

    # define the model
    model=GATnar(opts)
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': opts.lr_model}])
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # train epochs
    wandb.watch(model)
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

if __name__ == "__main__":

    # opts=get_options()

    # def run():
    #     return run()

    # if opts.sweep:
    #     print('starting sweep....')
    #     sweep_configuration = json.load(open("sweep.json"))
    #     sweep_id = wandb.sweep(sweep_configuration, project="Tuning_mixed_data")
    #     wandb.agent(sweep_id, run, count=100)
    
    # else:
    #     print('running without sweep')
    #     run()

    sweep_configuration = json.load(open("sweep.json"))
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="Tuning_mixed_data")
    print(sweep_id)
    wandb.agent(sweep_id='p980oobn', function=run, count=100)
    wandb.init()