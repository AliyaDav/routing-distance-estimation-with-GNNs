import numpy as np
import time
from tqdm import tqdm
import json

import torch
import torch.optim as optim
import wandb

from VRP_dataset import VRPDataset
from torch_geometric.loader import DataLoader

from nets.molecules_graph_regression.gated_gcn_net import GatedGCNNet

import warnings
warnings.filterwarnings("ignore", message="indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead.")


params = json.load(open('configs/molecules_graph_regression_GatedGCN_AQSOL_100k.json'))

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

torch.manual_seed(params['params']['seed'])

# wandb.init(project="experiments")

train_dataset = VRPDataset('vrp_data/train')
val_dataset = VRPDataset('vrp_data/validation')

# df = torch.load('/Users/aliyadavletshina/Desktop/thesis/routing-distance-estimation-with-GNNs/vrp_data/train/processed/data_25.pt')
# df.edge_attr
# len(os.listdir('/Users/aliyadavletshina/Desktop/thesis/routing-distance-estimation-with-GNNs/vrp_data/train/raw'))

train_dataloader = DataLoader(train_dataset, 
                                batch_size=params['params']['batch_size'],
                                num_workers=0,
                                shuffle=True, 
                                drop_last=True, 
                                generator=torch.Generator().manual_seed(1234)
                                )

valid_dataloader = DataLoader(val_dataset, 
                                batch_size=params['params']['batch_size'], 
                                num_workers=0,
                                shuffle=True, 
                                drop_last=True, 
                                generator=torch.Generator().manual_seed(1234)
                                )

# for i, batch in enumerate(train_dataloader):
#     print(batch)
#     if i == 2:
#         break

model=GatedGCNNet(params['net_params'])
model.parameters()
optimizer = optim.Adam([{'params': model.parameters(), 'lr': params['params']['init_lr']}])
lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.95 ** epoch)


# wandb.watch(model)

def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)

def calc_loss(graph_pred, targets):
    return torch.mean((torch.abs(graph_pred/targets - 1) * 100))
    
def validate(model, valid_dataloader):

    print(f'\nValidating...')
    model.eval()
    opt_gap = []

    for idx, bat in enumerate(valid_dataloader):
        with torch.no_grad():
            targets = bat.y.float()

            graph_pred = model(bat.x.float(), 
                    bat.edge_attr,
                    bat.edge_index, 
                    bat.batch)
            opt_gap_batch = calc_loss(graph_pred, targets)
            opt_gap.append(opt_gap_batch)

        wandb.log({f'val optimality gap': opt_gap_batch})

    return np.array(opt_gap).mean()



def train_epoch(epoch, model, train_dataloader, valid_dataloader, optimizer, lr_scheduler, device):
    
    print("\nStart train epoch {}".format(epoch))
    start_time = time.time()    
    step = epoch * (10000 // params['params']['batch_size'])

    model.train()
    epoch_loss = []

    for iter, batch in enumerate(train_dataloader):
        batch = batch.to(device)
        optimizer.zero_grad()
        print(batch.edge_attr[0][0])

        pred = model.forward(batch, batch.x, batch.edge_attr)
        loss = calc_loss(pred, batch.y.float())
        epoch_loss.append(loss)
        loss.backward()
        optimizer.step()
        
        step += 1

        if step % 100 == 0:
            wandb.log({'loss': loss, 'graph pred': pred, 'targets': batch.y.float() })

    lr_scheduler.step(epoch)
    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))
    
    epoch_avg_opt_gap = validate(model, valid_dataloader)
    wandb.log({'validation AOG': epoch_avg_opt_gap})
    wandb.log({'training AOG': np.array(epoch_loss).mean()})

# TODO: add edge features to the dataset


for epoch in range(params['params']['epochs']):
    start_training = time.time()   
    train_epoch(
        epoch, 
        model, 
        train_dataloader,
        valid_dataloader,
        optimizer, 
        lr_scheduler,
        device
    )

training_duration = time.time() - start_training
print(f'\nTraing of {params["params"]["epochs"]} epochs is finished in {training_duration} time.')
