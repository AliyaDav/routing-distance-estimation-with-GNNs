import time
import os

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

import wandb

from options import get_options
from VRP_dataset import VRPDataset
from models.GATNar import GATnar
from train import validate
from utils.functions import get_inner_model

import warnings
warnings.filterwarnings("ignore", message="indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead.")

if __name__ == "__main__":

    opts=get_options()

    val_dataset = VRPDataset('vrp_data_large/test')
    
    valid_dataloader = DataLoader(val_dataset, follow_batch = ['x', 'y'],
                                    batch_size=opts.batch_size, 
                                    num_workers=opts.num_workers,
                                    shuffle=True, 
                                    drop_last=True, 
                                    generator=torch.Generator().manual_seed(1234)
                                    )
    
    model=GATnar(opts)

    load_file = str(input('Set file name load the model from.\n'))
    load_path = os.path.join(os.getcwd(), 'models for inference', load_file)
    while not os.path.exists(load_path):
        print('File does not exist, try again\n')
        load_file = str(input('Set file name load the model from.\n'))
        load_path = os.path.join(os.getcwd(), 'models for inference', load_file)

    load_data = torch.load(load_path, map_location=lambda storage, loc: storage)
    model = get_inner_model(model)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})
    
    # Initialize optimizer
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': opts.lr_model}])    

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize Weights&Biases
    wandb.init(project="Evaluations")
    wandb.watch(model)

    start_time = time.time()

    val_loss, val_pred, val_rmse = validate(model, val_dataset, opts)

    eval_duration = time.time() - start_time
    print("Finished evaluation, took {} s".format(time.strftime('%H:%M:%S', time.gmtime(eval_duration))))

    wandb.log({f'Evaluation MAPE': val_loss,'Evaluation RMSE': val_rmse })
