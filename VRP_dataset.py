# from torch.utils.data import Dataset
from torch_geometric.data import Dataset, Data
import torch
import os
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

class VRPDataset(Dataset):
    """Class representing a PyTorch dataset of VRP instances, which is fed to a dataloader
    """
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None,
                 neighbors=0.2, knn_strat='percentage'
                 ):
        self.neighbors = neighbors
        self.knn_strat = knn_strat
        super(VRPDataset, self).__init__(root, transform, pre_transform, pre_filter)

        
    @property
    def raw_file_names(self):
        return [file for file in os.listdir(os.path.join(os.getcwd(),self.root, 'raw')) \
                    if file not in ['pre_filter.pt', 'pre_transform.pt', '.DS_Store']]

    @property
    def processed_file_names(self):
        return [file for file in os.listdir(os.path.join(os.getcwd(),self.root, 'processed')) \
                    if file not in ['pre_filter.pt', 'pre_transform.pt', '.DS_Store']]

    def download(self):
        pass
    
    def process(self):
        idx = 0
        for raw_path in self.raw_paths:

            data = torch.load(raw_path)

            node_features = data[:][0]
            targets = data[:][1]
            dist_matrix=data[:][2].squeeze()
            # adj_matrix = torch.ByteTensor(self._nearest_neighbor_graph(node_features[:,:,:2].squeeze(), \
            #                                 dist_matrix, self.neighbors, self.knn_strat))
            adj_matrix = np.ones((node_features.shape[0], node_features.shape[0]))
            edge_index = adj_matrix.nonzero().t().contiguous()
            edge_attr = self.get_edge_attr(dist_matrix, edge_index)
            # print('edge attributes', edge_attr)

            
            data = Data(x=node_features.squeeze(), 
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=targets
                        # adj_matrix=adj_matrix
                        ) 
            
            torch.save(data, 
                os.path.join(self.processed_dir, 
                            f'data_{idx}.pt'))
            
            idx += 1
                
                    
    def len(self):
        return len(self.processed_file_names)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(os.getcwd(), self.processed_dir, f'data_{idx}.pt'))

        return data

    def _nearest_neighbor_graph(self, nodes, dist_matrix, neighbors, knn_strat='percentage'):
            """Returns k-Nearest Neighbor graph as a **NEGATIVE** adjacency matrix
            """
            num_nodes = len(nodes)
            # If `neighbors` is a percentage, convert to int
            if knn_strat == 'percentage':
                neighbors = int(num_nodes * neighbors)
                print('num_neighbor', neighbors)
            
            
            if neighbors >= num_nodes-1 or neighbors == -1:
                W = np.ones((num_nodes, num_nodes))
            else:
                # Compute distance matrix
                W_val = squareform(pdist(nodes, metric='euclidean'))
                # W_val = dist_matrix
                W = np.zeros((num_nodes, num_nodes))
                
                # Determine k-nearest neighbors for each node
                knns = np.argpartition(W_val, kth=neighbors, axis=-1)[:, neighbors::-1]
                # Make connections
                for idx in range(num_nodes):
                    W[idx][knns[idx]] = 1
            
            # Remove self-connections
            np.fill_diagonal(W, 1)
            return W

    def get_edge_attr(self, dist_mat, edge_idx):
        

        edge_attr=[]
        for i in range(edge_idx.shape[1]):
            coord=tuple(edge_idx[:,i].tolist())
            edge_attr.append(dist_mat[coord].float())
        
        edge_attr = torch.stack(edge_attr)

        return edge_attr.reshape((edge_idx.shape[1],1))
    # torch.tensor(edge_attr, dtype=float).reshape((edge_idx.shape[1],1))
 