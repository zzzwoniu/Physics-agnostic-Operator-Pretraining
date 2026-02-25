import os
from functools import partial
import torch
from torch.utils import data
from util.misc import MultipleTensors
import numpy as np



class ElecShape(data.Dataset):
    def __init__(self, dataset_folder, split, transform, geom_types, num_geom, num_points, num_queries, sampling = True):
        
        self.dataset_folder = dataset_folder
        self.split = split               # train or validation
        self.geom_types = geom_types     # types of geometries
        self.num_geom = num_geom         # number of geometries for each type
        self.cumulative_sizes = np.cumsum(num_geom)
        self.num_points = num_points     # Pad or downsample point clouds
        self.num_queries = num_queries   # Downsample number of query points
        self.transform = transform
        self.sampling = sampling
        
        self.pc_files = []
        self.occ_files = []
        self.pert_occ_files = []
        for type in geom_types:
            def open_pc_file(path, idx):
                return np.load(f'{path}/body_{idx}.npz')
                
            def open_occ_file(path, idx):
                return np.load(f'{path}/occ_{idx}.npz')
            
            def open_pert_occ_file(path, idx):
                return np.load(f'{path}/pert_occ_{idx}.npz')
            
            path = os.path.join(dataset_folder, type, self.split)
            self.pc_files.append(partial(open_pc_file, path))
            self.occ_files.append(partial(open_occ_file, path))
            self.pert_occ_files.append(partial(open_pert_occ_file, path))
        
    def __getitem__(self, idx):
        
        type_idx = int(np.searchsorted(self.cumulative_sizes, idx + 1))
        
        if type_idx == 0:
            data_idx = idx
        else:
            data_idx = idx - self.cumulative_sizes[type_idx - 1]

        # Input point cloud to the encoder
        body = self.pc_files[type_idx](data_idx)['data'][:, :2]
        mask = self.pc_files[type_idx](data_idx)['data'][:, 3]
        body = body[mask > 0.5, :]

        if self.sampling:
            ind = np.random.default_rng().choice(body.shape[0], self.num_points, replace=False)
            body = body[ind]                              # Downsample to (num_points, 2)
        body = torch.from_numpy(body)
        
        # Query coordinates and label for the decoder, following occupancy paradigm
        vol_queries = self.occ_files[type_idx](data_idx)['data'][:,:2]   # (Nquery, 2)
        vol_labels = self.occ_files[type_idx](data_idx)['data'][:,2]    # (Nquery, )
        near_queries = self.pert_occ_files[type_idx](data_idx)['data'][:,:2] # (Nquery, 2)
        near_labels = self.pert_occ_files[type_idx](data_idx)['data'][:,2]  # (Nquery, )
        
        # Filter out unlabeled points with label -1
        vol_queries = vol_queries[vol_labels >= 0, :]
        vol_labels = vol_labels[vol_labels >= 0]
        near_queries = near_queries[near_labels >= 0, :]
        near_labels = near_labels[near_labels >= 0]
        
        if self.sampling:
            ind = np.random.default_rng().choice(vol_queries.shape[0], self.num_queries, replace=False)
            vol_queries = vol_queries[ind]
            vol_labels = vol_labels[ind]
            
            ind = np.random.default_rng().choice(near_queries.shape[0], self.num_queries, replace=False)
            near_queries = near_queries[ind]
            near_labels = near_labels[ind]
            
        vol_queries = torch.from_numpy(vol_queries)
        vol_labels = torch.from_numpy(vol_labels)
        cls = np.zeros(len(self.geom_types), dtype=int)
        cls[type_idx] = 1

        if self.split == 'train':
            near_queries = torch.from_numpy(near_queries)
            near_labels = torch.from_numpy(near_labels).float()

            queries = torch.cat([vol_queries, near_queries], dim=0)
            labels = torch.cat([vol_labels, near_labels], dim=0)
        else:
            queries = vol_queries
            labels = vol_labels
        
        return queries, labels, body, cls
        
    def __len__(self):
        return self.cumulative_sizes[-1]
    
    


class ElecField(data.Dataset):
    def __init__(self, dataset_folder, split, transform, geom_types, num_geom, num_points, num_queries, sampling = True, normalize = True, mesh = True):
        
        self.dataset_folder = dataset_folder
        self.split = split               # train or validation
        self.geom_types = geom_types     # types of geometries
        self.num_geom = num_geom         # number of geometries for each type
        self.cumulative_sizes = np.cumsum(num_geom)
        self.num_points = num_points     # Pad or downsample point clouds
        self.num_queries = num_queries   # Downsample number of query points
        self.transform = transform
        self.sampling = sampling
        self.normalize = normalize
        self.mesh = mesh
        
        self.pc_files = []
        self.query_files = []
        self.occ_files = []
        for type in geom_types:
            def open_pc_file(path, idx):
                return np.load(f'{path}/body_{idx}.npz')
                
            def open_query_file(path, idx):
                return np.load(f'{path}/field_{idx}.npz')
            
            def open_occ_file(path, idx):
                return np.load(f'{path}/occ_{idx}.npz')
            
            path = os.path.join(dataset_folder, type, self.split)
            self.pc_files.append(partial(open_pc_file, path))
            self.query_files.append(partial(open_query_file, path))
            self.occ_files.append(partial(open_occ_file, path))
            
        # Compute means and stds for each type of geometry, over the first 100 random parameterizations
        means = []
        stds = []
        for j in range(len(geom_types)):
            type = geom_types[j]
            path = os.path.join(dataset_folder, type, 'train')
            for i in range(100):
                if self.mesh:
                    data = np.load(f'{path}/body_{i}.npz')['data']
                else:
                    data = np.load(f'{path}/field_{i}.npz')['data']
                Phi = data[:,2:3]
                means.append(Phi.mean())
                stds.append(Phi.std())
                    
        mean = np.mean(means)
        std = np.sqrt(np.mean(np.array(stds)**2 + np.array(means)**2) - mean**2)
        self.normalizer = (torch.tensor(mean), torch.tensor(std))
        
    def __getitem__(self, idx):
        
        type_idx = int(np.searchsorted(self.cumulative_sizes, idx + 1))
        
        if type_idx == 0:
            data_idx = idx
        else:
            data_idx = idx - self.cumulative_sizes[type_idx - 1]

        # Retrieve preloaded data
        body = self.pc_files[type_idx](data_idx)['data'][:, :2]
        mask = self.pc_files[type_idx](data_idx)['data'][:, 3:]
        body = body[mask[:,0] > 0.5,:]
        if self.mesh:
            queries = self.pc_files[type_idx](data_idx)['data'][:, :2]
            Efield = self.pc_files[type_idx](data_idx)['data'][:, 2:3]
        else:
            queries = self.query_files[type_idx](data_idx)['data'][:, :2]
            Efield = self.query_files[type_idx](data_idx)['data'][:, 2:3]
            mask = self.occ_files[type_idx](data_idx)['data'][:,2:3]

        # Downsampling
        if self.sampling:
            ind = np.random.default_rng().choice(body.shape[0], self.num_points, replace=False)
            body = body[ind]                              # Downsample to (num_points, 2)
        body = torch.from_numpy(body)
        
        if self.sampling:
            ind = np.random.default_rng().choice(queries.shape[0], self.num_queries, replace=False)
            queries = queries[ind]                              # Downsample to (num_queries, 2)
            Efield = Efield[ind]
            mask = mask[ind]
            
        queries = torch.from_numpy(queries)
        Efield = torch.from_numpy(Efield)
        mask = torch.from_numpy(mask) 

        queries = torch.cat((queries, mask), dim=-1)
        
        if self.normalize:
            mean, std = self.normalizer
            Efield = (Efield - mean)/std
        else:
            Efield = Efield
        
        # Features include core point clouds and winding point clouds   
        features = MultipleTensors([body])
        
        return queries, Efield, features, torch.ones_like(mask) # material domain is the same as computation domain for this dataset

        
    def __len__(self):
        return self.cumulative_sizes[-1]