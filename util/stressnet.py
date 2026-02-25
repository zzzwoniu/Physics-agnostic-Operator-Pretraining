
import os
from functools import partial
import torch
from torch.utils import data
from util.misc import MultipleTensors
import numpy as np


class StressShape(data.Dataset):
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
        
        if self.sampling:
            ind = np.random.default_rng().choice(body.shape[0], self.num_points, replace=False)
            body = body[ind]                              # Downsample to (num_points, 2)
        body = torch.from_numpy(body)
        
        # Query coordinates and label for the decoder, following occupancy paradigm
        vol_queries = self.occ_files[type_idx](data_idx)['data'][:,:2]   # (Nquery, 2)
        vol_labels = self.occ_files[type_idx](data_idx)['data'][:,2]    # (Nquery, )
        near_queries = self.pert_occ_files[type_idx](data_idx)['data'][:,:2] # (Nquery, 2)
        near_labels = self.pert_occ_files[type_idx](data_idx)['data'][:,2]  # (Nquery, )
        
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
    



class StressField(data.Dataset):
    def __init__(self, dataset_folder, split, transform, geom_types, num_geom, num_points, num_queries, sampling = True, normalize = True, mesh = True, model = 'GNOT', use_VAE = False):
        
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
        self.model = model
        self.use_VAE = use_VAE
        
        self.pc_files = []
        self.query_files = []
        for type in geom_types:
            def open_pc_file(path, idx):
                return np.load(f'{path}/body_{idx}.npz')
                
            def open_query_file(path, idx):
                return np.load(f'{path}/field_{idx}.npz')
            
            path = os.path.join(dataset_folder, type, self.split)
            self.pc_files.append(partial(open_pc_file, path))
            self.query_files.append(partial(open_query_file, path))
            
        # Compute means and stds for each type of geometry, over the first 100 random parameterizations
        means = []
        stds = []
        for j in range(len(geom_types)):
            type = geom_types[j]
            path = os.path.join(dataset_folder, type, 'train')
            for i in range(100):
                data = np.load(f'{path}/body_{i}.npz')['data']
                Stress = data[:,2:]
                means.append(Stress.mean())
                stds.append(Stress.std())
                    
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
        
        if self.mesh:
            queries = self.pc_files[type_idx](data_idx)['data'][:, :2]
            Sfield = self.pc_files[type_idx](data_idx)['data'][:, 2:]
        else:
            queries = self.query_files[type_idx](data_idx)['data'][:, :2]
            Sfield = self.query_files[type_idx](data_idx)['data'][:, 2:]

        # Downsampling
        if self.sampling:
            ind = np.random.default_rng().choice(body.shape[0], self.num_points, replace=False)
            body = body[ind]                                    # Downsample to (num_points, 2)
        body = torch.from_numpy(body)
        
        if self.sampling:
            ind = np.random.default_rng().choice(queries.shape[0], self.num_queries, replace=False)
            queries = queries[ind]                              # Downsample to (num_queries, 2)
            Sfield = Sfield[ind]
        queries = torch.from_numpy(queries)
        Sfield = torch.from_numpy(Sfield)
        
        # Mask
        mask = torch.ones_like(Sfield)
        mask[Sfield < 0] = 0
        queries = torch.cat((queries, mask), dim=-1)
        
        if self.normalize:
            mean, std = self.normalizer
            Sfield = (Sfield - mean)/std
        else:
            Sfield = Sfield
        
        # Features include core point clouds and winding point clouds   
        features = MultipleTensors([body])
        
        return queries, Sfield, features, mask

        
    def __len__(self):
        return self.cumulative_sizes[-1]
    