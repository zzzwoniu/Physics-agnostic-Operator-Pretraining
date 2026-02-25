import os
from functools import partial
from scipy.io import loadmat
import torch
from torch.utils import data
from util.misc import MultipleTensors
import numpy as np


class MagcoreShape(data.Dataset):
    def __init__(self, dataset_folder, split, transform, geom_types, num_geom, num_points, num_queries, sampling = True, comp = 'core'):
        
        self.dataset_folder = dataset_folder
        self.split = split               # train or validation
        self.geom_types = geom_types     # types of magcores
        self.num_geom = num_geom         # number of geometries for each type of magcore
        self.cumulative_sizes = np.cumsum(num_geom)
        self.num_points = num_points     # Pad or downsample point clouds
        self.num_queries = num_queries   # Downsample number of query points
        self.transform = transform
        self.sampling = sampling
        self.comp = comp                 # Just core is enough as shape of winding depends on the shape of core
        
        self.pc_files = []
        self.query_files = []
        for type in geom_types:
            def open_pc_file(path, idx):
                return loadmat(f'{path}/Bfield_{self.comp}/data_{idx}.mat')
                
            def open_query_file(path, idx):
                return loadmat(f'{path}/Occupancy_{self.comp}/data_{idx}.mat')
            
            path = os.path.join(dataset_folder, type, self.split)
            self.pc_files.append(partial(open_pc_file, path))
            self.query_files.append(partial(open_query_file, path))
        
    def __getitem__(self, idx):
        
        type_idx = int(np.searchsorted(self.cumulative_sizes, idx + 1))
        
        if type_idx == 0:
            data_idx = idx
        else:
            data_idx = idx - self.cumulative_sizes[type_idx - 1]

        # Input point cloud to the encoder
        body = self.pc_files[type_idx](data_idx)
        body = np.concatenate((body['x'],body['y'],body['z']), axis=1)
        if self.sampling:
            ind = np.random.default_rng().choice(body.shape[0], self.num_points, replace=False)
            body = body[ind]                              # Downsample to (num_points, 3)
        body = torch.from_numpy(body)
        
        # Query coordinates and label for the decoder, following occupancy paradigm
        vol_queries = self.query_files[type_idx](data_idx)[f'rand_occ_{self.comp}'][:,:3]   # (Nquery, 3)
        vol_labels = self.query_files[type_idx](data_idx)[f'rand_occ_{self.comp}'][:,3]    # (Nquery, )
        near_queries = self.query_files[type_idx](data_idx)[f'nodes_occ_{self.comp}'][:,:3] # (Nquery, 3)
        near_labels = self.query_files[type_idx](data_idx)[f'nodes_occ_{self.comp}'][:,3]  # (Nquery, )
        
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

        if self.transform:
            body, queries = self.transform(body, queries)
        else:
            factor = torch.tensor([[35.,35.,35.]], dtype=body.dtype)
            body = body/factor
            queries = queries/factor
        
        return queries, labels, body, cls
        
    def __len__(self):
        return self.cumulative_sizes[-1]
    
    

class MagcoreField(data.Dataset):
    def __init__(self, dataset_folder, split, transform, geom_types, num_geom, num_points, num_queries, sampling = True, normalize = True, mesh = True):
        
        self.dataset_folder = dataset_folder
        self.split = split               # train or validation
        self.geom_types = geom_types     # types of magcores
        self.num_geom = num_geom         # number of geometries for each type of magcore
        self.cumulative_sizes = np.cumsum(num_geom)
        self.num_points = num_points     # Pad or downsample point clouds
        self.num_queries = num_queries   # Downsample number of query points
        self.transform = transform
        self.sampling = sampling
        self.normalize = normalize
        self.mesh = mesh
        
        self.core_pc_files = []
        self.query_files = []
        for type in geom_types:
            def open_core_pc_file(path, idx):
                return loadmat(f'{path}/Bfield_core/data_{idx}.mat')
                
            def open_query_file(path, idx):
                return loadmat(f'{path}/Bfield_rand/data_{idx}.mat')
            
            path = os.path.join(dataset_folder, type, self.split)
            self.core_pc_files.append(partial(open_core_pc_file, path))
            self.query_files.append(partial(open_query_file, path))
        
        # Compute means and stds for each type of geometry, over the first 100 random parameterizations
        means = []
        stds = []
        for j in range(len(geom_types)):
            type = geom_types[j]
            path = os.path.join(dataset_folder, type, 'train')
            for i in range(100):
                data = loadmat(f'{path}/Bfield_core/data_{i}.mat')
                B = np.concatenate((data['Bx'],data['By'],data['Bz']), axis=1)
                B = np.linalg.norm(B, axis=1)
                means.append(B.mean())
                stds.append(B.std())
              
        mean = np.mean(means)
        std = np.sqrt(np.mean(np.array(stds)**2 + np.array(means)**2) - mean**2)
        self.normalizer = (torch.tensor(mean), torch.tensor(std))

        
    def __getitem__(self, idx):
        
        type_idx = int(np.searchsorted(self.cumulative_sizes, idx + 1))
        
        if type_idx == 0:
            data_idx = idx
        else:
            data_idx = idx - self.cumulative_sizes[type_idx - 1]

        # Input point cloud to the encoder
        core = self.core_pc_files[type_idx](data_idx)
        core_body = np.concatenate((core['x'],core['y'],core['z']), axis=1)
        
        if self.mesh:
            queries = np.concatenate((core['x'],core['y'],core['z']), axis=1)
            Bfield = np.concatenate((core['Bx'],core['By'],core['Bz']), axis=1)
        else:
            field = self.query_files[type_idx](data_idx)['Bfield_rand_coor']
            queries = field[:,:3]
            Bfield = field[:,3:]
        
        # Downsampling
        if self.sampling:
            ind = np.random.default_rng().choice(core_body.shape[0], self.num_points, replace=False)
            core_body = core_body[ind]                          # Downsample to (num_points, 3)
        core_body = torch.from_numpy(core_body)
        
        if self.sampling:
            ind = np.random.default_rng().choice(queries.shape[0], self.num_queries, replace=False)
            queries = queries[ind]                              # Downsample to (num_queries, 3)
            Bfield = Bfield[ind]
        queries = torch.from_numpy(queries)
        Bfield = torch.from_numpy(Bfield)
        
        if self.transform:
            queries, queries = self.transform(queries, queries)
        else:
            factor = torch.tensor([[35.,35.,35.]], dtype=queries.dtype) # Normalize the computation domain
            core_body[:,:3] = core_body[:,:3]/factor
            queries = queries/factor
            
        # Mask
        mask = torch.ones_like(Bfield[:,-1:])
        if not self.mesh:
            mask[np.abs(Bfield[:,-1:] - 1) > 0.1] = 0
        queries = torch.cat((queries, mask), dim=-1)
            
        if self.normalize:
            mean, std = self.normalizer
            Bfield = (Bfield[:,:3].norm(dim=1, keepdim=True) - mean)/std
        else:
            Bfield = Bfield[:,:3].norm(dim=1, keepdim=True)
        
        features = MultipleTensors([core_body])
        
        return queries, Bfield, features, mask
    
    def __len__(self):
        return self.cumulative_sizes[-1]