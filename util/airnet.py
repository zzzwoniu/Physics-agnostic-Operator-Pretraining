import os
from functools import partial
import torch
from torch.utils import data
from util.misc import MultipleTensors
import numpy as np


class AirShape(data.Dataset):
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
    


class AirField(data.Dataset):
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
            
            if self.split == 'train':
                path = os.path.join(dataset_folder, type, 'train_')
            else:
                path = os.path.join(dataset_folder, type, 'val_')
            self.pc_files.append(partial(open_pc_file, path))
            self.query_files.append(partial(open_query_file, path))
            
        # Compute means and stds for each type of geometry, over the first 100 random parameterizations
        pmeans = []
        pstds = []
        us = []
        
        for j in range(len(geom_types)):
            type = geom_types[j]
            path = os.path.join(dataset_folder, type, 'train_')
            for i in range(100):
                if self.mesh:
                    data = np.load(f'{path}/body_{i}.npz')
                else:
                    data = np.load(f'{path}/field_{i}.npz')
                Pressure = data['data'][:,-1:]
                u = data['feature']
                pmeans.append(Pressure.mean())
                pstds.append(Pressure.std())
                us.append(u)
                    
        pmean = np.mean(pmeans)
        pstd = np.sqrt(np.mean(np.array(pstds)**2 + np.array(pmeans)**2) - pmean**2)
        
        us = np.stack(us)
        umean = np.mean(us, axis=0)
        ustd = np.std(us, axis=0)
        
        self.normalizer = (torch.tensor(pmean), torch.tensor(pstd), torch.tensor(umean), torch.tensor(ustd))

        
    def __getitem__(self, idx):
        
        type_idx = int(np.searchsorted(self.cumulative_sizes, idx + 1))
        
        if type_idx == 0:
            data_idx = idx
        else:
            data_idx = idx - self.cumulative_sizes[type_idx - 1]

        # For queries, if pc_files (x, y, sdf, nx, ny), if query_files (x, y, sdf)
        body = self.pc_files[type_idx](data_idx)['data'][:, :2]
        
        u = self.pc_files[type_idx](data_idx)['feature']
        if self.mesh:
            queries = self.pc_files[type_idx](data_idx)['data'][:, :5]
            Pfield = self.pc_files[type_idx](data_idx)['data'][:, -1:]
        else:
            queries = self.query_files[type_idx](data_idx)['data'][:, :3]
            Pfield = self.query_files[type_idx](data_idx)['data'][:, -1:]

        # Downsampling
        if self.sampling:
            ind = np.random.default_rng().choice(body.shape[0], self.num_points, replace=False)
            body = body[ind]                              # Downsample to (num_points, 2)
        body = torch.from_numpy(body)
        u = torch.from_numpy(u)
        
        if self.sampling:
            ind = np.random.default_rng().choice(queries.shape[0], self.num_queries, replace=False)
            queries = queries[ind]                        # Downsample to (num_queries, 3/5)
            Pfield = Pfield[ind]
        queries = torch.from_numpy(queries)
        Pfield = torch.from_numpy(Pfield)
        
        # Mask
        mask = torch.ones_like(Pfield)
        mask[queries[:,2:3] < -0.01] = 0
        queries = torch.cat((queries[:,:2], mask), dim=-1)
        
        if self.normalize:
            pmean, pstd, umean, ustd = self.normalizer
            Pfield = (Pfield - pmean)/pstd
            u = (u - umean)/ustd
        
        # Features include core point clouds and winding point clouds 
        features = MultipleTensors([body])
        queries = torch.cat([queries[:,:3], u.expand(queries.shape[0], -1)], dim=-1)
        
        return queries, Pfield, features, mask

        
    def __len__(self):
        return self.cumulative_sizes[-1]