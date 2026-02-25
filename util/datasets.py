from .magcorenet import MagcoreShape, MagcoreField
from .stressnet import StressField, StressShape
from .airnet import AirShape, AirField
from .elecnet import ElecShape, ElecField

    
def build_magcore_shape_dataset(split, args):
    if split == 'train':
        return MagcoreShape(args.data_path, split=split, transform=None, geom_types=args.geom_types, num_geom=args.num_geom, num_points=args.num_points, num_queries=args.num_queries, sampling = True, comp = args.comp)
    elif split == 'val':
        return MagcoreShape(args.data_path, split=split, transform=None, geom_types=args.geom_types, num_geom=args.val_num_geom, num_points=args.num_points, num_queries=args.num_queries, sampling = True, comp = args.comp)


def build_Bfield_dataset(split, args, mesh = True):
    if split == 'train':
        return MagcoreField(args.data_path, split=split, transform=None, geom_types=args.geom_types, num_geom=args.num_geom, num_points=args.num_points, num_queries=args.num_queries, sampling = True, normalize = args.normalize, mesh = mesh)
    elif split == 'val':
        return MagcoreField(args.data_path, split=split, transform=None, geom_types=args.geom_types, num_geom=args.val_num_geom, num_points=args.num_points, num_queries=args.num_queries, sampling = True, normalize = args.normalize, mesh = mesh)


def build_stress_shape_dataset(split, args):
    if split == 'train':
        return StressShape(args.data_path, split=split, transform=None, geom_types=args.geom_types, num_geom=args.num_geom, num_points=args.num_points, num_queries=args.num_queries, sampling = True)
    elif split == 'val':
        return StressShape(args.data_path, split=split, transform=None, geom_types=args.geom_types, num_geom=args.val_num_geom, num_points=args.num_points, num_queries=args.num_queries, sampling = True)  


def build_Sfield_dataset(split, args, mesh = True):
    if split == 'train':
        return StressField(args.data_path, split=split, transform=None, geom_types=args.geom_types, num_geom=args.num_geom, num_points=args.num_points, num_queries=args.num_queries, sampling = True, normalize = args.normalize, mesh = mesh, model = args.model, use_VAE = args.use_VAE)
    elif split == 'val':
        return StressField(args.data_path, split=split, transform=None, geom_types=args.geom_types, num_geom=args.val_num_geom, num_points=args.num_points, num_queries=args.num_queries, sampling = True, normalize = args.normalize, mesh = mesh, model = args.model, use_VAE = args.use_VAE)    
    
    
def build_airfran_shape_dataset(split, args):
    if split == 'train':
        return AirShape(args.data_path, split=split, transform=None, geom_types=args.geom_types, num_geom=args.num_geom, num_points=args.num_points, num_queries=args.num_queries, sampling = True)
    elif split == 'val':
        return AirShape(args.data_path, split=split, transform=None, geom_types=args.geom_types, num_geom=args.val_num_geom, num_points=args.num_points, num_queries=args.num_queries, sampling = True)
    
    
def build_Pfield_dataset(split, args, mesh = True):
    if split == 'train':
        return AirField(args.data_path, split=split, transform=None, geom_types=args.geom_types, num_geom=args.num_geom, num_points=args.num_points, num_queries=args.num_queries, sampling = True, normalize = args.normalize, mesh = mesh, model = args.model, use_VAE = args.use_VAE)
    elif split == 'val':
        return AirField(args.data_path, split=split, transform=None, geom_types=args.geom_types, num_geom=args.val_num_geom, num_points=args.num_points, num_queries=args.num_queries, sampling = True, normalize = args.normalize, mesh = mesh, model = args.model, use_VAE = args.use_VAE)


def build_elec_shape_dataset(split, args):
    if split == 'train':
        return ElecShape(args.data_path, split=split, transform=None, geom_types=args.geom_types, num_geom=args.num_geom, num_points=args.num_points, num_queries=args.num_queries, sampling = True)
    elif split == 'val':
        return ElecShape(args.data_path, split=split, transform=None, geom_types=args.geom_types, num_geom=args.val_num_geom, num_points=args.num_points, num_queries=args.num_queries, sampling = True)
    
    
def build_Efield_dataset(split, args, mesh = True):
    if split == 'train':
        return ElecField(args.data_path, split=split, transform=None, geom_types=args.geom_types, num_geom=args.num_geom, num_points=args.num_points, num_queries=args.num_queries, sampling = True, normalize = args.normalize, mesh = mesh)
    elif split == 'val':
        return ElecField(args.data_path, split=split, transform=None, geom_types=args.geom_types, num_geom=args.val_num_geom, num_points=args.num_points, num_queries=args.num_queries, sampling = True, normalize = args.normalize, mesh = mesh)