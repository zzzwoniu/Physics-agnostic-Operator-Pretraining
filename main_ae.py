import argparse
import datetime
import json
import numpy as np
import os
import time
import yaml
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
torch.set_num_threads(8)
import util.misc as misc
from util.datasets import build_magcore_shape_dataset, build_stress_shape_dataset, build_airfran_shape_dataset, build_elec_shape_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from models import BulkVAE
from engine_ae import train_one_epoch, evaluate

def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_args_parser():
    parser = argparse.ArgumentParser('Autoencoder', add_help=False)
    parser.add_argument('--batch_size', default=100, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--num_inputs', default=2048, type=int,
                        help='input size')
    parser.add_argument('--num_latents', default=256, type=int,
                        help='latent size')
    parser.add_argument('--depth', default=6, type=int,
                        help='number of attention layers on latent space')
    parser.add_argument('--dim', default=256, type=int,
                        help='embedding dim')
    parser.add_argument('--latent_dim', default=32, type=int,
                        help='latent dim')
    parser.add_argument('--linear', default=False, type=bool,
                        help='linear or quadratic attention')
    parser.add_argument('--drop_path_rate', default=0.05, type=float,
                        help='drop path rate')
    parser.add_argument('--kl_weight', default=0.001, type=float,
                        help='kl weight')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')


    # Dataset parameters
    parser.add_argument('--dataset', default='Stress', type=str,
                        help='dataset name')
    parser.add_argument('--data_path', default='your_data_path', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='your_output_dir', type=str,
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='your_log_dir', type=str,
                        help='path where to tensorboard log')
    parser.add_argument('--geom_types', type=str, nargs='+', default=['elec_0_3_0_7_0_2'],
                        help='geometry types to be trained')
    parser.add_argument('--num_geom', type=int, nargs='+', default=[3800],
                        help='number of each geometry to be trained')
    parser.add_argument('--val_num_geom', type=int, nargs='+', default=[200],
                        help='number of each geometry to be trained')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='number of input points')
    parser.add_argument('--num_queries', type=int, default=1024,
                        help='number of query points')
    parser.add_argument('--transform', default=False, type=bool,
                        help='randomly scale the bulk points or not')
    parser.add_argument('--comp', default='core', type=str,
                        help='used when multiple components are involved')
    
    # Training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--weight_dtype', default="fp32", type=str,
                        help='Training data weight type')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=False)

    # distributed training parameters
    parser.add_argument('--world_size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if args.dataset == 'Stress':
        dataset_train = build_stress_shape_dataset('train', args=args)
        dataset_val = build_stress_shape_dataset('val', args=args)
    elif args.dataset == 'AirfRans':
        dataset_train = build_airfran_shape_dataset('train', args=args)
        dataset_val = build_airfran_shape_dataset('val', args=args)
    elif args.dataset == 'Inductor':
        dataset_train = build_magcore_shape_dataset('train', args=args)
        dataset_val = build_magcore_shape_dataset('val', args=args)
    elif args.dataset == 'Elec':
        dataset_train = build_elec_shape_dataset('train', args=args)
        dataset_val = build_elec_shape_dataset('val', args=args)
    else:
        raise NotImplementedError("This dataset is not yet implemented.")

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        prefetch_factor=2,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        # batch_size=args.batch_size,
        batch_size=1,
        # num_workers=args.num_workers,
        num_workers=1,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    model = BulkVAE.create_autoencoder(args.depth, args.dim, M=args.num_latents, latent_dim=args.latent_dim, N=args.num_inputs, linear=args.linear, drop_path_rate=args.drop_path_rate, deterministic=False, dataset=args.dataset)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if True: #args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    opt_vae = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr)
    loss_scaler = NativeScaler()

    criterion = torch.nn.BCEWithLogitsLoss()

    print("criterion = %s" % str(criterion))

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_iou = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if True: # args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            opt_vae, device, epoch, loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 400 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=opt_vae,
                loss_scaler=loss_scaler, epoch=epoch)

        if epoch % 10 == 0 or epoch + 1 == args.epochs:
            test_stats = evaluate(data_loader_val, model, device, args)
            print(f"iou of the network on the {len(dataset_val)} test images: {test_stats['iou']:.3f}")
            max_iou = max(max_iou, test_stats["iou"])
            print(f'Max iou: {(100*max_iou):.2f}%')

            if log_writer is not None:
                log_writer.add_scalar('perf/test_iou', test_stats['iou'], epoch)
                log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    
    # First parser to get the config path
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, help='Path to config YAML file')
    config_args, remaining_argv = config_parser.parse_known_args()

    # Load default config if specified
    default_args = {}
    if config_args.config:
        default_args = load_yaml_config(config_args.config)

    # Now build full parser and apply defaults from config
    parser = get_args_parser()
    parser.set_defaults(**default_args)
    args = parser.parse_args(remaining_argv)
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)