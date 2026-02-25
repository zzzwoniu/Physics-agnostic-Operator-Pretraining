import math
import sys
from typing import Iterable
import torch
import numpy as np
import util.misc as misc
import util.lr_sched as lr_sched



def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, opt_vae: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None):
    
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    
    if args.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif args.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    elif args.weight_dtype == "fp32":
        weight_dtype = torch.float32

    accum_iter = args.accum_iter

    opt_vae.zero_grad()

    kl_weight = args.kl_weight

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (points, labels, body, cls) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(opt_vae, data_iter_step / len(data_loader) + epoch, args)

        points = points.to(device, non_blocking=True, dtype=weight_dtype)
        labels = labels.to(device, non_blocking=True, dtype=weight_dtype)
        body = body.to(device, non_blocking=True, dtype=weight_dtype)
        cls = cls.to(device, non_blocking=True, dtype=weight_dtype)
            
        # Train VAE
        with torch.amp.autocast('cuda', enabled=False):
            outputs = model(points, body)
            if 'kl' in outputs:
                loss_kl = outputs['kl']
                loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            else:
                loss_kl = None

            outputs = outputs['logits']
            loss_vol = criterion(outputs[:, :args.num_queries], labels[:, :args.num_queries])
            loss_near = criterion(outputs[:, args.num_queries:], labels[:, args.num_queries:])
            
            if loss_kl is not None:
                loss = loss_vol + 0.1 * loss_near + kl_weight * loss_kl
            else:
                loss = loss_vol + 0.1 * loss_near

        loss_value = loss.item()

        # Compute iou of occupancy
        threshold = 0
        pred = torch.zeros_like(outputs[:, :args.num_queries])
        pred[outputs[:, :args.num_queries]>=threshold] = 1

        accuracy = (pred==labels[:, :args.num_queries]).float().sum(dim=1) / labels[:, :args.num_queries].shape[1]
        accuracy = accuracy.mean()
        intersection = (pred * labels[:, :args.num_queries]).sum(dim=1)
        union = (pred + labels[:, :args.num_queries]).gt(0).sum(dim=1) + 1e-5
        iou = intersection * 1.0 / union
        iou = iou.mean()
        

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, opt_vae, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            opt_vae.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        if loss_kl is not None:
            metric_logger.update(loss_kl=loss_kl.item())

        metric_logger.update(iou=iou.item())

        min_lr = 10.
        max_lr = 0.
        for group in opt_vae.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    
    criterion = torch.nn.BCEWithLogitsLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    if args.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif args.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    elif args.weight_dtype == "fp32":
        weight_dtype = torch.float32

    # switch to evaluation mode
    model.eval()

    for points, labels, body, cls in metric_logger.log_every(data_loader, 50, header):

        points = points.to(device, non_blocking=True, dtype=weight_dtype)
        labels = labels.to(device, non_blocking=True, dtype=weight_dtype)
        body = body.to(device, non_blocking=True, dtype=weight_dtype)
        cls = cls.to(device, non_blocking=True, dtype=weight_dtype)

        # compute output
        with torch.amp.autocast('cuda',enabled=False):

            outputs = model(points, body)
            if 'kl' in outputs:
                loss_kl = outputs['kl']
                loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            else:
                loss_kl = None

            outputs = outputs['logits']
            loss = criterion(outputs, labels)

        threshold = 0
        
        # IOU
        pred = torch.zeros_like(outputs)
        pred[outputs>=threshold] = 1

        accuracy = (pred==labels).float().sum(dim=1) / labels.shape[1]
        accuracy = accuracy.mean()
        intersection = (pred * labels).sum(dim=1)
        union = (pred + labels).gt(0).sum(dim=1)
        iou = intersection * 1.0 / union + 1e-5
        iou = iou.mean()

        batch_size = points.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['iou'].update(iou.item(), n=batch_size)

        if loss_kl is not None:
            metric_logger.update(loss_kl=loss_kl.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* iou {iou.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(iou=metric_logger.iou, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}