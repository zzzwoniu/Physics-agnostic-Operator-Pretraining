import math
import sys
from typing import Iterable
import torch
import util.misc as misc
import util.lr_sched as lr_sched
from util.criterion import SimpleLpLoss
from util.misc import MultipleTensors


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None, VAEs=None):
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

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (queries, field, features, mask) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        queries = queries.to(device, non_blocking=True, dtype=weight_dtype)
        field = field.to(device, non_blocking=True, dtype=weight_dtype)
        features = features.to(device, non_blocking=True, dtype=weight_dtype)
        mask = mask.to(device, non_blocking=True, dtype=weight_dtype)

        with torch.amp.autocast('cuda',enabled=False):
            if VAEs is not None:
                with torch.no_grad():
                    features_ = MultipleTensors([VAEs[i].encode(features[i], return_kl=False) for i in range(len(args.branch_sizes))])
                outputs = model(queries, features_)
            else:
                outputs = model(queries, features)

            loss = criterion(outputs, field, mask=mask)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
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
def evaluate(data_loader, model, device, args, VAEs=None):
    
    criterion = SimpleLpLoss(size_average=True)

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

    for queries, field, features, mask in metric_logger.log_every(data_loader, 50, header):

        queries = queries.to(device, non_blocking=True, dtype=weight_dtype)
        field = field.to(device, non_blocking=True, dtype=weight_dtype)
        features = features.to(device, non_blocking=True, dtype=weight_dtype)
        mask = mask.to(device, non_blocking=True, dtype=weight_dtype)

        # compute output
        with torch.amp.autocast('cuda',enabled=False):

            if VAEs is not None:
                with torch.no_grad():
                    features_ = MultipleTensors([VAEs[i].encode(features[i], return_kl=False) for i in range(len(VAEs))])
                outputs = model(queries, features_)
            else:
                outputs = model(queries, features)

            loss = criterion(outputs, field, mask=mask)

        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}