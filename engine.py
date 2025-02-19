"""
Train and eval functions used in main.py
"""

import math
import sys
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import pruning
from pruning import *
from utils import *


def fine_train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.CrossEntropyLoss,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    set_training_mode=True,
    args=None,
):

    model.train(set_training_mode)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.cosub:
            samples = torch.cat((samples, samples), dim=0)

        with torch.cuda.amp.autocast():
            outputs = model(samples).logits
            if not args.cosub:
                loss = criterion(outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0] // 2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets)
                loss = loss + 0.25 * criterion(outputs[1], targets)
                loss = loss + 0.25 * criterion(
                    outputs[0], outputs[1].detach().sigmoid()
                )
                loss = loss + 0.25 * criterion(
                    outputs[1], outputs[0].detach().sigmoid()
                )

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.CrossEntropyLoss,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    set_training_mode=True,
    args=None,
    iteration=int,
):

    model.train(set_training_mode)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    target_sparsity = 0

    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.cosub:
            samples = torch.cat((samples, samples), dim=0)

        if (epoch + 1) < args.target_epoch:
            target_sparsity = (
                args.prune_rate
                - args.prune_rate
                * (1 - iteration / (args.target_epoch * len(data_loader))) ** 3
            )
        else:
            target_sparsity = args.prune_rate

        if epoch == args.target_epoch:
            model.module.set_all_type_values(0)

        if epoch >= args.warmup and epoch < args.target_epoch:
            if i % args.prune_freq == 0:
                if args.method == "ours":
                    masks = pruning.get_vit_masks(
                        model, target_sparsity, args.prune_imp, args.mag_type
                    )
                    pruning.apply_vit_masks(model, masks)
                elif args.method == "original":
                    masks = pruning.original_get_vit_masks(
                        model, target_sparsity, args.prune_imp, args.mag_type
                    )
                    pruning.apply_vit_masks(model, masks)
                elif args.method == "new":
                    masks = pruning.new_get_vit_masks(
                        model, target_sparsity, args.prune_imp, args.mag_type
                    )
                    pruning.apply_vit_masks(model, masks)
            iteration += 1

        with torch.cuda.amp.autocast():
            outputs = model(samples).logits
            if not args.cosub:
                loss = criterion(outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0] // 2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets)
                loss = loss + 0.25 * criterion(outputs[1], targets)
                loss = loss + 0.25 * criterion(
                    outputs[0], outputs[1].detach().sigmoid()
                )
                loss = loss + 0.25 * criterion(
                    outputs[1], outputs[0].detach().sigmoid()
                )

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return (
        {k: meter.global_avg for k, meter in metric_logger.meters.items()},
        target_sparsity,
        iteration,
    )


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images).logits
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
