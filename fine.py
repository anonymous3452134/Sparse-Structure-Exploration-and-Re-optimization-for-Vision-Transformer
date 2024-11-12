import os
import time
import torch
import datetime
from pathlib import Path
from torch.optim import AdamW
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from transformers import AutoFeatureExtractor, ViTForImageClassification
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import NativeScaler, get_state_dict, ModelEma

from utils import *
from pruning import *
from config import get_train_args
from engine import evaluate, fine_train_one_epoch
from dataset.samplers import RASampler
from dataset.datasets import build_dataset
from dataset.augment import new_data_aug_generator


def replace_all_linear_layers(module):
    for name, sub_module in module.named_children():
        if isinstance(sub_module, nn.Linear):
            new_module = MaskLinear(sub_module.in_features, sub_module.out_features)
            with torch.no_grad():
                new_module.weight.copy_(sub_module.weight)
                if sub_module.bias is not None:
                    new_module.bias.copy_(sub_module.bias)
            setattr(module, name, new_module)
        else:
            replace_all_linear_layers(sub_module)


def set_dropout_to_zero(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Dropout):
            setattr(module, name, nn.Dropout(p=0.0))
        else:
            set_dropout_to_zero(child)


def hyperparam():
    args = config.config()
    return args


class DeiTWrapper(nn.Module):
    def __init__(self, model):
        super(DeiTWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def set_all_type_values(self, type_value):
        for module in self.model.modules():
            if isinstance(module, MaskLinear):
                module.set_type_value(type_value)


def reset_momentum(optimizer):
    for group in optimizer.param_groups:
        for p in group["params"]:
            param_state = optimizer.state[p]
            if "momentum_buffer" in param_state:
                del param_state["momentum_buffer"]


def remove_module_prefix(state_dict):
    return {k.replace("module.model.", ""): v for k, v in state_dict.items()}


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cu_num
    init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    # utils.set_seed(args)
    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    # Load pretrained DeiT model
    if args.model_size == "tiny":
        model_name = "facebook/deit-tiny-patch16-224"
        m_name = "facebook/deit-tiny-distilled-patch16-224"
    elif args.model_size == "small":
        model_name = "facebook/deit-small-patch16-224"
        m_name = "facebook/deit-small-distilled-patch16-224"
    elif args.model_size == "base":
        model_name = "facebook/deit-base-patch16-224"
        m_name = "facebook/deit-base-distilled-patch16-224"

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)

    replace_all_linear_layers(model)
    set_dropout_to_zero(model)

    state_dict = torch.load(
        "./output/pretrain50_static0.0001tiny0.4_new_grad_L18_imagnet.pth"
    )

    state_dict = remove_module_prefix(state_dict["model"])
    model.load_state_dict(state_dict)

    model = DeiTWrapper(model)
    model.to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model.module.set_all_type_values(0)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )

    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * get_world_size() / 512.0
        args.lr = linear_scaled_lr

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_scaler = NativeScaler()

    lr_scheduler = scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    # lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=0)

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = fine_train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            model_ema,
            mixup_fn,
            set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
            args=args,
        )

        lr_scheduler.step(epoch)

        print(f"Current learning rate: {lr_scheduler.get_last_lr()[0]}")

        if args.output_dir:
            checkpoint_name = f"pre50_fine{str(args.epochs)}_DSP_static{str(args.lr)}{args.model_size}{str(args.prune_rate)}_{args.method}_{args.mag_type}_{args.prune_imp}{args.prune_freq}_imagnet.pth"
            checkpoint_paths = [output_dir / checkpoint_name]
            for checkpoint_path in checkpoint_paths:
                save_on_master(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "model_ema": get_state_dict(model_ema),
                        "scaler": loss_scaler.state_dict(),
                        "args": args,
                    },
                    checkpoint_path,
                )

        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )

        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if args.output_dir:
                checkpoint_name = f"Best_pre50_fine{str(args.epochs)}_DSP_static{str(args.lr)}{args.model_size}{str(args.prune_rate)}_{args.method}_{args.mag_type}_{args.prune_imp}{args.prune_freq}_imagnet.pth"
                checkpoint_paths = [output_dir / checkpoint_name]
                for checkpoint_path in checkpoint_paths:
                    save_on_master(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "model_ema": get_state_dict(model_ema),
                            "scaler": loss_scaler.state_dict(),
                            "args": args,
                        },
                        checkpoint_path,
                    )

        print(f"Max accuracy: {max_accuracy:.2f}%")

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
        }

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_train_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cu_num
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
