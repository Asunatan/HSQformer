import argparse
import logging
import math
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from timm.data import Mixup
from timm.utils import ModelEmaV3
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import AUROC, Accuracy, F1Score, MetricCollection, Precision, Recall, Specificity
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import (
    DenseNet201_Weights,
    ResNet101_Weights,
    ResNeXt101_32X8D_Weights,
    Swin_B_Weights,
    Swin_S_Weights,
    ViT_B_16_Weights,
    densenet201,
    resnet101,
    resnext101_32x8d,
    swin_b,
    swin_s,
    vit_b_16,
)
from dataset.data_loading import Liver_Dataset
from models.LivNet import LENet, LivNet_Baseline, LivNet_no_moe
from models import cswin
from models.convnext_segmentaion import convnext_base, convnext_small
from models.davit import DaViT
from models.focalnet import focalnet_base_lrf
from models.hiera_segmentation import Hiera
from models.mpvit import mpvit_base
from models.pvt_v2 import pvt_v2_b5
from models.swin_transformer_v2 import SwinTransformerV2
from models.thynet import ThyNet
from utils.loss import FocalLoss
from utils.model_stats import log_model_statistics
from utils.train_eval import CAM_visualization, evaluate, metric_info, train_one_epoch, visual_feature

LOGGER = logging.getLogger("HSQformer")


def model_log_name(args: argparse.Namespace) -> str:
    if args.model == "LENet" or args.model.startswith("LENet-"):
        return "-".join(
            [
                "LENet",
                args.serial_parallel,
                args.sparse_dense,
                f"{args.num_experts}experts",
                f"top_{args.top_k}",
                args.head_type,
                *(["cat_moe_head"] if args.cat_moe_head else []),
                "q_former_depths" + "_".join(str(v) for v in args.q_former_depths),
                "stage_dims" + "_".join(str(v) for v in args.stage_dims),
                "q_former_heads" + "_".join(str(v) for v in args.q_former_head_num),
                f"num_query_{args.num_query_tokens}",
                f"query_dim_{args.query_dim}",
            ]
        )
    return args.model


def configure_logger(log_dir: str, args: argparse.Namespace, fold_index: int) -> logging.Logger:
    logger = logging.getLogger("HSQformer")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{timestamp}_fold{fold_index}_{model_log_name(args)}_training.log"
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename), mode="a")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_model_name(args) -> None:
    if args.model == "LENet":
        args.model = "-".join(
            [
                args.model,
                args.serial_parallel,
                args.sparse_dense,
                f"{args.num_experts}experts",
                f"top_{args.top_k}",
                args.head_type,
                *(["cat_moe_head"] if args.cat_moe_head else []),
                "q_former_depths" + "_".join(str(v) for v in args.q_former_depths),
                f"num_query_{args.num_query_tokens}",
                f"query_dim_{args.query_dim}",
            ]
        )


def build_model(args):
    model_name = args.model

    if model_name == "HSQformer_Baseline":
        LOGGER.info("HSQformer_Baseline")
        return LivNet_Baseline()
    if model_name == "HSQformer_no_moe":
        LOGGER.info("HSQformer_no_moe")
        return LivNet_no_moe()
    if "LENet" in model_name:
        LOGGER.info("LENet")
        return LENet(args=args)
    if model_name == "DaViT":
        LOGGER.info("DaViT")
        model = DaViT(
            patch_size=4,
            window_size=7,
            embed_dims=(128, 256, 512, 1024),
            num_heads=(4, 8, 16, 32),
            depths=(1, 1, 9, 1),
            mlp_ratio=4.0,
            overlapped_patch=False,
        )
        checkpoint = torch.load("/home/uax/SCY/LiverClassification/Weights/DaViTmodel_best.pth.tar", map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        model.head = nn.Linear(in_features=model.head.in_features, out_features=args.num_classes)
        return model
    if model_name == "Mpvit_base":
        LOGGER.info("Mpvit_base")
        model = mpvit_base()
        weight = torch.load(r"/home/uax/SCY/LiverClassification/Weights/mpvit_base.pth", map_location="cpu")["model"]
        model.load_state_dict(weight)
        model.cls_head.cls = nn.Linear(in_features=model.cls_head.cls.in_features, out_features=args.num_classes)
        return model
    if model_name == "PVT_v2_b5":
        LOGGER.info("PVT_v2_b5")
        model = pvt_v2_b5()
        weight = torch.load(r"/home/uax/SCY/LiverClassification/Weights/pvt_v2_b5.pth", map_location="cpu")
        model.load_state_dict(weight)
        model.head = nn.Linear(in_features=model.head.in_features, out_features=args.num_classes)
        return model
    if model_name == "FocalNet":
        LOGGER.info("FocalNet")
        model = focalnet_base_lrf(focal_levels=[3, 3, 3, 3])
        weight = torch.load(r"/home/uax/SCY/LiverClassification/Weights/focalnet_base_lrf.pth", map_location="cpu")["model"]
        model.load_state_dict(weight)
        model.head = nn.Linear(in_features=model.head.in_features, out_features=args.num_classes)
        return model
    if model_name == "Resnet":
        LOGGER.info("Resnet")
        model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=args.num_classes)
        return model
    if model_name == "Densenet":
        LOGGER.info("Densenet")
        model = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=args.num_classes)
        return model
    if model_name == "Resnext":
        LOGGER.info("Resnext")
        model = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=args.num_classes)
        return model
    if model_name == "Thynet":
        LOGGER.info("Thynet")
        return ThyNet(num_classes=args.num_classes)
    if model_name == "Hiera":
        LOGGER.info("Hiera")
        model = Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3), drop_path_rate=0.0)
        weight = torch.load(r"/home/uax/SCY/LiverClassification/Weights/hiera_base_224.pth", map_location="cpu")["model_state"]
        model.load_state_dict(weight)
        model.head.projection = nn.Linear(in_features=model.head.projection.in_features, out_features=args.num_classes)
        return model
    if model_name == "Cswin":
        LOGGER.info("Cswin")
        model = cswin.CSWin_96_24322_base_224(drop_path_rate=0.0)
        pretrain_dict = torch.load("/home/uax/SCY/LiverClassification/Weights/cswin_base_224.pth")
        model.load_state_dict(pretrain_dict["state_dict_ema"])
        model.head = nn.Linear(in_features=model.head.in_features, out_features=args.num_classes)
        return model
    if model_name == "Convnext":
        LOGGER.info("Convnext")
        model = convnext_small(pretrained=False, in_22k=False, return_feature=False)
        pretrain_dict = torch.load(r"/home/uax/SCY/LiverClassification/Weights/convnext_small_22k_224.pth", map_location="cpu")
        model.load_state_dict(pretrain_dict["model"], strict=False)
        model.head = nn.Linear(in_features=model.head.in_features, out_features=args.num_classes)
        return model
    if model_name == "Convnext_B":
        LOGGER.info("Convnext_B")
        model = convnext_base(pretrained=False, in_22k=False, drop_path_rate=0.0, return_feature=False)
        pretrain_dict = torch.load(r"/home/uax/SCY/LiverClassification/Weights/convnext_base_22k_224.pth", map_location="cpu")
        model.load_state_dict(pretrain_dict["model"], strict=False)
        model.head = nn.Linear(in_features=model.head.in_features, out_features=args.num_classes)
        return model
    if model_name == "SwinTransformerV2":
        LOGGER.info("SwinTransformerV2")
        model = SwinTransformerV2(
            img_size=384,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=24,
            pretrained_window_sizes=[12, 12, 12, 6],
            drop_path_rate=0.2,
        )
        pretrain_dict = torch.load("/home/uax/SCY/LiverClassification/Weights/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth")
        model.load_state_dict(pretrain_dict["model"])
        model.head = nn.Linear(in_features=model.head.in_features, out_features=args.num_classes)
        return model
    if model_name == "SwinTransformer":
        LOGGER.info("SwinTransformer")
        model = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        model.head = nn.Linear(in_features=model.head.in_features, out_features=args.num_classes)
        return model
    if model_name == "SwinTransformer_S":
        LOGGER.info("SwinTransformer_S")
        model = swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)
        model.head = nn.Linear(in_features=model.head.in_features, out_features=args.num_classes)
        return model
    if model_name == "ViT":
        LOGGER.info("ViT")
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(in_features=model.heads.head.in_features, out_features=args.num_classes)
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def build_metric_collection(args):
    return MetricCollection(
        [
            Accuracy(task=args.categories, num_classes=args.num_classes, average="macro"),
            Precision(task=args.categories, num_classes=args.num_classes, average="macro"),
            Recall(task=args.categories, num_classes=args.num_classes, average="macro"),
            Specificity(task=args.categories, num_classes=args.num_classes, average="macro"),
            F1Score(task=args.categories, num_classes=args.num_classes, average="macro"),
            AUROC(task=args.categories, num_classes=args.num_classes, average="macro"),
        ]
    )


def log_dataset_distribution(name: str, dataframe: pd.DataFrame, num_classes: int) -> None:
    labels = dataframe["labels"]
    sample_count = len(labels)
    class_counts = labels.value_counts()
    distribution = []

    for class_id in range(num_classes):
        count = int(class_counts.get(class_id, 0))
        percentage = count / sample_count if sample_count else 0
        distribution.append(f"class {class_id}: {count} ({percentage:.1%})")

    for label, count in class_counts.items():
        if label not in range(num_classes):
            distribution.append(f"class {label}: {int(count)} ({count / sample_count:.1%})")

    missing_labels = int(labels.isna().sum())
    if missing_labels:
        distribution.append(
            f"missing label: {missing_labels} ({missing_labels / sample_count:.1%})"
        )

    summary = (
        f"{name} dataset: {sample_count} samples; "
        f"class distribution: {', '.join(distribution)}"
    )
    LOGGER.info("%s", summary)
    print(summary)


def build_data_loaders(args, k_fold: int):
    if not args.k_fold:
        raise ValueError("Please use 5-fold cross validation")

    data_dir = args.data_dir
    train_df = pd.read_csv(os.path.join(data_dir, f"train_{k_fold}.csv"))
    valid_df = pd.read_csv(os.path.join(data_dir, f"val_{k_fold}.csv"))
    multi_center_df = pd.read_csv(os.path.join(data_dir, "multicenter_test.csv"))
    high_risk_test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    single_center_test_df = pd.read_csv(os.path.join(data_dir, "test3.csv"))

    train7_df = pd.read_csv(os.path.join(data_dir, "train_7.csv"))
    train_df = pd.concat([train_df,train7_df], axis=0)

    multi_center_df = pd.concat([multi_center_df, valid_df], axis=0)

    train_dataset = Liver_Dataset(train_df, "train")
    val_dataset = Liver_Dataset(valid_df, "val")
    single_center_test_dataset = Liver_Dataset(single_center_test_df, "val")
    multi_center_test_dataset = Liver_Dataset(multi_center_df, "val")
    high_risk_test_dataset = Liver_Dataset(high_risk_test_df, "val")

    for split_name, dataset in (
        ("Train", train_dataset),
        ("Validation", val_dataset),
        ("Single-center test", single_center_test_dataset),
        ("Multi-center test", multi_center_test_dataset),
        ("High-risk test", high_risk_test_dataset),
    ):
        log_dataset_distribution(
            split_name,
            dataset.path_df,
            args.num_classes,
        )
    
    if args.weight_sampler:
        class_count = np.bincount(train_df["labels"].values)
        class_weights = 1.0 / class_count
        sample_weights = [class_weights[int(label)] for label in train_df["labels"].values]
        sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=int(2 * len(train_dataset)))
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    single_center_loader = DataLoader(single_center_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    multi_center_loader = DataLoader(multi_center_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    high_risk_loader = DataLoader(high_risk_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return {
        "train": train_loader,
        "val": val_loader,
        "single_center": single_center_loader,
        "multi_center": multi_center_loader,
        "high_risk": high_risk_loader,
    }


def build_optimizer(args, model):
    if not getattr(args, "train_backbone", False):
        frozen_backbones = []
        for name in ("swintransformer", "convnext"):
            backbone = getattr(model, name, None)
            if isinstance(backbone, nn.Module):
                for para in backbone.parameters():
                    para.requires_grad_(False)
                frozen_backbones.append(name)
        if not frozen_backbones:
            raise ValueError(
                "train_backbone=False requires a model with a "
                "'swintransformer' or 'convnext' backbone"
            )
        LOGGER.info("Frozen backbone modules: %s", ", ".join(frozen_backbones))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "stage1" in name:
                LOGGER.info("freeze_layers: %s", name)
                para.requires_grad_(False)
        trainable_params = [para for para in model.parameters() if para.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters remain for the optimizer")
        return optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_param_groups and hasattr(model, "swintransformer") and hasattr(model, "convnext"):
        swin_params = [p for p in model.swintransformer.parameters() if p.requires_grad]
        convnext_params = [p for p in model.convnext.parameters() if p.requires_grad]
        backbone_param_ids = {
            id(p) for p in model.swintransformer.parameters()
        } | {id(p) for p in model.convnext.parameters()}
        other_params = [
            p for p in model.parameters()
            if p.requires_grad and id(p) not in backbone_param_ids
        ]
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        grouped_params = swin_params + convnext_params + other_params
        if {id(p) for p in trainable_params} != {id(p) for p in grouped_params}:
            raise ValueError("Learning-rate parameter groups do not cover all trainable parameters")

        param_groups = [
            {"params": swin_params, "lr": args.lr},
            {"params": convnext_params, "lr": args.lr},
            {"params": other_params, "lr": args.lr * 2.0},
        ]
        param_groups = [group for group in param_groups if group["params"]]
        if not param_groups:
            raise ValueError("No trainable parameters remain for the optimizer")
        return optim.AdamW(
            param_groups,
            weight_decay=args.weight_decay,
        )

    trainable_params = [para for para in model.parameters() if para.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters remain for the optimizer")
    return optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(args, optimizer, train_loader):
    total_steps = args.epochs * len(train_loader)
    if total_steps <= 0:
        raise ValueError("CosineAnnealingLR requires at least one training step")

    if args.warmup_ratio is not None:
        if not 0 <= args.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in the range [0, 1)")
        warmup_steps = math.ceil(total_steps * args.warmup_ratio)
    else:
        if args.warmup_epochs < 0:
            raise ValueError("warmup_epochs cannot be negative")
        warmup_steps = math.ceil(len(train_loader) * args.warmup_epochs)

    if warmup_steps == 0:
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=0.0,
        )
    if warmup_steps >= total_steps:
        raise ValueError("Warmup steps must be fewer than total training steps")

    warmup = lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=0.0,
    )
    return lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def save_checkpoint(path: str, model, optimizer, scheduler, epoch: int, model_ema=None) -> None:
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_schedule": scheduler.state_dict(),
    }
    torch.save(checkpoint, path)

    if model_ema is not None:
        ema_path = path.replace("_val", "_ema")
        ema_checkpoint = {
            "epoch": epoch,
            "state_dict": model_ema.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_schedule": scheduler.state_dict(),
        }
        torch.save(ema_checkpoint, ema_path)


def evaluate_suite(model, loaders, criterion, metric_collection, device, args, logger):
    results = {}
    for name, key in [
        ("Single_center_test", "single_center"),
        ("Multi_center_test", "multi_center"),
        ("High_risk_test", "high_risk"),
    ]:
        test_loss = evaluate(model=model, data_loader=loaders[key], criterion=criterion, metric_collection=metric_collection, device=device, args=args)
        test_metric = metric_info(metric_collection)
        logger.info(f"{name}: loss:{test_loss:.4f}, {test_metric}")
        results[key] = {
            "loss": test_loss / len(loaders[key]),
            "metrics": test_metric,
        }
    return results


def save_training_curves(history, lr_history, output_path, fold_index):
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean batch loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_loss"], marker="o", color="tab:orange", label="Validation")
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean batch loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if lr_history:
        step_numbers = range(1, len(lr_history) + 1)
        group_count = len(lr_history[0])
        for group_index in range(group_count):
            rates = [step_lrs[group_index] for step_lrs in lr_history]
            axes[2].plot(step_numbers, rates, label=f"Parameter group {group_index + 1}")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Training step")
    axes[2].set_ylabel("Learning rate")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    figure.suptitle(f"Training Curves - Fold {fold_index}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_auc_curves(history, output_path, fold_index):
    epochs = range(1, len(history["auc/validation"]) + 1)
    figure, axis = plt.subplots(figsize=(10, 6))

    for dataset_name, values in history.items():
        if not dataset_name.startswith("auc/"):
            continue
        if len(values) != len(epochs):
            continue
        axis.plot(epochs, values, marker="o", label=dataset_name.removeprefix("auc/"))

    axis.set_title(f"AUROC Curves - Fold {fold_index}")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("AUROC")
    axis.set_ylim(0, 1)
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def log_metrics_to_tensorboard(writer, split_name, loss, metrics, epoch):
    writer.add_scalar(f"loss/{split_name}", loss, epoch)
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(
            f"metrics/{split_name}/{metric_name}",
            float(metric_value),
            epoch,
        )
        if split_name == "validation":
            writer.add_scalar(
                f"validation/{metric_name}",
                float(metric_value),
                epoch,
            )


def run_fold(rank: int, k_fold: int, args) -> None:
    if not torch.cuda.is_available():
        raise EnvironmentError("not find GPU device for training.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    set_seed(args.seed + k_fold)
    normalize_model_name(args)
    model = build_model(args)
    model.to(rank)
    log_model_statistics(model, LOGGER)
    loaders = build_data_loaders(args, k_fold)
    use_focal_loss = args.focal_loss
    criterion_focal = FocalLoss(alpha=0.25, gamma=2, task_type="multi-class", label_smoothing=args.smoothing, num_classes=args.num_classes) if use_focal_loss else None
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    LOGGER.info("Model Name: %s", args.model)
    LOGGER.info("Model Configuration:\n%s", model)
    for attribute, value in vars(args).items():
        LOGGER.info("%s: %s", attribute, value)

    metric_collection = build_metric_collection(args)
    metric_collection.to("cpu")

    if args.visual_feature:
        checkpoint = torch.load(os.path.join(args.checkpoints, f"{args.model}_val{k_fold}.pth"), map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        visual_feature(model, metric_collection=metric_collection, device=rank, data_loader=loaders["val"])
        return

    if args.cam_visualization:
        CAM_visualization(model, loaders["val"])
        return

    if args.test:
        checkpoint = torch.load(os.path.join(args.checkpoints, f"{args.model}_val{k_fold}.pth"), map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        test_writer = SummaryWriter(
            log_dir=os.path.join(
                args.log_dir, "tensorboard", args.model, f"fold_{k_fold}"
            )
        )
        try:
            test_results = evaluate_suite(
                model, loaders, criterion, metric_collection, rank, args, LOGGER
            )
            checkpoint_epoch = checkpoint.get("epoch", 0)
            for dataset_name, result in test_results.items():
                log_metrics_to_tensorboard(
                    test_writer,
                    dataset_name,
                    result["loss"],
                    result["metrics"],
                    checkpoint_epoch,
                )
            test_writer.flush()
        finally:
            test_writer.close()
        return

    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV3(model, decay=args.model_ema_decay, use_warmup=args.model_ema_warmup, device="cpu" if args.model_ema_force_cpu else None)

    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, loaders["train"])
    scaler = torch.cuda.amp.GradScaler() if args.apm else None
    mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, num_classes=args.num_classes) if args.mixup else None

    best_val_auc = 0.0
    num_updates = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "auc/validation": [],
        "auc/train": [],
        "auc/single_center": [],
        "auc/multi_center": [],
        "auc/high_risk": [],
    }
    lr_history = []
    tensorboard_dir = os.path.join(args.log_dir, "tensorboard", args.model, f"fold_{k_fold}")
    writer = SummaryWriter(log_dir=tensorboard_dir)
    logged_lr_steps = 0

    if args.resume:
        checkpoint = torch.load(os.path.join(args.checkpoints, f"{args.model}_ema{k_fold}.pth"), map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])

    for epoch in range(args.epochs):
        if args.lr_param_groups and len(optimizer.param_groups) >= 3:
            LOGGER.info(
                " k_fold %s epoch %s LR backbone: %s LR other: %s LR other: %s",
                k_fold,
                epoch,
                optimizer.param_groups[0]["lr"],
                optimizer.param_groups[1]["lr"],
                optimizer.param_groups[2]["lr"],
            )
        else:
            LOGGER.info(" k_fold %s epoch %s LR backbone: %s", k_fold, epoch, optimizer.param_groups[0]["lr"])

        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            metric_collection=metric_collection,
            num_updates=num_updates,
            epoch=epoch,
            scheduler=scheduler,
            criterion_focal=criterion_focal,
            mixup_fn=mixup_fn,
            data_loader=loaders["train"],
            device=rank,
            ema_updata_epoch=args.ema_updata_epoch,
            criterion=criterion,
            scaler=scaler,
            model_ema=model_ema,
            lr_history=lr_history,
        )
        mean_train_loss = train_loss / len(loaders["train"])
        history["train_loss"].append(mean_train_loss)
        train_metric = metric_info(metric_collection)
        history["auc/train"].append(train_metric["AUROC"])
        log_metrics_to_tensorboard(
            writer, "train", mean_train_loss, train_metric, epoch
        )
        LOGGER.info(f"Train: loss:{train_loss:.4f}, {train_metric}")

        val_loss = evaluate(model=model, data_loader=loaders["val"], criterion=criterion, metric_collection=metric_collection, device=rank, args=args)
        mean_val_loss = val_loss / len(loaders["val"])
        history["val_loss"].append(mean_val_loss)
        val_metric = metric_info(metric_collection)
        history["auc/validation"].append(val_metric["AUROC"])
        log_metrics_to_tensorboard(
            writer, "validation", mean_val_loss, val_metric, epoch
        )
        LOGGER.info(f"Val: loss:{val_loss:.4f}, {val_metric}")
        test_results = evaluate_suite(
            model, loaders, criterion, metric_collection, rank, args, LOGGER
        )
        for dataset_name, result in test_results.items():
            history[f"auc/{dataset_name}"].append(result["metrics"]["AUROC"])
            log_metrics_to_tensorboard(
                writer,
                dataset_name,
                result["loss"],
                result["metrics"],
                epoch,
            )

        for step, step_lrs in enumerate(lr_history[logged_lr_steps:], start=logged_lr_steps):
            for group_index, learning_rate in enumerate(step_lrs):
                writer.add_scalar(f"learning_rate/group_{group_index + 1}", learning_rate, step)
        logged_lr_steps = len(lr_history)
        writer.flush()



        if val_metric["AUROC"] > best_val_auc:
            best_val_auc = val_metric["AUROC"]
            if model_ema is not None:
                num_updates += 1
                model_ema.update(model, step=num_updates)
            save_checkpoint(
                os.path.join(args.checkpoints, f"{args.model}_val{k_fold}.pth"),
                model,
                optimizer,
                scheduler,
                epoch,
                model_ema=model_ema,
            )

    writer.close()
    plot_path = os.path.join(args.log_dir, f"{args.model}_fold{k_fold}_training_curves.png")
    save_training_curves(history, lr_history, plot_path, k_fold)
    LOGGER.info("Saved training curves to %s", plot_path)
    auc_plot_path = os.path.join(args.log_dir, f"{args.model}_fold{k_fold}_auc_curves.png")
    save_auc_curves(history, auc_plot_path, k_fold)
    LOGGER.info("Saved AUROC curves to %s", auc_plot_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default=os.path.join(os.path.dirname(__file__), "logs"))
    parser.add_argument("--data_dir", type=str, default='/home/uax/SCY/HSQ/HSQformer/dataset')
    parser.add_argument("--mixup", type=bool, default=True, help="mixup alpha, mixup enabled if > 0. (default: 0.)")
    parser.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)")
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--categories", type=str, default="binary", choices=["binary", "multiclass", "multilabel"])
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--optimizer", default="AdamW", help="optimizer")
    parser.add_argument("--seed", type=int, default=42, help="set seed")
    parser.add_argument("--resume", type=bool, default=False, help="put the path to resuming file if needed")
    parser.add_argument("--criterion", default="CrossEntropyLoss", help="criterion")
    parser.add_argument("--focal_loss",default=True,help="Enable or disable Focal Loss",)
    parser.add_argument("--lr_param_groups", type=bool, default=True, help="lr_param_groups")
    parser.add_argument("--weight_sampler", type=bool, default=True, help="weight sampler")
    parser.add_argument("--model-ema-force-cpu", action="store_true", default=False, help="Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.")
    parser.add_argument("--model-ema-decay", type=float, default=0.9998, help="Decay factor for model weights moving average (default: 0.9998)")
    parser.add_argument("--model-ema-warmup", action="store_true", help="Enable warmup for model EMA decay.")
    parser.add_argument("--ema_updata_epoch", type=int, default=100, help="Number of epochs to update EMA model")
    parser.add_argument("--model_ema", type=bool, default=False, help="Enable or disable model EMA")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--freeze_layers", type=bool, default=False)
    parser.add_argument("--train_backbone",default=True,help="Enable or freeze backbone training")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    warmup_group = parser.add_mutually_exclusive_group()
    warmup_group.add_argument("--warmup_epochs", type=float, default=1, help="Warmup duration in epochs (default: 1)")
    warmup_group.add_argument("--warmup_ratio", type=float, default=None, help="Warmup fraction of all training steps, e.g. 0.05")
    parser.add_argument("--apm", type=bool, default=True, help="Automatic Mixed Precision")
    parser.add_argument("--distributed", type=bool, default=False, help="distributed")
    parser.add_argument("--cam_visualization", type=bool, default=False)
    parser.add_argument("--visual_feature", type=bool, default=False)
    parser.add_argument("--k_fold", type=bool, default=True)
    parser.add_argument("--statistics", type=bool, default=False)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--checkpoints", type=str, default=os.path.join(os.path.dirname(__file__), "checkpoint_2026_10_12"))

    parser.add_argument(
        "--model",
        type=str,
        default="LENet",
        choices=[
            "Resnet",
            "Mobilenet",
            "Cswin",
            "Convnext",
            "SwinTransformerV2",
            "SwinTransformer",
            "Load_contrastive",
            "DaViT",
            "Liver",
            "Hiera",
            "Densenet",
            "Resnext",
            "Thynet",
            "ViT",
            "LivNet_Baseline",
            "SwinTransformer_S",
            "LivNet_no_moe",
            "LivNet_MOE",
            "LENet",
            "Convnext_B",
            "PVT_v2_b5",
            "FocalNet",
            "Mpvit_base",
        ],
    )
    parser.add_argument("--serial_parallel", type=str, default="serial", choices=["parallel", "serial", "invert_serial"], help="Types of attention")
    parser.add_argument("--sparse_dense", type=str, default="sparse_token", choices=["sparse_token", "dense_token", "sparse_expert", "dense_expert", "mlp"], help="Types of MOE")
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--head_type", type=str, default="linear", choices=["moe_head", "linear", "mlp"])
    parser.add_argument("--cat_moe_head", type=bool, default=False)
    parser.add_argument("--q_former_depths", type=int, nargs="*", default=[1, 1, 1, 1])
    parser.add_argument("--stage_dims", type=int, nargs="*", default=[96, 192, 384, 768])
    parser.add_argument("--q_former_head_num", type=int, nargs="*", default=[3, 6, 12, 24])
    parser.add_argument("--num_query_tokens", type=int, default=200)
    parser.add_argument("--query_dim", type=int, default=384)
    args = parser.parse_args()
    if args.warmup_epochs is None and args.warmup_ratio is None:
        args.warmup_epochs = 1.0
    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.checkpoints, exist_ok=True)

    if args.k_fold:
        for idx in range(5):
            print(f"第{idx}折")
            normalize_model_name(args)
            configure_logger(args.log_dir, args, idx)
            run_fold(0, idx, args)
    else:
        normalize_model_name(args)
        configure_logger(args.log_dir, args, 0)
        run_fold(0, 0, args)
