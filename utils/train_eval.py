import math
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.utils import make_grid
from tqdm import tqdm


def metric_info(metric_collection):
    """Return metric values as rounded, CPU-backed Python scalars."""
    return {
        name.replace("Binary", ""): np.around(value.detach().cpu().item(), decimals=4)
        for name, value in metric_collection.compute().items()
    }


def KD_loss(logits_student, logits_teacher, temperature=1.0):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss = F.kl_div(log_pred_student, pred_teacher, reduction="batchmean")
    return loss * temperature**2


def DKD_loss(logits_student, logits_teacher, target, alpha=1.0, beta=8.0, temperature=1.0):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_target_probs = cat_mask(pred_student, gt_mask, other_mask)
    teacher_target_probs = cat_mask(pred_teacher, gt_mask, other_mask)
    target_loss = F.kl_div(
        torch.log(student_target_probs),
        teacher_target_probs,
        reduction="sum",
    )
    target_loss = target_loss * temperature**2 / target.shape[0]

    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    non_target_loss = F.kl_div(
        log_pred_student_part2,
        pred_teacher_part2,
        reduction="sum",
    )
    non_target_loss = non_target_loss * temperature**2 / target.shape[0]
    return alpha * target_loss + beta * non_target_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1, 1)
    return torch.zeros_like(logits).scatter_(1, target, 1).bool()


def _get_other_mask(logits, target):
    target = target.reshape(-1, 1)
    return torch.ones_like(logits).scatter_(1, target, 0).bool()


def cat_mask(values, target_mask, other_mask):
    target_values = (values * target_mask).sum(dim=1, keepdim=True)
    other_values = (values * other_mask).sum(dim=1, keepdim=True)
    return torch.cat([target_values, other_values], dim=1)


def _resolve_device(device):
    if isinstance(device, int):
        return torch.device("cuda", device)
    return torch.device(device)


def _autocast(device):
    return torch.amp.autocast(
        device_type=device.type,
        enabled=device.type == "cuda",
    )


def train_one_epoch(
    model,
    optimizer,
    metric_collection=None,
    data_loader=None,
    device=0,
    num_updates=0,
    epoch=0,
    criterion_focal=None,
    scheduler=None,
    criterion=None,
    mixup_fn=None,
    scaler=None,
    model_ema=None,
    ema_updata_epoch=None,
    lr_history=None,
):
    """Run one training epoch and return the sum of batch losses."""
    device = _resolve_device(device)
    if metric_collection is not None:
        metric_collection.reset()
    model.train()
    total_loss = 0.0
    progress = tqdm(data_loader, desc="train")

    for images, labels in progress:
        if lr_history is not None:
            lr_history.append([group["lr"] for group in optimizer.param_groups])

        targets = labels.to(device)
        images = images.to(device)
        if mixup_fn is not None:
            images, mixed_labels = mixup_fn(images, targets)
        else:
            mixed_labels = targets

        with _autocast(device):
            outputs = model(images)
            if criterion_focal is not None:
                loss = criterion(outputs, mixed_labels) + criterion_focal(
                    outputs, mixed_labels
                )
            else:
                loss = criterion(outputs, mixed_labels)

        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")

        optimizer.zero_grad()
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        progress.set_description(f"loss: {loss.item():.6f}")
        if metric_collection is not None:
            probabilities = torch.softmax(outputs.detach(), dim=1)
            metric_collection.update(
                probabilities[:, 1].cpu(),
                targets.detach().cpu(),
            )
        if scheduler is not None:
            scheduler.step()

    return total_loss


@torch.no_grad()
def evaluate(
    model,
    metric_collection=None,
    data_loader=None,
    device=0,
    criterion=None,
    args=None,
    k_fold=None,
):
    """Evaluate a model and return the sum of batch losses."""
    device = _resolve_device(device)
    if metric_collection is not None:
        metric_collection.reset()
    model.eval()
    total_loss = 0.0
    progress = tqdm(data_loader, desc="validation", file=sys.stdout)
    image_paths = []
    positive_scores = []
    all_labels = []

    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        with _autocast(device):
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item()
        progress.set_description(f"loss: {loss.item():.4f}")
        probabilities = outputs.softmax(dim=1).detach().cpu()

        if metric_collection is not None:
            metric_collection.update(probabilities[:, 1], labels.detach().cpu())

        if args is not None and args.output_csv is not None:
            image_paths.extend(batch_paths)
            positive_scores.extend(probabilities[:, 1].numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    if args is not None and args.output_csv is not None:
        predictions = pd.DataFrame(
            {
                "id": image_paths,
                "M": positive_scores,
                "label": all_labels,
            }
        )
        predictions.to_csv(f"{args.output_csv}{k_fold}.csv", index=False)

    return total_loss


@torch.no_grad()
def test_prob(
    model,
    metric_collection=None,
    num_classes=2,
    data_loader=None,
    device=0,
    optimizer=None,
    criterion=None,
    mixup_fn=None,
    scaler=None,
    aux_loss=None,
    model_ema=None,
    criterion_focal=None,
    output_excel=False,
    k_flod=None,
):
    """Export predictions, update metrics, and return the sum of batch losses."""
    device = _resolve_device(device)
    model.eval()
    if metric_collection is not None:
        metric_collection.reset()

    predictions = []
    labels_all = []
    image_paths = []
    total_loss = 0.0

    for images, labels, batch_paths in tqdm(data_loader, desc="test", file=sys.stdout):
        images = images.to(device)
        labels = labels.to(device)
        with _autocast(device):
            outputs = model(images)
            if isinstance(outputs, list):
                batch_loss = sum(criterion(output, labels) for output in outputs)
                batch_predictions = torch.stack(outputs).sum(dim=0)
            else:
                batch_predictions = outputs
                batch_loss = criterion(batch_predictions, labels) if criterion is not None else None

        if batch_loss is not None:
            total_loss += batch_loss.item()
        predictions.append(batch_predictions.detach())
        labels_all.append(labels.detach())
        image_paths.extend(batch_paths)

    if not predictions:
        raise ValueError("Cannot run test_prob with an empty data loader")

    probabilities = torch.cat(predictions).softmax(dim=1).cpu()
    labels_cpu = torch.cat(labels_all).cpu()
    predicted_labels = probabilities.argmax(dim=1)
    results = pd.DataFrame(
        {
            "id": image_paths,
            "B": probabilities[:, 0].numpy(),
            "M": probabilities[:, 1].numpy(),
            "AI_label": predicted_labels.numpy(),
            "label": labels_cpu.numpy(),
        }
    )
    results.to_csv(f"LivNet_3020_test{k_flod}.csv", index=True)

    if metric_collection is not None:
        metric_collection.update(predicted_labels.float(), labels_cpu)
    return total_loss
@torch.no_grad()
def liver_2classification_test(
    model,
    metric_collection=None,
    num_classes=2,
    data_loader=None,
    device=0,
    criterion=None,
    mixup_fn=None,
    scaler=None,
    aux_loss=None,
    model_ema=None,
    criterion_focal=None,
    output_excel=False,
):
    """Average logits over five passes and update metrics with predicted labels."""
    device = _resolve_device(device)
    model.eval()
    if metric_collection is not None:
        metric_collection.reset()

    averaged_logits = None
    labels_all = None
    passes = 5

    for _ in range(passes):
        pass_logits = []
        pass_labels = []
        for images, labels in tqdm(data_loader, desc="test", file=sys.stdout):
            images = images.to(device)
            labels = labels.to(device)
            with _autocast(device):
                outputs = model(images)
                if not isinstance(outputs, (list, tuple)):
                    raise TypeError("Expected the model to return a sequence of logits")
                batch_logits = torch.stack(outputs).sum(dim=0)
            pass_logits.append(batch_logits)
            pass_labels.append(labels)

        if not pass_logits:
            raise ValueError("Cannot run liver_2classification_test with an empty data loader")
        logits = torch.cat(pass_logits, dim=0)
        labels_all = torch.cat(pass_labels, dim=0)
        averaged_logits = logits if averaged_logits is None else averaged_logits + logits

    predicted_labels = (averaged_logits / passes).softmax(dim=1).argmax(dim=1)
    if metric_collection is not None:
        metric_collection.update(predicted_labels.float(), labels_all.cpu())
    return 0.0


@torch.no_grad()
def rl_extract_feature(
    model,
    metric_collection=None,
    data_df_len=None,
    num_classes=2,
    data_loader=None,
    device=0,
    criterion=None,
    mixup_fn=None,
    scaler=None,
    aux_loss=None,
    model_ema=None,
    criterion_focal=None,
    output_excel=False,
):
    """Extract concatenated stage features, probabilities, and labels."""
    device = _resolve_device(device)
    model.eval()
    if metric_collection is not None:
        metric_collection.reset()
    if not aux_loss:
        raise ValueError("rl_extract_feature requires aux_loss=True")

    extracted_batches = []
    for images, labels in tqdm(data_loader, desc="extract features", file=sys.stdout):
        images = images.to(device)
        labels = labels.to(device)
        targets = labels.float().unsqueeze(1)

        with _autocast(device):
            probabilities, stage_features = model(images)
            pooled_features = [
                feature.mean(dim=(-2, -1))
                for feature in stage_features
            ]
            combined_features = torch.cat(pooled_features, dim=1)
            combined_logits = torch.stack(probabilities).sum(dim=0)
            batch_features = torch.cat(
                (combined_features, combined_logits.softmax(dim=1), targets),
                dim=1,
            )

        extracted_batches.append(batch_features.cpu())
        if metric_collection is not None:
            metric_collection.update(
                combined_logits.argmax(dim=1).cpu(),
                labels.cpu(),
            )

    if not extracted_batches:
        return np.empty((0, 0), dtype=np.float32)
    return torch.cat(extracted_batches, dim=0).numpy()


@torch.no_grad()
def visual_feature(
    model,
    metric_collection=None,
    data_df_len=None,
    num_classes=2,
    data_loader=None,
    device=0,
    criterion=None,
    mixup_fn=None,
    scaler=None,
    aux_loss=None,
    model_ema=None,
    criterion_focal=None,
    output_excel=False,
    output_dir="/home/uax/SCY/Decouple_liver/visual_feature/combine",
):
    """Save one channel-grid image per returned model feature stage."""
    device = _resolve_device(device)
    model.eval()
    if metric_collection is not None:
        metric_collection.reset()
    os.makedirs(output_dir, exist_ok=True)

    for batch in tqdm(data_loader, desc="visualize features", file=sys.stdout):
        images = batch[0].to(device)
        with _autocast(device):
            model_output = model(images)

        if isinstance(model_output, tuple) and len(model_output) == 2:
            _, features = model_output
        else:
            features = model_output

        if isinstance(features, torch.Tensor):
            features = [features]

        for stage_index, feature in enumerate(features):
            if feature.ndim == 3:
                batch_size, token_count, channels = feature.shape
                side_length = math.isqrt(token_count)
                if side_length * side_length != token_count:
                    raise ValueError(
                        f"Stage {stage_index} has {token_count} tokens; cannot reshape to a square feature map"
                    )
                feature = feature.transpose(1, 2).reshape(
                    batch_size, channels, side_length, side_length
                )
            elif feature.ndim != 4:
                continue

            channel_maps = feature[0, :48].detach().cpu()
            grid = make_grid(channel_maps.unsqueeze(1), normalize=True, nrow=8)
            figure, axis = plt.subplots(figsize=(12, 12))
            axis.imshow(grid.permute(1, 2, 0).numpy())
            axis.axis("off")
            figure.savefig(
                os.path.join(output_dir, f"stage_{stage_index}.png"),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close(figure)


def reshape_transform(tensor, height=20, width=20):
    """Convert token features shaped [batch, tokens, channels] to NCHW."""
    return tensor.reshape(
        tensor.size(0),
        height,
        width,
        tensor.size(2),
    ).permute(0, 3, 1, 2)


def CAM_visualization(
    model,
    data_loader,
    device=0,
    checkpoint_path="/media/uax/CA4E64EFA9C3DA83/HCC/ablation/LENet-serial-sparse_token-4experts-top_2-linear-q_former_depths2_2_6_2-num_query_400-query_dim_768_val0.pth",
    output_path="/home/uax/SCY/Decouple_liver/visual_feature/cam/LIVER",
):
    device = _resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    os.makedirs(output_path, exist_ok=True)
    model.eval()
    cam = GradCAM(
        model=model,
        target_layers=[model.sc_att_mlp[-1]],
        reshape_transform=reshape_transform,
    )

    for images, _, image_paths in tqdm(data_loader, desc="cam", file=sys.stdout):
        images = images.to(device)
        grayscale_cam = cam(
            input_tensor=images,
            eigen_smooth=False,
            aug_smooth=False,
        )[0]
        image_path = image_paths[0]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0
        cam_image = show_cam_on_image(image, grayscale_cam, use_rgb=True)
        image_name = os.path.basename(image_path).split("_")[0]
        output_file = os.path.join(output_path, f"{image_name}.png")
        if not cv2.imwrite(output_file, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Could not save CAM image: {output_file}")
