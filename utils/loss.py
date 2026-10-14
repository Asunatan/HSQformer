import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss

class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None,n_classes=2, size_average=True,e=1e-7):
        super(BCEDiceLoss, self).__init__()
        self.e=e
        self.n_classes=n_classes

    def forward(self, pred, mask):
        batch_size, channel, _, _ = pred.size()

        # BCE_loss
        bce_loss = nn.CrossEntropyLoss()(pred, mask.long())

        # Dice_loss
        pred = torch.softmax(pred, dim=1).contiguous().view(batch_size, self.n_classes, -1)
        mask = F.one_hot(mask.long(), num_classes=self.n_classes).contiguous().permute(0, 3, 1, 2).view(batch_size,
                                                                                                        self.n_classes,
                                                                                                        -1)
        inter = torch.sum(pred * mask, 2)
        union = torch.sum(pred, 2) + torch.sum(mask, 2)
        dice = (2.0 * inter + self.e) / (union + self.e)
        dice = dice.mean()

        return 0.4*bce_loss + 0.6*(1 - dice)


def focal_loss(logits, labels, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      logits: A float tensor of size [batch, num_classes].
      labels: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    bce_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * bce_loss

    weighted_loss = alpha * loss
    loss = torch.sum(weighted_loss)
    loss /= torch.sum(labels)
    return loss


class ClassBalancedLoss(torch.nn.Module):
    def __init__(self, samples_per_class=None, num_classes=2, label_smoothing=False, beta=0.9999, gamma=2,mixup=None,
                 loss_type="focal"):
        super(ClassBalancedLoss, self).__init__()
        if loss_type not in ["focal", "sigmoid", "softmax"]:
            loss_type = "focal"
        if samples_per_class is None:
            samples_per_class = [1] * num_classes
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / (np.array(effective_num) + float("1e-8"))
        self.constant_sum = len(samples_per_class)#总类别数
        weights = (weights / np.sum(weights) * self.constant_sum).astype(np.float32)

        self.class_weights = weights
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.mixup = mixup
        if self.mixup :
            self.is_smoothing = False
        else :
            self.is_smoothing = label_smoothing

    def update(self, samples_per_class):#这里有个问题，如果全部抽到一类的怎么解决的？这个问题还大概率出现在二分类任务上
        samples_per_class=samples_per_class.cpu().numpy()
        if samples_per_class is None:
            return
        effective_num = 1.0 - np.power(self.beta, samples_per_class)
        weights = (1.0 - self.beta) / np.array(effective_num)
        self.constant_sum = len(samples_per_class)
        weights = (weights / np.sum(weights) * self.constant_sum).astype(np.float32)
        self.class_weights = weights

    def specific_smoothing(self, label, smoothing_rate=0.9):
        """
        para: label one_hot [N, num_cls]
        """
        num_class = label.shape[1]
        cls_foronebatch = label.argmax(dim=1)  # [N, 1]
        for i, cls_id in enumerate(cls_foronebatch):
            label_onehot = torch.zeros(1, num_class) * (1 - smoothing_rate) / (num_class - 1)
            label_onehot[:, cls_id] = smoothing_rate
            label[i] = label_onehot
        return label

    def forward(self, x, y):
        _, num_classes = x.shape
        if len(y.shape)>1:
            labels_one_hot=y
            weights = torch.tensor(self.class_weights, device=x.device).index_select(0, y.argmax(1))
        else:
            labels_one_hot = F.one_hot(y, num_classes).float()
            labels_one_hot = self.specific_smoothing(labels_one_hot)
            weights = torch.tensor(self.class_weights, device=x.device).index_select(0, y)
        #weights = torch.tensor(self.class_weights, device=x.device).index_select(0, y.argmax(1))

        weights = weights.unsqueeze(1)

        if self.loss_type == "focal":
            cb_loss = focal_loss(x, labels_one_hot, weights, self.gamma)
            # cb_loss = self.loss(x)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(x, labels_one_hot, weights)
        else:  # softmax
            pred = x.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(pred, labels_one_hot, weights)
        return cb_loss



class Simple_FocalLoss(nn.Module):#fast ai库和
    def __init__(self, alpha=None,  gamma=2,label_smoothing=0. ,reduction='mean'):
        super(Simple_FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ce = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none')
        self.reduction = reduction
    def forward(self, input, target):
        logp = self.ce(input, target)
        pt = torch.exp(-logp)
        loss = (1 - pt) ** self.gamma * logp
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss




class FocalLoss(nn.Module):
    """
    Unified Focal Loss for binary, multi-class, and multi-label classification.

    Supports:
      - Hard labels
      - Soft labels (probability distributions)
      - Label smoothing (applied internally for hard labels)
      - Class balancing via alpha (scalar or per-class)

    Args:
        gamma (float): Focusing parameter.
        alpha: Balancing factor.
               - binary / multi-label: scalar in [0, 1] (positive-class weight)
               - multi-class: scalar or sequence of length num_classes
               - None: no balancing
        reduction (str): 'none' | 'mean' | 'sum'
        task_type (str): 'binary' | 'multi-class' | 'multi-label'
        num_classes (int): Required for 'multi-class'.
        label_smoothing (float): In [0, 1). Only applied when targets are hard labels.
    """

    def __init__(
        self,
        gamma=2.0,
        alpha=None,
        reduction="mean",
        task_type="multi-class",
        num_classes=None,
        label_smoothing=0.0,
    ):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        if task_type not in {"binary", "multi-class", "multi-label"}:
            raise ValueError(
                "task_type must be 'binary', 'multi-class', or 'multi-label'"
            )
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if task_type == "multi-class" and num_classes is None:
            raise ValueError("num_classes must be specified for multi-class tasks")

        self.gamma = float(gamma)
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes
        self.label_smoothing = float(label_smoothing)

        # Normalize alpha to a tensor (kept on CPU; moved to device in forward).
        self.alpha = self._normalize_alpha(alpha, task_type, num_classes)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_alpha(alpha, task_type, num_classes):
        if alpha is None:
            return None
        if isinstance(alpha, (list, tuple)):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, (int, float)):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().clone().float()
        else:
            raise TypeError("alpha must be None, a number, a list/tuple, or a tensor")

        if task_type == "multi-class":
            if alpha.dim() == 0:
                pass  # scalar
            elif alpha.numel() == num_classes:
                alpha = alpha.reshape(-1)
            else:
                raise ValueError(
                    "multi-class alpha must be a scalar or have one weight per class"
                )
        else:
            if alpha.numel() != 1:
                raise ValueError(
                    f"{task_type} alpha must be a scalar positive-class weight"
                )
            alpha = alpha.reshape(())
        return alpha

    def _reduce(self, loss):
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    # ------------------------------------------------------------------ #
    # Entry
    # ------------------------------------------------------------------ #
    def forward(self, inputs, targets):
        if self.task_type == "binary":
            return self._binary_forward(inputs, targets)
        if self.task_type == "multi-class":
            return self._multi_class_forward(inputs, targets)
        return self._multi_label_forward(inputs, targets)

    # ------------------------------------------------------------------ #
    # Binary
    # ------------------------------------------------------------------ #
    def _binary_forward(self, inputs, targets):
        # Accept logits shaped [B] or [B, 1]
        if inputs.dim() == 2 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)
        if inputs.dim() != 1:
            raise ValueError("binary inputs must have shape [B] or [B, 1]")

        if targets.dim() == 2 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        if targets.dim() != 1 or targets.shape != inputs.shape:
            raise ValueError("binary targets must have the same shape as inputs")

        targets = targets.to(device=inputs.device, dtype=inputs.dtype)
        if torch.any((targets < 0) | (targets > 1)):
            raise ValueError("binary targets must be in [0, 1]")

        # Apply label smoothing only for hard labels.
        is_hard = bool(torch.all((targets == 0) | (targets == 1)))
        if is_hard and self.label_smoothing > 0:
            eps = self.label_smoothing
            targets = targets * (1.0 - eps) + eps * 0.5

        # Numerically stable per-class log-probs
        log_p = F.logsigmoid(inputs)          # log(sigmoid(x))
        log_1p = F.logsigmoid(-inputs)        # log(1 - sigmoid(x))
        p = log_p.exp()
        p1 = p.clamp(0.0, 1.0)

        # Per-branch focal weights
        loss_pos = -((1.0 - p1) ** self.gamma) * log_p   # weight for y = 1
        loss_neg = -(p1 ** self.gamma) * log_1p          # weight for y = 0

        if self.alpha is not None:
            alpha = self.alpha.to(device=inputs.device, dtype=inputs.dtype)
            loss = targets * alpha * loss_pos + (1.0 - targets) * (1.0 - alpha) * loss_neg
        else:
            loss = targets * loss_pos + (1.0 - targets) * loss_neg

        return self._reduce(loss)

    # ------------------------------------------------------------------ #
    # Multi-class
    # ------------------------------------------------------------------ #
    def _multi_class_forward(self, inputs, targets):
        if inputs.dim() != 2:
            raise ValueError("multi-class inputs must have shape [B, C]")
        B, C = inputs.shape
        if C != self.num_classes:
            raise ValueError(
                f"inputs has {C} classes but num_classes={self.num_classes}"
            )

        log_probs = F.log_softmax(inputs, dim=1)     # [B, C]
        probs = log_probs.exp()                      # [B, C]

        # Build soft target distribution [B, C]
        if targets.dim() == 1:
            targets = targets.to(device=inputs.device, dtype=torch.long)
            if torch.any((targets < 0) | (targets >= C)):
                raise ValueError("target class index out of range")
            target_probs = F.one_hot(targets, num_classes=C).to(log_probs.dtype)
            if self.label_smoothing > 0:
                eps = self.label_smoothing
                target_probs = target_probs * (1.0 - eps) + eps / C
        elif targets.dim() == 2 and targets.shape == inputs.shape:
            target_probs = targets.to(device=inputs.device, dtype=log_probs.dtype)
            if torch.any((target_probs < 0) | (target_probs > 1)):
                raise ValueError("soft targets must be in [0, 1]")
        else:
            raise ValueError(
                "multi-class targets must be class indices [B] or "
                "soft labels with shape [B, C]"
            )

        # Per-class focal weighting (works for hard and soft targets).
        focal_weight = (1.0 - probs.clamp(0.0, 1.0)) ** self.gamma
        loss = -(target_probs * focal_weight * log_probs)   # [B, C]

        if self.alpha is not None:
            alpha = self.alpha.to(device=inputs.device, dtype=log_probs.dtype)
            if alpha.numel() == 1:
                loss = loss * alpha
            else:
                loss = loss * alpha.reshape(1, C)

        loss = loss.sum(dim=1)   # [B]

        return self._reduce(loss)

    # ------------------------------------------------------------------ #
    # Multi-label
    # ------------------------------------------------------------------ #
    def _multi_label_forward(self, inputs, targets):
        if inputs.dim() != 2:
            raise ValueError("multi-label inputs must have shape [B, C]")

        targets = targets.to(device=inputs.device, dtype=inputs.dtype)
        if targets.shape != inputs.shape:
            raise ValueError("multi-label targets must have the same shape as inputs")
        if torch.any((targets < 0) | (targets > 1)):
            raise ValueError("multi-label targets must be in [0, 1]")

        # Optional label smoothing for hard (0/1) multi-label targets
        is_hard = bool(torch.all((targets == 0) | (targets == 1)))
        if is_hard and self.label_smoothing > 0:
            eps = self.label_smoothing
            targets = targets * (1.0 - eps) + eps * 0.5

        log_p = F.logsigmoid(inputs)          # log(sigmoid(x))
        log_1p = F.logsigmoid(-inputs)        # log(1 - sigmoid(x))
        p = log_p.exp().clamp(0.0, 1.0)

        loss_pos = -((1.0 - p) ** self.gamma) * log_p
        loss_neg = -(p ** self.gamma) * log_1p

        if self.alpha is not None:
            alpha = self.alpha.to(device=inputs.device, dtype=inputs.dtype)
            loss = targets * alpha * loss_pos + (1.0 - targets) * (1.0 - alpha) * loss_neg
        else:
            loss = targets * loss_pos + (1.0 - targets) * loss_neg

        return self._reduce(loss)


class DiceLoss(nn.Module):
    def __init__(self, weight=None,n_classes=2, size_average=True,e=1e-7):
        super(DiceLoss, self).__init__()
        self.e=e
        self.n_classes=n_classes

    def forward(self, pred, mask):
        batch_size, channel, _, _ = pred.size()
        # Dice_loss
        pred = torch.softmax(pred, dim=1).contiguous().view(batch_size, self.n_classes, -1)
        mask = F.one_hot(mask.long(), num_classes=self.n_classes).contiguous().permute(0, 3, 1, 2).view(batch_size,
                                                                                                        self.n_classes,
                                                                                                        -1)
        inter = torch.sum(pred * mask, 2)
        union = torch.sum(pred, 2) + torch.sum(mask, 2)
        dice = (2.0 * inter + self.e) / (union + self.e)
        dice = dice.mean()

        return 1 - dice

####################################################################################################3

def dice_loss(pred,target,valid_mask,smooth=1,exponent=2,class_weight=None,ignore_index=255):
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            dice_loss = binary_dice_loss(
                pred[:, i],
                target[..., i],
                valid_mask=valid_mask,
                smooth=smooth,
                exponent=exponent)
            if class_weight is not None:
                dice_loss *= class_weight[i]
            total_loss += dice_loss
    return total_loss / num_classes

def binary_dice_loss(pred, target, valid_mask, smooth=1, exponent=2, **kwargs):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth

    return 1 - num / den

class offical_DiceLoss(nn.Module):

    def __init__(self,smooth=1,exponent=2,reduction='mean',class_weight=None,loss_weight=1.0,
                 ignore_index=255,loss_name='loss_dice',**kwargs):
        super(offical_DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.class_weight = None
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,pred,target,avg_factor=None,reduction_override=None,**kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(torch.clamp(target.long(), 0, num_classes - 1),num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()

        loss = self.loss_weight * dice_loss(pred,one_hot_target,valid_mask=valid_mask,smooth=self.smooth,exponent=self.exponent,class_weight=class_weight,ignore_index=self.ignore_index)
        return loss.sum()
































