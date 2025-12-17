import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def get_w(logits_student, logits_teacher, H_ub, t=3.0):
    _p_t = F.softmax(logits_teacher / t, dim=1)
    H_t = -torch.sum(_p_t * torch.log(torch.clamp(_p_t, 1e-10)), dim=1) 
    _p_s = F.softmax(logits_student / t, dim=1)
    H_s = -torch.sum(_p_s * torch.log(torch.clamp(_p_s, min=1e-10)), dim=1)
    
    w = (H_t + (H_t * H_s / H_ub)) / 2
    return w

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature, EA=True, H_ub=None, LS=False):
    logits_student = normalize(logits_student) if LS else logits_student
    logits_teacher = normalize(logits_teacher) if LS else logits_teacher
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction='none') 
        * (temperature ** 2)
    ).sum(dim = 1)
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='none') 
        * (temperature ** 2)
    ).sum(dim = 1)
    if EA:
        w_EA = get_w(logits_student, logits_teacher, H_ub).detach()
        return ((alpha * tckd_loss + beta * nckd_loss) * w_EA).mean()
    else:
        return (alpha * tckd_loss + beta * nckd_loss).mean()


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class DKD(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(DKD, self).__init__(student, teacher)
        self.EA = cfg.DISTILLER.EA
        self.t = cfg.DISTILLER.t
        self.LS = cfg.DISTILLER.LS
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        if self.LS:
            self.alpha = cfg.DKD.ALPHA * cfg.KD.LOSS.KD_WEIGHT
            self.beta = cfg.DKD.BETA * cfg.KD.LOSS.KD_WEIGHT
        else:
            self.alpha = cfg.DKD.ALPHA
            self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP
        
        student_device = next(student.parameters()).device
        if cfg.DATASET.TYPE == 'imagenet':
            self.H_ub = torch.log(torch.tensor(1000.0, dtype=torch.float32, device=student_device)) 
        else:
            self.H_ub = torch.tensor(100.0, dtype=torch.float32, device=student_device)

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
            self.EA,
            self.H_ub,
            self.LS
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict
