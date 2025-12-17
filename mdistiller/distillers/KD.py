import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def get_w(logits_student, logits_teacher, H_ub, t=3.0):
    _p_t = F.softmax(logits_teacher / t, dim=1)
    H_t = -torch.sum(_p_t * torch.log(_p_t + 1e-10), dim=1) 
    _p_s = F.softmax(logits_student / t, dim=1)
    H_s = -torch.sum(_p_s * torch.log(_p_s + 1e-10), dim=1)
    
    w = (H_t + (H_t * H_s / H_ub)) / 2
    return w

def kd_loss(logits_student, logits_teacher, temperature, EA=False, t=3.0, H_ub=None):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd *= temperature ** 2
    if EA:
        w_EA = get_w(logits_student, logits_teacher, H_ub, t).detach()
        return (loss_kd * w_EA).mean()
    else:
        return loss_kd.mean()


class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.EA = cfg.DISTILLER.EA
        self.t = cfg.DISTILLER.t
        
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
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, self.temperature, self.EA, self.t, self.H_ub 
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
