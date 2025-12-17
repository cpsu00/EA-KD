from termios import CEOL
from turtle import st
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ._base import Distiller
from .loss import CrossEntropyLabelSmooth

def get_w(logits_student, logits_teacher, H_ub, t=3.0):
    _p_t = F.softmax(logits_teacher / t, dim=1)
    H_t = -torch.sum(_p_t * torch.log(_p_t + 1e-10), dim=1) 
    _p_s = F.softmax(logits_student / t, dim=1)
    H_s = -torch.sum(_p_s * torch.log(_p_s + 1e-10), dim=1)
    
    w = (H_t + (H_t * H_s / H_ub)) / 2
    return w

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand=False, EA=0, H_ub=None):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd *= temperature**2
    if EA:
        w_EA = get_w(logits_student_in, logits_teacher_in, H_ub).detach()
        return (loss_kd * w_EA).mean()
    else:
        return loss_kd.mean()
    

def cc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num
    return consistency_loss


def bc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / batch_size
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / batch_size
    return consistency_loss


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_data_conf(x, y, lam, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = lam.reshape(-1,1,1,1)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class KD_ours(Distiller):
    def __init__(self, student, teacher, cfg):
        super(KD_ours, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.logit_stand = cfg.DISTILLER.LS
        self.EA = cfg.DISTILLER.EA
        self.H_ub = torch.log(torch.tensor(100.0, dtype=torch.float32)).cuda()

    def forward_train(self, image_weak, image_strong, target, **kwargs):
        logits_student_weak, _ = self.student(image_weak)
        logits_student_strong, _ = self.student(image_strong)
        with torch.no_grad():
            logits_teacher_weak, _ = self.teacher(image_weak)
            logits_teacher_strong, _ = self.teacher(image_strong)

        batch_size, class_num = logits_student_strong.shape

        pred_teacher_weak = F.softmax(logits_teacher_weak.detach(), dim=1)
        confidence, pseudo_labels = pred_teacher_weak.max(dim=1)
        confidence = confidence.detach()
        conf_thresh = np.percentile(
            confidence.cpu().numpy().flatten(), 50
        )
        mask = confidence.le(conf_thresh).bool()

        class_confidence = torch.sum(pred_teacher_weak, dim=0)
        class_confidence = class_confidence.detach()
        class_confidence_thresh = np.percentile(
            class_confidence.cpu().numpy().flatten(), 50
        )
        class_conf_mask = class_confidence.le(class_confidence_thresh).bool()

        # losses
        loss_ce = self.ce_loss_weight * (F.cross_entropy(logits_student_weak, target) + F.cross_entropy(logits_student_strong, target))
        loss_kd_weak = self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.temperature,
            logit_stand=self.logit_stand,
            EA=self.EA,
            H_ub=self.H_ub
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            3.0,
            logit_stand=self.logit_stand,
            EA=self.EA,
            H_ub=self.H_ub
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            5.0,
            logit_stand=self.logit_stand,
            EA=self.EA,
            H_ub=self.H_ub
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
            logit_stand=self.logit_stand,
            EA=self.EA,
            H_ub=self.H_ub
        ) * mask).mean()) + self.kd_loss_weight * ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
            logit_stand=self.logit_stand,
            EA=self.EA,
            H_ub=self.H_ub
        ) * mask).mean())

        loss_kd_strong = self.kd_loss_weight * kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            self.temperature,
            logit_stand=self.logit_stand,
            EA=self.EA,
            H_ub=self.H_ub
        ) + self.kd_loss_weight * kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            3.0,
            logit_stand=self.logit_stand,
            EA=self.EA,
            H_ub=self.H_ub
        ) + self.kd_loss_weight * kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            5.0,
            logit_stand=self.logit_stand,
            EA=self.EA,
            H_ub=self.H_ub
        ) + self.kd_loss_weight * kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
            logit_stand=self.logit_stand,
            EA=self.EA,
            H_ub=self.H_ub
        ) + self.kd_loss_weight * kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
            logit_stand=self.logit_stand,
            EA=self.EA,
            H_ub=self.H_ub
        )

        loss_cc_weak = self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.temperature,
            # reduce=False
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            3.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            5.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
        ) * class_conf_mask).mean())
        loss_cc_strong = self.kd_loss_weight * cc_loss(
            logits_student_strong,
            logits_teacher_strong,
            self.temperature,
        ) + self.kd_loss_weight * cc_loss(
            logits_student_strong,
            logits_teacher_strong,
            3.0,
        ) + self.kd_loss_weight * cc_loss(
            logits_student_strong,
            logits_teacher_strong,
            5.0,
        ) + self.kd_loss_weight * cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
        ) + self.kd_loss_weight * cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
        )
        loss_bc_weak = self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.temperature,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            3.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            5.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
        ) * mask).mean())
        loss_bc_strong = self.kd_loss_weight * ((bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            self.temperature,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            3.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            5.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            2.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            6.0,
        ) * mask).mean())
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd_weak + loss_kd_strong,
            "loss_cc": loss_cc_weak,
            "loss_bc": loss_bc_weak
        }
        return logits_student_weak, losses_dict

