import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils import TVD, topK_overlap_loss


class PGDAttacker:
    def __init__(self, radius, steps, step_size, random_start, norm_type, ascending=True, device='cpu'):
        self.radius = radius  # attack radius
        self.steps = steps  # how many step to conduct pgd
        self.step_size = step_size  # coefficient of PGD
        self.random_start = random_start
        self.norm_type = norm_type  # which norm of your noise
        self.ascending = ascending  # perform gradient ascending, i.e, to maximum the loss
        self.device = device

    def _clip_(self, adv_x, x):
        adv_x -= x
        if self.norm_type == 'l-infty':
            adv_x.clamp_(-self.radius, self.radius)
        else:
            if self.norm_type == 'l2':
                norm = (adv_x.reshape(adv_x.shape[0], -1) ** 2).sum(dim=1).sqrt()
            elif self.norm_type == 'l1':
                norm = adv_x.reshape(adv_x.shape[0], -1).abs().sum(dim=1)
            norm = norm.reshape(-1, *([1] * (len(x.shape) - 1)))
            adv_x /= (norm + 1e-10)
            adv_x *= norm.clamp(max=self.radius)
        adv_x += x
        # adv_x.clamp_(0, 1)  # 原地限制范围，将超过指定范围的值裁剪到范围内

    def perturb_delta(self, feats, mask, y, graph, target_model):  # TVD loss
        adv_feats = feats.clone()

        ratio = 0.05
        binary_matrix = np.random.choice([0, 1], size=feats.shape, p=[1 - ratio / 100, ratio / 100])
        mask_ = binary_matrix.astype(bool)  # 随机生成一个(2708, 1433)形状的二进制矩阵，每个元素都是 0 或 1

        if self.norm_type == 'l-infty':
            adv_feats += 2 * (torch.rand_like(feats) - 0.5) * self.radius * mask_  # [-radius, radius]
        else:
            adv_feats += 2 * (torch.rand_like(feats) - 0.5) * self.radius / self.steps * mask_
        self._clip_(adv_feats, feats)

        # temporarily shutdown autograd of model to improve PGD efficiency
        for pp in target_model.parameters():
            pp.requires_grad = False

        for step in range(self.steps):
            adv_feats.requires_grad_()
            y_delta, _, _ = target_model(graph, adv_feats)
            loss = TVD(y, y_delta)
            grad = torch.autograd.grad(loss, [adv_feats])[0]

            with torch.no_grad():
                if not self.ascending:
                    grad.mul_(-1)

                if self.norm_type == 'l-infty':
                    adv_feats.add_(torch.sign(grad), alpha=self.step_size)
                else:
                    if self.norm_type == 'l2':
                        grad_norm = (grad.reshape(grad.shape[0], -1) ** 2).sum(dim=1).sqrt()
                    elif self.norm_type == 'l1':
                        grad_norm = grad.reshape(grad.shape[0], -1).abs().sum(dim=1)
                    grad_norm = grad_norm.reshape(-1, *([1] * (len(feats.shape) - 1)))
                    scaled_grad = grad / (grad_norm + 1e-10)
                    adv_feats.add_(scaled_grad, alpha=self.step_size)  # 原地加

                self._clip_(adv_feats, feats)

        # reopen autograd of model after PGD
        for pp in target_model.parameters():
            pp.requires_grad = True

        return adv_feats

    def perturb_rho(self, feats, orig_att, graph, target_model):  # top-K loss
        adv_feats = feats.clone()

        ratio = 0.05
        binary_matrix = np.random.choice([0, 1], size=feats.shape, p=[1 - ratio / 100, ratio / 100])
        mask_ = binary_matrix.astype(bool)  # 随机生成一个(2708, 1433)形状的二进制矩阵，每个元素都是 0 或 1

        if self.norm_type == 'l-infty':
            adv_feats += 2 * (torch.rand_like(feats) - 0.5) * self.radius * mask_  # [-radius, radius]
        else:
            adv_feats += 2 * (torch.rand_like(feats) - 0.5) * self.radius / self.steps * mask_
        self._clip_(adv_feats, feats)

        # temporarily shutdown autograd of model to improve PGD efficiency
        for pp in target_model.parameters():
            pp.requires_grad = False

        for step in range(self.steps):
            adv_feats.requires_grad_()
            _, _, att = target_model(graph, adv_feats)
            loss = 0
            for i in range(orig_att.shape[1]):
                loss += topK_overlap_loss(att[:, i], orig_att[:, i], graph)
            grad = torch.autograd.grad(loss, [adv_feats])[0]

            with torch.no_grad():
                if not self.ascending:
                    grad.mul_(-1)

                if self.norm_type == 'l-infty':
                    adv_feats.add_(torch.sign(grad), alpha=self.step_size)
                else:
                    if self.norm_type == 'l2':
                        grad_norm = (grad.reshape(grad.shape[0], -1) ** 2).sum(dim=1).sqrt()
                    elif self.norm_type == 'l1':
                        grad_norm = grad.reshape(grad.shape[0], -1).abs().sum(dim=1)
                    grad_norm = grad_norm.reshape(-1, *([1] * (len(feats.shape) - 1)))
                    scaled_grad = grad / (grad_norm + 1e-10)
                    adv_feats.add_(scaled_grad, alpha=self.step_size)  # 原地加

                self._clip_(adv_feats, feats)

        # reopen autograd of model after PGD
        for pp in target_model.parameters():
            pp.requires_grad = True

        return adv_feats
