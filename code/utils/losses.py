import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss


def kl_loss(inputs, targets, ep=1e-8):
    kl_loss=nn.KLDivLoss(reduction='mean')
    consist_loss = kl_loss(torch.log(inputs+ep), targets)
    return consist_loss


def soft_ce_loss(inputs, target, ep=1e-8):
    logprobs = torch.log(inputs+ep)
    return  torch.mean(-(target[:,0,...]*logprobs[:,0,...]+target[:,1,...]*logprobs[:,1,...]))


def mse_loss(input1, input2):
    return torch.mean((input1 - input2)**2)


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersection = torch.sum(score * target)
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_correlation(gmm_segments1, gmm_segments2):
    num_clusters = gmm_segments1.shape[0]
    correlations = []

    for cluster_id in range(num_clusters):
        segment1 = gmm_segments1[cluster_id].flatten()
        segment2 = gmm_segments2[cluster_id].flatten()
        mean1 = torch.mean(segment1)
        mean2 = torch.mean(segment2)
        cov = torch.sum((segment1 - mean1) * (segment2 - mean2))
        std1 = torch.sqrt(torch.sum((segment1 - mean1) ** 2))
        std2 = torch.sqrt(torch.sum((segment2 - mean2) ** 2))
        correlation = cov / (std1 * std2 + 1e-8)
        correlations.append(correlation)
    correlation_mean = torch.stack(correlations).mean()

    return correlation_mean


def generate_gmm_segments(num_clusters=4, height=256, width=256):
    gmm_segments = np.zeros((num_clusters, height, width))

    for cluster_id in range(num_clusters):
        mean = np.random.rand() * 0.5 + 0.25
        std_dev = np.random.rand() * 0.1 + 0.05
        gmm_segments[cluster_id] = np.clip(np.random.normal(mean, std_dev, (height, width)), 0, 1)
    gmm_segments /= np.sum(gmm_segments, axis=0, keepdims=True)

    return gmm_segments


def consistency_loss(pseudo_labels_branch1, pseudo_labels_branch2):
    # print("output1:", output1.shape)
    # print("output2:", output2.shape)
    # print("pseudo_labels:", pseudo_labels_branch1.shape)
    num_classes = pseudo_labels_branch1.shape[1]
    pseudo_labels_branch1 = pseudo_labels_branch1.to('cuda')
    pseudo_labels_branch2 = pseudo_labels_branch2.to('cuda')
    pseudo_labels_branch1 = pseudo_labels_branch1.unsqueeze(0)
    pseudo_labels_branch2 = pseudo_labels_branch2.unsqueeze(0)
    pseudo_labels_branch1_one_hot = F.one_hot(pseudo_labels_branch1.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    pseudo_labels_branch1_probs = F.softmax(pseudo_labels_branch1_one_hot, dim=1)
    pseudo_labels_branch2_one_hot = F.one_hot(pseudo_labels_branch2.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    pseudo_labels_branch2_probs = F.softmax(pseudo_labels_branch2_one_hot, dim=1)
    ce_loss_fn = nn.CrossEntropyLoss()
    ce_loss_branch1 = ce_loss_fn(pseudo_labels_branch1_probs, pseudo_labels_branch2.long())
    ce_loss_branch2 = ce_loss_fn(pseudo_labels_branch2_probs, pseudo_labels_branch1.long())
    total_loss = ce_loss_branch1 + ce_loss_branch2
    return total_loss


if __name__ == '__main__':
    gmm_segments1 = generate_gmm_segments()
    # gmm_segments2 = generate_gmm_segments()
    #
    # # print("gmm_segments1 shape:", gmm_segments1.shape)
    # # print("gmm_segments2 shape:", gmm_segments2.shape)
    #
    # correlations = calculate_correlation(gmm_segments1, gmm_segments2)
    # print(correlations)
    #
    # pseudo_labels = generate_pseudo_labels(gmm_segments1)
    #
    # # 假设 output1 和 output2 是两个分支的预测输出
    # output1 = torch.rand((4, 256, 256), requires_grad=True)  # 模拟分支 1 的输出
    # output2 = torch.rand((4, 256, 256), requires_grad=True)  # 模拟分支 2 的输出
    #
    # # 计算一致性正则化损失
    # loss = consistency_loss(output1, output2, pseudo_labels)
    # print("Consistency Regularization Loss:", loss.item())