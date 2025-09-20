import argparse
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.nn import avg_pool_neighbor_x
from torch_geometric.utils import degree, train_test_split_edges
# ↓↓↓ 新增：导入 PyG 相关类（解决 NameError）↓↓↓
from torch_geometric.data import Data  # 导入 Data 类
from torch_geometric.utils import degree, train_test_split_edges  # 确保已有这行（之前可能漏了）
# ↑↑↑ 新增导入 ↑↑↑
# lib.py 顶部导入（已有则跳过）
import numpy as np
from torch_geometric.utils import remove_self_loops, to_undirected


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def eval_edge_pred(adj_pred, val_edges, edge_labels):
    logits = adj_pred[val_edges]
    logits = np.nan_to_num(logits)
    roc_auc = roc_auc_score(edge_labels, logits)
    ap_score = average_precision_score(edge_labels, logits)
    return roc_auc, ap_score


def eval_node_cls(nc_logits, labels, binary=False):  # 新增binary参数
    """ evaluate node classification results """
    if binary:  # 二分类逻辑（combined数据集用）
        # 对输出用sigmoid激活，大于0.5视为正例
        preds = torch.sigmoid(nc_logits[:, 1]) > 0.5  # 取第二类作为正例
        correct = torch.sum(preds == labels)
        f_measure = correct.item() / len(labels)  # 计算准确率
    elif len(labels.size()) == 2:  # 原多标签逻辑（保留）
        preds = torch.round(torch.sigmoid(nc_logits))
        tp = len(torch.nonzero(preds * labels))
        tn = len(torch.nonzero((1 - preds) * (1 - labels)))
        fp = len(torch.nonzero(preds * (1 - labels)))
        fn = len(torch.nonzero((1 - preds) * labels))
        pre, rec, f1 = 0., 0., 0.
        if tp + fp > 0:
            pre = tp / (tp + fp)
        if tp + fn > 0:
            rec = tp / (tp + fn)
        if pre + rec > 0:
            f_measure = (2 * pre * rec) / (pre + rec)
    else:  # 原多分类逻辑（保留）
        preds = torch.argmax(nc_logits, dim=1)
        correct = torch.sum(preds == labels)
        f_measure = correct.item() / len(labels)
    return f_measure


def loss_fn(hidden1, summary1):
    r"""Computes the margin objective."""
    marginloss_fn = nn.MarginRankingLoss(0.5, reduction='none')
    shuf_index = torch.randperm(summary1.size(0))

    hidden2 = hidden1[shuf_index]
    summary2 = summary1[shuf_index]

    logits_aa = torch.sigmoid(torch.sum(hidden1 * summary1, dim=-1))
    logits_bb = torch.sigmoid(torch.sum(hidden2 * summary2, dim=-1))
    logits_ab = torch.sigmoid(torch.sum(hidden1 * summary2, dim=-1))
    logits_ba = torch.sigmoid(torch.sum(hidden2 * summary1, dim=-1))
    TotalLoss = 0.0
    ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)
    TotalLoss += marginloss_fn(logits_aa, logits_ba, ones)
    TotalLoss += marginloss_fn(logits_bb, logits_ab, ones)

    return TotalLoss


def loss_fn(hidden1, data):
    r"""Computes the margin objective."""
    datac = copy.copy(data)
    datac.x = hidden1
    summary1 = avg_pool_neighbor_x(datac).x

    marginloss_fn = nn.MarginRankingLoss(0.5, reduction='none')
    shuf_index = torch.randperm(summary1.size(0))

    hidden2 = hidden1[shuf_index]
    summary2 = summary1[shuf_index]

    logits_aa = torch.sigmoid(torch.sum(hidden1 * summary1, dim=-1))
    logits_bb = torch.sigmoid(torch.sum(hidden2 * summary2, dim=-1))
    logits_ab = torch.sigmoid(torch.sum(hidden1 * summary2, dim=-1))
    logits_ba = torch.sigmoid(torch.sum(hidden2 * summary1, dim=-1))
    TotalLoss = 0.0
    ones = torch.ones(logits_aa.size(0)).cuda(logits_aa.device)
    TotalLoss += marginloss_fn(logits_aa, logits_ba, ones)
    TotalLoss += marginloss_fn(logits_bb, logits_ab, ones)

    return TotalLoss


def loss_cal(x, x_aug):
    T = 0.2
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.clamp(torch.einsum('i,j->ij', x_abs, x_aug_abs), min=1e-5)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()

    return loss


def triplet_margin_loss(anchor, positive, negative, w=0.5, margin=1.0, p=2, eps=1e-06, swap=False, reduction='mean'):
    # 计算两两之间的距离
    d_ap = torch.norm(anchor - positive, p=p, dim=-1)  # (N,)
    d_an = torch.norm(anchor - negative, p=p, dim=-1)  # (N,)
    d_pn = torch.norm(positive - negative, p=p, dim=-1)  # (N,)
    # 如果swap为True，交换d_an和d_pn中的最小值
    if swap:
        d_an = torch.min(d_an, d_pn)
    # 计算损失函数
    loss = torch.clamp(d_ap - w * d_an + margin, min=0.0)  # (N,)
    # 根据reduction参数返回结果x
    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    else:
        raise ValueError('Invalid reduction: {}'.format(reduction))


def loss_cal2(x, x_aug, temperature=0.2, sym=True):
    # x and x_aug shape -> Batch x proj_hidden_dim

    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / temperature)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    if sym:
        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

        loss_0 = - torch.log(loss_0).mean()
        loss_1 = - torch.log(loss_1).mean()
        loss = (loss_0 + loss_1) / 2.0
    else:
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss_1 = - torch.log(loss_1).mean()
        return loss_1

    return loss


def loss_com(summary, label):
    label = torch.tensor(label).to(device=summary.device)
    x = summary[label == 0]
    # 选择属于第二类的tensor
    x_aug = summary[label == 1]
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)
    T = 0.2
    batch_size, _ = x.size()
    sim_matrix = torch.einsum('ik,jk->ij', x, x) / torch.clamp(torch.einsum('i,j->ij', x_abs, x_abs), min=1e-5)
    sim_matrix_neg = torch.einsum('ik,jk->ij', x, x_aug) / torch.clamp(torch.einsum('i,j->ij', x_abs, x_aug_abs),
                                                                       min=1e-5)
    sim_matrix = torch.exp(sim_matrix / T)
    sim_matrix_neg = torch.exp(sim_matrix_neg / T)
    # pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss_pos = sim_matrix.mean(dim=1) / (sim_matrix_neg.sum(dim=1))
    loss_pos = - torch.log(loss_pos).mean()

    batch_size, _ = x_aug.size()
    sim_matrix = torch.einsum('ik,jk->ij', x_aug, x_aug) / torch.clamp(torch.einsum('i,j->ij', x_aug_abs, x_aug_abs),
                                                                       min=1e-5)
    sim_matrix_neg = torch.einsum('ik,jk->ij', x_aug, x) / torch.clamp(torch.einsum('i,j->ij', x_aug_abs, x_abs),
                                                                       min=1e-5)
    sim_matrix = torch.exp(sim_matrix / T)
    sim_matrix_neg = torch.exp(sim_matrix_neg / T)
    # pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss_neg = sim_matrix.mean(dim=1) / (sim_matrix_neg.sum(dim=1))
    loss_neg = - torch.log(loss_neg).mean()

    loss = loss_pos + loss_neg

    return loss


def loss_compare(summary, label):
    # datac = copy.copy(data)
    # datac.x = summary
    # summary = avg_pool_neighbor_x(datac).x
    # 选择属于第一类的tensor
    label = torch.tensor(label).to(device=summary.device)
    # x1 = torch.masked_select(summary, label == 0).reshape(-1, 128)
    x1 = summary[label == 0].mean(dim=-1).unsqueeze(-1)
    # 选择属于第二类的tensor
    x2 = summary[label == 1].mean(dim=-1).unsqueeze(-1)

    pos_p = torch.exp(torch.mm(x1, x1.transpose(0, 1))).mean(dim=-1)
    pos_n = torch.exp(torch.mm(x1, x2.transpose(0, 1))).mean(dim=-1)
    loss_pos = torch.log(pos_p / pos_n).mean()
    neg_p = torch.exp(torch.mm(x2, x2.transpose(0, 1))).mean(dim=-1)
    neg_n = torch.exp(torch.mm(x2, x1.transpose(0, 1))).mean(dim=-1)
    loss_neg = torch.log(neg_p / neg_n).mean()
    loss = (loss_pos + loss_neg) * label.size(0)
    # mse = F.mse_loss
    # tr_loss = F.triplet_margin_loss
    # sff_index_a = torch.randperm(x1.size(0)).to(device=summary.device)
    # sff_index_b = torch.randperm(x2.size(0)).to(device=summary.device)
    #
    # sff_index_c = torch.randint(0, x1.size(0), (x2.size(0),)).to(device=summary.device)
    # sff_index_d = torch.randint(0, x2.size(0), (x1.size(0),)).to(device=summary.device)
    #
    # y1 = x1[sff_index_a]
    # y2 = x2[sff_index_b]
    #
    # z1 = x1[sff_index_c]
    # z2 = x2[sff_index_d]
    # x_pos = torch.cat((y1, y2), dim=0)
    # x_neg = torch.cat((z2, z1), dim=0)
    # pdist = nn.PairwiseDistance(p=2)
    # sim_pos = torch.exp(pdist(x_new, x_pos))
    # sim_neg = torch.exp(pdist(x_new, x_neg))
    # loss = torch.log(sim_pos/sim_neg).mean()
    # loss = tr_loss(x_new, x_pos, x_neg, reduction='mean')
    # loss = mse(x_new, x_pos)
    return loss


def loss_means(summary, label_all, num_classes):
    r"""Computes the margin objective."""
    std = 0
    for i in range(num_classes):
        node = summary[label_all == i]
        if node.shape[0] != 0:
            std += node.std()
    return std


def get_lr_schedule_by_sigmoid(n_epochs, lr, warmup):
    """ schedule the learning rate with the sigmoid function.
    The learning rate will start with near zero and end with near lr """
    factors = torch.FloatTensor(np.arange(n_epochs))
    factors = ((factors / factors[-1]) * (warmup * 2)) - warmup
    factors = torch.sigmoid(factors)
    # range the factors to [0, 1]
    factors = (factors - factors[0]) / (factors[-1] - factors[0])
    lr_schedule = factors * lr
    return lr_schedule


def compute_loss_para(adj):
    pos_weight = (adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = (
            adj.shape[0]
            * adj.shape[0]
            / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    )
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm

###新加
# ↓↓↓ 手动负边采样辅助函数（必须放在get_ep_data之前）↓↓↓
def _generate_negative_edges(pos_edges, num_nodes, num_neg_needed):
    pos_edge_set = set()
    for u, v in pos_edges.T.numpy():
        pos_edge_set.add(tuple(sorted((u, v))))
    neg_edges = []
    while len(neg_edges) < num_neg_needed:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        if u == v:
            continue
        neg_edge = tuple(sorted((u, v)))
        if neg_edge not in pos_edge_set:
            neg_edges.append([u, v])
    return torch.tensor(neg_edges, dtype=torch.long).T

####大改
# 在 lib.py 中修改 get_ep_data 函数
def get_ep_data(data, args):
    new_label = None
    adj_m = None
    norm_w = None
    pos_weight = None
    train_edge = None

    if args.task == 1:  # 链接预测任务
        # 关键修复：检查 edge_index 是否存在
        if data.edge_index is None:
            raise ValueError("data.edge_index 为 None，无法生成邻接矩阵")

        # 生成训练/验证/测试集边（确保返回值正确）
        train_edge = train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1)

        # 提取训练集正边（用于生成负边）
        train_pos_edges = train_edge.train_pos_edge_index
        if train_pos_edges is None:
            raise ValueError("train_test_split_edges 未生成 train_pos_edge_index")

        # 生成负边（若不存在）
        if train_edge.train_neg_edge_index is None:
            num_train_pos = train_pos_edges.size(1)
            train_edge.train_neg_edge_index = generate_neg_edges(train_pos_edges, data.num_nodes, num_train_pos)
        if train_edge.val_neg_edge_index is None:
            num_val_pos = train_edge.val_pos_edge_index.size(1)
            train_edge.val_neg_edge_index = generate_neg_edges(train_pos_edges, data.num_nodes, num_val_pos)
        if train_edge.test_neg_edge_index is None:
            num_test_pos = train_edge.test_pos_edge_index.size(1)
            train_edge.test_neg_edge_index = generate_neg_edges(train_pos_edges, data.num_nodes, num_test_pos)

        # 构建邻接矩阵（修复 edge_index 可能为 None 的问题）
        adj_m = torch.sparse_coo_tensor(
            data.edge_index,  # 确保此处 edge_index 不为空
            torch.ones(data.edge_index.size(1)),  # 错误发生位置，添加前置检查
            (data.num_nodes, data.num_nodes)
        ).to_dense()

        # 计算正负样本权重
        pos_weight = (adj_m.numel() - adj_m.sum()) / adj_m.sum()
        norm_w = adj_m.numel() / (2 * (adj_m.numel() - adj_m.sum()))

    return new_label, adj_m, norm_w, pos_weight, train_edge

# 新增：负边生成函数（确保负边不与正边重复）
def generate_neg_edges(pos_edges, num_nodes, num_neg):
    # 新增：参数校验
    if pos_edges is None:
        raise ValueError("pos_edges不能为None，请检查train_pos_edge_index是否存在")
    if not isinstance(pos_edges, torch.Tensor):
        raise TypeError(f"pos_edges必须是torch.Tensor，实际为{type(pos_edges)}")
    if pos_edges.dim() != 2 or pos_edges.size(0) != 2:
        raise ValueError(f"pos_edges形状应为[2, 边数]，实际为{pos_edges.shape}")

    # 正边转为集合便于快速查询（去重+无向化）
    pos_set = set()
    for i in range(pos_edges.size(1)):
        u, v = pos_edges[0, i].item(), pos_edges[1, i].item()
        if u > v:
            u, v = v, u  # 确保u <= v，统一无向边存储格式
        pos_set.add((u, v))

    neg_edges = []
    while len(neg_edges) < num_neg:
        # 随机生成节点对（排除自环）
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        if (u, v) not in pos_set:
            neg_edges.append((u, v))
            pos_set.add((u, v))  # 避免重复负边

    return torch.tensor(neg_edges).t().long()  # 转为[2, num_neg]的形状