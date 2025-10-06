import copy
import gc
import os
import pickle

import scipy.sparse as sp
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import statistics
import argparse

import optuna
from scipy.sparse import csr_matrix
from torch.utils.hipify.hipify_python import str2bool
from torch_geometric.utils import train_test_split_edges, degree
from torch_geometric.data import Data
from lib import eval_node_cls, setup_seed, loss_means, get_lr_schedule_by_sigmoid, get_ep_data, eval_edge_pred, loss_fn, \
    loss_compare, loss_cal, loss_cal2, loss_com, triplet_margin_loss
from models_ACGA import GCN_Net
import torch
import numpy as np
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from spliter import louvain, random_spliter, origin, rand_walk

import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_parser():
    parser = argparse.ArgumentParser(description='Description: Script to run our model.')
    parser.add_argument('--dataset', help='Cora, Citeseer or Pubmed. Default=Cora', default='Cora')
    parser.add_argument('--hidden_size', type=int, help='hidden size', default=128)
    parser.add_argument('--emb_size', type=int, help='gae/vgae embedding size', default=32)
    parser.add_argument('--spliter', help='spliter method.Default=random_spliter (rand_walk)', default="random_spliter")
    parser.add_argument('--gae', type=str2bool, help='whether use GAE ', default=True)
    parser.add_argument('--use_bns', type=str2bool, help='whether use bns for GNN layer ', default=True)
    parser.add_argument('--task', type=int, help='node cls = 0, edge predict = 1', default=0)
    parser.add_argument('--alpha', type=float, default=0.4)  # 0.32
    parser.add_argument('--beta', type=float, default=0.7)  # 5.85
    parser.add_argument('--gamma', type=float, default=2.85)  # 1.9
    # 补充 subgraph_size 参数（关键修改）
    parser.add_argument('--subgraph_size', type=int, default=200,  # 类型为整数，默认值64（可根据数据调整）
                        help='Size of subgraph used in training (adjust based on your dataset)')
    return parser


def train_rep(model, data, num_classes, alpha=0.5, beta=3.0, gamma=2.0, train_edge=None, new_label=None):
    # train_rep函数用于训练模型的学习部分，通过多轮迭代优化模型参数
    # model为待训练的ACGA模型，train_rep为训练加载器(批次数据)，optimizer 为优化器
    # criterion为损失函数，epochs为训练轮数
    model.train()  # 将模型设为训练模式（启用dropout、批归一化更新什么的）
    batch = data.to(device)
    if isinstance(new_label, np.ndarray):
        label_all = new_label
    else:
        label_all = batch.y
    # ############### need delet
    # label_all = batch.y
    # ############### need delet
    # alpha,beta,gamma,epoch,acc
    alpha = alpha  # MAX: Cora: 0.5,3,2,300,83.6,    0.58,  0.65,  3.4000,
    beta = beta  # 2
    gamma = gamma  # 2
    if train_edge is not None:
        adj = train_edge.train_pos_edge_index
        summary, summary_pos, summary_neg, loss_co = model.train_present(batch, label_all, adj)
    else:
        summary, summary_pos, summary_neg, loss_co = model.train_present(batch, label=label_all)
    loss_s = loss_means(summary, label_all, num_classes)
    loss_cl = torch.nn.functional.triplet_margin_loss(summary, summary_pos, summary_neg, reduction='mean')
    loss = beta * ((1 - alpha) * loss_cl + alpha * loss_co) + loss_s * gamma
    return loss


def train_cls(model, data, args, criterion, optimizer, epoch):
    model.train()
    batch = data.to(device)
    label_all = batch.y
    predict = model(batch)
    # predict = model.forward_emb(batch, embedding)
    if data.x.size(0) == args.subgraph_size:
        loss_nc = criterion(predict, label_all)
    else:
        predict = predict[batch.train_mask]
        label = batch.y[batch.train_mask]
        loss_nc = criterion(predict, label)
    # if epoch % 20 == 0:
    # print(f'epoch: {epoch} loss_nc:{loss_nc:.4f}')
    # loss_nc = loss_nc
    # loss_nc.backward()
    # optimizer.step()

    # return loss_nc.item()
    return loss_nc


def test_cls(model, data):
    r"""Evaluates latent space quality via a logistic regression downstream task."""
    model.eval()
    # criterion = F.CrossEntropyLoss()
    batch = data
    predict = model(batch)
    # predict = model.test(batch)
    gcn_val_z = predict[batch.val_mask]
    gcn_test_z = predict[batch.test_mask]
    val_y = batch.y[batch.val_mask]
    test_y = batch.y[batch.test_mask]
    gcn_val_acc = eval_node_cls(gcn_val_z, val_y)
    gcn_test_acc = eval_node_cls(gcn_test_z, test_y)
    return gcn_val_acc, gcn_test_acc


def train_ep(model, data, train_edge, adj_m, norm_w, pos_weight, optimizer, args, wight):
    model.train()
    kl_divergence = 0
    batch = data.to(device)
    adj = train_edge.train_pos_edge_index
    adj_logit = model(batch, edge=adj)
    loss_nc = norm_w * F.binary_cross_entropy_with_logits(adj_logit.view(-1), adj_m.view(-1), pos_weight=pos_weight)
    # if not args.gae:
    #     mu = model.ep.mean
    #     logit = model.ep.logstd
    #     kl_divergence = 0.5 * (1 + 2 * logit - mu ** 2 - torch.exp(2 * logit)).sum(1).mean()
    loss_nc = loss_nc - kl_divergence  # 恢复KL项
    # loss_nc.backward()
    # optimizer.step()

    return loss_nc


def test_ep(model, data, train_edge):
    model.eval()
    adj = train_edge.train_pos_edge_index
    adj_logit = model(data, adj)

    # 新增：打印预测值的标准差（衡量波动程度）
    print(f"模型预测值标准差：{adj_logit.std().item():.4f}")  # 标准差应>0

    val_edges = torch.cat((train_edge.val_pos_edge_index, train_edge.val_neg_edge_index), axis=1).cpu().numpy()
    val_edge_labels = np.concatenate(
        [np.ones(train_edge.val_pos_edge_index.size(1)), np.zeros(train_edge.val_neg_edge_index.size(1))])



    test_edges = torch.cat((train_edge.test_pos_edge_index, train_edge.test_neg_edge_index), axis=1).cpu().numpy()
    test_edge_labels = np.concatenate(
        [np.ones(train_edge.test_pos_edge_index.size(1)), np.zeros(train_edge.test_neg_edge_index.size(1))])

    adj_pred = adj_logit.cpu()
    ep_auc, ep_ap = eval_edge_pred(adj_pred, val_edges, val_edge_labels)
    # print(f'EPNet train: auc {ep_auc:.4f}, ap {ep_ap:.4f}')

    ep_auc_test, ep_ap_test = eval_edge_pred(adj_pred, test_edges, test_edge_labels)
    # print(f'EPNet train,Final: auc {ep_auc_test:.4f}, ap {ep_ap:.4f}')

    return ep_auc, ep_ap, ep_auc_test, ep_ap_test


spliter_dict = {
    'louvain': louvain,
    'random_spliter': random_spliter,
    'origin': origin,
    'rand_walk': rand_walk,
    'nothing': lambda x: x  # Identity function
}


def scipysp_to_pytorchsp(sp_mx):
    """ converts scipy sparse matrix to pytorch sparse matrix """
    if not sp.isspmatrix_coo(sp_mx):
        sp_mx = sp_mx.tocoo()
    coords = np.vstack((sp_mx.row, sp_mx.col)).transpose()
    values = sp_mx.data
    shape = sp_mx.shape
    pyt_sp_mx = torch.sparse.FloatTensor(torch.LongTensor(coords.T),
                                         torch.FloatTensor(values),
                                         torch.Size(shape))
    return pyt_sp_mx

# train_edge=None(可选参数，用于传递预处理好的边集，默认None,由get_ep_data自动生成)
def main(train_edge=None):
    parser = get_parser()  # 从工具函数获取参数解析器（？）

    try:
        args = parser.parse_args()  # 解析命令行输入的参数
    except 'parser error':
        exit()  # 如果解析失败，直接退出程序
    print(args)  # 打印解析后的参数，（方便调试（确认任务，数据集，超参数是否正确））【具体打印什么？】
    setup_seed(1024)  # 固定随机种子，确保每次运行结果可复现，避免随机因素干扰实验
    # 数据加载与预处理
    # 这部分是 读取原始数据-格式标准化-封装为PyGData对象的过程
    tvt_nids = pickle.load(open(f'./data_new/graphs/{args.dataset}_tvt_nids.pkl', 'rb'))  # 节点划分，用于节点分类任务
    adj_orig = pickle.load(open(f'./data_new/graphs/{args.dataset}_adj.pkl', 'rb'))  # 原始邻接矩阵（图的边结构，稀疏矩阵格式）
    features = pickle.load(open(f'./data_new/graphs/{args.dataset}_features.pkl', 'rb'))  # 节点特征
    labels = pickle.load(open(f'./data_new/graphs/{args.dataset}_labels.pkl', 'rb'))  # 节点标签 （节点分类使用）
    # 数据预处理中的格式转换步骤
    # 对“特征矩阵、标签，邻接矩阵”三种核心数据，分别做格式转换，确保符合PyTorch/PyC的输入要求
    # 特征矩阵（features）的格式转换，统一为PyTorch float32张量
    if not isinstance(features, torch.Tensor):  # 如果不是张量（SciPy稀疏矩阵/Numpy数组）
        if isinstance(features, csr_matrix):  # 若为Scipy CSR稀疏矩阵
            features = features.toarray()  # 先转为稠密NumPy数组（PyTorch不直接支持CSR）
        # 数据格式转换（确保为Pytorch张量）
        features = torch.from_numpy(features).type(torch.float32)  # 转为float32

    # 标签的格式转换,统一为PyTorch int64张量
    if not isinstance(labels, torch.Tensor):
        labels = torch.from_numpy(labels).type(torch.int64)  # int64对应PyTorch的long类型

    # 邻接矩阵的格式转换：从Scipy稀疏矩阵-PyTorch稀疏边索引
    if not isinstance(adj_orig, sp.coo_matrix):  # 将邻接矩阵转换为SciPy的coo_matrix(坐标格式稀疏矩阵)
        adj_orig = sp.coo_matrix(adj_orig)
    adj_orig.setdiag(1)  # 给邻接矩阵加自环（保留节点自身信息,GCN等模型常见操作）
    # 转为PyTorch稀疏张量→稠密矩阵→再转为稀疏边索引（shape [2, 边数]，符合PyG的edge_index格式）
    adj_orig = scipysp_to_pytorchsp(adj_orig).to_dense()  # （？）
    adj_orig = adj_orig.to_sparse().indices()

    # 封装为PyG Data对象（数据打包，简化后续传递）
    # 计算类别数（节点分类任务用，统计标签的唯一值数量）（？）
    num_classes = torch.unique(labels).size(0)
    # 构建PyG的Data对象（将所有数据封装为一个对象，方便传递给模型/函数）
    data = Data(
        x=features,  # 节点特征【节点数，特征维度】
        edge_index=adj_orig,  # 边索引【2，边数】
        y=labels,  # 节点标签【节点数】
        train_mask=tvt_nids[0],  # 训练节点mask([节点数]，True表示该节点用于训练)
        val_mask=tvt_nids[1],  # 验证节点mask
        test_mask=tvt_nids[2],  # 测试节点mask （？）
        num_classes=num_classes  # 类别数（方便后续模型初始化）
    )
    # 重新复制mask(确保mask与Data对象绑定，节点分类时通过mask筛选训练/验证节点)
    data.train_mask, data.val_mask, data.test_mask = tvt_nids
    # 提取关键参数（后续模型初始化）
    num_classes = data.num_classes  # 类别数
    feature_size = data.x.size(1)  # 节点特征维度（模型输入维度）
    data = data.to(device)  # 将数据转移到GPU/CPU

    # 训练前初始化（“工具准备”，配置损失函数，模型，任务专属数据）
    print('Start training !!!')  # 训练启动标识，方便日志查看
    val_acc_list, test_acc_list = [], []  # （？）
    n_epochs = 10000  # 最大训练轮数
    # 统一使用交叉熵损失，无论类别数是否为2，统一使用多分类损失（更通用，避免形状问题）
    nc_criterion = torch.nn.CrossEntropyLoss()
    num = 11  # 训练轮数（重复训练10次，取平均值，减少随机波动对结果的影响）

    # 初始化模ACGA模型（核心！根据任务类型适配模型结构）
    model = GCN_Net(feature_size,  # 输入特征维度
                    num_classes,  # 类别数（节点分类任务用，边预测任务可忽略）
                    hidden=args.hidden_size,  # GCN隐藏层维度（命令行参数指定，如32/64）
                    emb_size=args.emb_size,  # 节点嵌入维度（边预测任务用，如16）（？这些都是干什么的）
                    dropout=0.5,  # Dropout概率（防止过拟合）
                    gae=args.gae,  # 是否用GAE(0=VGAE,1=GAE,命令行参数指定)
                    use_bns=args.use_bns,  # 是否用批归一化（？）
                    task=args.task).to(device)  # 任务类型
    # 调用get_ep_data生成边预测任务专属数据（节点分类返回None）
    new_label, adj_m, norm_w, pos_weight, train_edge = get_ep_data(data.cpu(), args)
    # 若task==1,将边预测专属数据转移到设备
    if args.task == 1:
        adj_m, pos_weight, train_edge = [x.to(device) for x in [adj_m, pos_weight, train_edge]]
        val_ap_list, test_ap_list = [], []  # 边预测额外记录AP指标（比AUC更关注正样本的预测精度）
    # 提取超参数，（从命令行参数获取，控制损失函数权重）  （？）
    alpha, beta, gamma = args.alpha, args.beta, args.gamma

    # 核心训练循环（主流程，按任务类型分支执行）
    for weight in range(1, num):  # 重复训练10次
        if args.task == 0:
            # 初始化训练超参数（学习率，权重衰减，控制优化器更新）
            lr, weight_decay = 5e-4, 5e-4  # , 5e-4  # , 5e-4  # , 5e-4
            # 早停相关参数（防止过拟合，验证集性能连续下降则终止训练）
            best_val_acc, last_test_acc, early_stop, patience = 0, 0, 0, 200  #
            model.reset_parameters()  # 每次重复训练前重置模型参数（避免前一次训练的参数影响）
            # 初始化优化器（Adam优化器，适配GCN参数更新）
            optimizer_cls = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            # 单轮训练循环（最大n_epoch轮，实际通过早停提前终止）
            for epoch in range(n_epochs):  # n_epochs,800
                # 1.训练表征学习（train_rep）:计算ACGA的核心损失（对比损失+重构损失+类别内损失）（？）
                rep_loss = train_rep(model, data, 2, alpha=alpha, beta=beta, gamma=gamma, new_label=new_label)
                # 2.训练分类头（train_cls）:计算节点分类损失（交叉熵），并返回分类损失值（？）
                cls_loss = train_cls(model, data, args, nc_criterion, optimizer_cls, epoch)
                # 3.总损失=表征巡视+分类损失（联合优化表征和分类头）
                loss = rep_loss + cls_loss

                # 4.反向传播与参数更新
                loss.backward()  # 计算梯度（从总损失反向传播到模型所有参数）
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 梯度裁剪（防止梯度爆炸）
                optimizer_cls.step()  # 优化器更新参数（根据梯度调整权重）
                optimizer_cls.zero_grad()  # 清空梯度（避免下一轮梯度累积）
                # 5.验证模型（无梯度计算，节省资源）
                with torch.no_grad():
                    val_acc, test_acc = test_cls(model, data)  # 计算验证集和测试机准确率
                # 6.早停逻辑（保留验证集性能最好的模型参数对应的测试机性能 ）
                if val_acc > best_val_acc:  # 若当前验证准确率更高
                    best_val_acc, last_test_acc, early_stop = val_acc, test_acc, 0
                else:
                    early_stop += 1  # 验证准确率下降，计数器+1

                if early_stop >= patience:  # 若连续200轮验证准确率未提升，终止训练
                    break
            # 打印单词训练结果（方便实时查看每轮重复训练的性能）
            print(f'num = {weight}, best_val_acc = {best_val_acc * 100:.1f}%, '
                  f'last_test_acc = {last_test_acc * 100:.1f}%')
        else:
            lr, weight_decay = 5e-4, 5e-4  # 5e-4
            # 早停与性能指标（边预测关注AUC和AP两个指标）
            best_val_acc, best_val_ap, last_test_acc, last_test_ap, early_stop, patience = 0, 0, 0, 0, 0, 200
            model.reset_parameters()  # 重置模型参数
            optimizer_ep = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # 边预测优化器

            for epoch in range(n_epochs):  # 同样重复训练10次
                # 1.训练表征学习（train_rep）:传入train_edge,适配边预测的正负样本
                rep_loss = train_rep(model, data, num_classes, alpha=alpha, beta=beta, gamma=gamma,
                                     train_edge=train_edge, new_label=new_label)
                # 2.训练边预测头（train_ep):计算边预测损失（带正负样本权重的二元交叉熵）
                ep_loss = train_ep(model, data, train_edge, adj_m, norm_w, pos_weight, optimizer_ep, args, weight)
                # 3.总损失=表征损失+边预测损失
                loss = rep_loss + ep_loss

                # 4.反向传播与参数更新（同节点分类，含梯度裁剪）
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 限制梯度范围
                optimizer_ep.step()
                optimizer_ep.zero_grad()

                # 5.验证模型（计算边预测AUC和AP指标）
                with torch.no_grad():
                    val_acc, val_ap, test_acc, test_ap = test_ep(model, data, train_edge)

                # 6.早停逻辑（以验证集AUC为核心指标）
                if val_acc > best_val_acc:
                    best_val_acc, best_val_ap, last_test_acc, last_test_ap, early_stop, = val_acc, val_ap, test_acc, \
                                                                                          test_ap, 0
                else:
                    early_stop += 1
                if early_stop >= patience:
                    break

            # 打印单次训练结果（同时展示AUC和AP）
            print(f'wight = {weight}, best_val_auc = {best_val_acc * 100:.1f}%, best_val_ap = {best_val_ap * 100:.1f}% '
                  f'last_test_auc = {last_test_acc * 100:.1f}%, last_test_ap = {last_test_ap * 100:.1f}%')
        # 记录边预测专属结果
        val_acc_list.append(best_val_acc)
        test_acc_list.append(last_test_acc)
        if args.task == 1:
            val_ap_list.append(best_val_ap)
            test_ap_list.append(last_test_ap)
    # 计算10次训练的平均准确率和标准差（统计显著性分析）
    avg_val_acc = statistics.mean(val_acc_list)
    avg_test_acc = statistics.mean(test_acc_list)
    std_acc = np.std(np.array(test_acc_list))
    # 若为边预测任务，需要额外统计AP指标
    if args.task == 1:
        avg_val_ap = statistics.mean(val_ap_list)
        avg_test_ap = statistics.mean(test_ap_list)
        std_ap = np.std(np.array(test_ap_list))
        # 打印边预测最终结果（含AUC和AP的均值、标准差）
        print(f'train num:{num - 1}, avg val auc: {avg_val_acc * 100:.1f}%, avg val ap:{avg_val_ap * 100:.1f}%, '
              f'avg test auc: {avg_test_acc * 100:.1f}%, avg test ap:{avg_test_ap * 100:.1f} '
              f'std auc:{std_acc * 100:.2f}, std ap:{std_ap * 100:.2f},')
    else:
        # 打印节点分类最终结果（含准确率的均值、标准差）
        print(f'train num:{num - 1}, avg val acc: {avg_val_acc * 100:.1f}%, '
              f'avg test acc: {avg_test_acc * 100:.1f}%, std acc:{std_acc * 100:.2f},')
    # 资源释放（避免GPU内存泄露，尤其是重复训练多次后）
    del data, model  # 删除数据和模型对象
    gc.collect()  # 手动触发Python垃圾回收
    torch.cuda.empty_cache()  # 清空GPU缓存（关键！避免后续实验因内存不足而报错）

    return avg_test_acc
# 输出测试集平均准确率/auc,作为模型最终性能指标，


if __name__ == '__main__':
    main()
