import numpy as np
import torch
from torch_geometric.data import Data
import community as community_louvain  # 需安装 python-louvain
import networkx as nx
from sklearn.model_selection import train_test_split


def origin(data: Data) -> Data:
    """使用数据集原始的分割方式（如自带的train_mask/val_mask/test_mask）"""
    # 假设数据已包含train_mask、val_mask、test_mask
    if not hasattr(data, 'train_mask') or not hasattr(data, 'val_mask') or not hasattr(data, 'test_mask'):
        raise ValueError("原始数据缺少分割掩码（train_mask/val_mask/test_mask）")
    return data


def random_spliter(data: Data, train_ratio: float = 0.6, val_ratio: float = 0.2, seed: int = 1024) -> Data:
    """
    随机分割节点为训练集、验证集和测试集
    :param data: 图数据对象（包含节点数）
    :param train_ratio: 训练集比例
    :param val_ratio: 验证集比例（测试集比例为1 - train_ratio - val_ratio）
    :param seed: 随机种子，保证可复现性
    :return: 带分割掩码的图数据
    """
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)

    # 先划分训练集和剩余集
    train_indices, rest_indices = train_test_split(
        indices, test_size=1 - train_ratio, random_state=seed
    )
    # 再从剩余集中划分验证集和测试集
    val_indices, test_indices = train_test_split(
        rest_indices, test_size=(1 - train_ratio - val_ratio) / (1 - train_ratio), random_state=seed
    )

    # 初始化掩码
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # 设置掩码
    data.train_mask[train_indices] = True
    data.val_mask[val_indices] = True
    data.test_mask[test_indices] = True

    return data


def louvain(data: Data, train_comm_ratio: float = 0.5, seed: int = 1024) -> Data:
    """
    基于Louvain社区检测算法分割数据：将部分社区作为训练集，其余作为测试/验证集
    :param data: 图数据对象（包含edge_index）
    :param train_comm_ratio: 用于训练的社区比例
    :param seed: 随机种子
    :return: 带分割掩码的图数据
    """
    # 将PyG的edge_index转换为NetworkX图
    edges = data.edge_index.numpy().T
    g = nx.Graph()
    g.add_edges_from(edges)

    # 使用Louvain算法获取社区划分
    partition = community_louvain.best_partition(g, random_state=seed)
    communities = list(set(partition.values()))  # 所有社区的ID
    num_communities = len(communities)

    # 随机选择部分社区作为训练集
    np.random.seed(seed)
    train_comms = np.random.choice(
        communities,
        size=int(num_communities * train_comm_ratio),
        replace=False
    )

    # 划分节点到训练集、验证集、测试集（验证集从训练社区中随机划分）
    train_nodes = [node for node, comm in partition.items() if comm in train_comms]
    test_nodes = [node for node, comm in partition.items() if comm not in train_comms]

    # 从训练节点中划分验证集（20%作为验证）
    train_nodes, val_nodes = train_test_split(train_nodes, test_size=0.2, random_state=seed)

    # 初始化掩码
    num_nodes = data.num_nodes
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # 设置掩码
    data.train_mask[train_nodes] = True
    data.val_mask[val_nodes] = True
    data.test_mask[test_nodes] = True

    return data


def rand_walk(data: Data, subgraph_size: int, train_ratio: float = 0.6, seed: int = 1024) -> Data:
    """
    基于随机游走的子图分割：从随机节点开始游走生成子图，作为训练集，剩余为测试集
    :param data: 图数据对象（包含edge_index和num_nodes）
    :param subgraph_size: 子图节点数量（训练集大致规模）
    :param train_ratio: 训练集占比（用于从子图中划分验证集）
    :param seed: 随机种子
    :return: 带分割掩码的图数据
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 构建邻接表
    adj = [[] for _ in range(data.num_nodes)]
    edges = data.edge_index.numpy()
    for u, v in zip(edges[0], edges[1]):
        adj[u].append(v)
        adj[v].append(u)  # 假设无向图

    # 随机选择起点开始游走
    start_node = np.random.choice(data.num_nodes)
    visited = set()
    current = start_node
    visited.add(current)

    # 随机游走生成子图
    while len(visited) < subgraph_size:
        neighbors = adj[current]
        if not neighbors:
            break  # 孤立节点，无法继续游走
        current = np.random.choice(neighbors)
        visited.add(current)
        if len(visited) >= subgraph_size:
            break

    # 从游走得到的节点中划分训练集和验证集
    train_nodes, val_nodes = train_test_split(
        list(visited), test_size=1 - train_ratio, random_state=seed
    )
    # 测试集为未被访问的节点
    test_nodes = [node for node in range(data.num_nodes) if node not in visited]

    # 初始化掩码
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    # 设置掩码
    data.train_mask[train_nodes] = True
    data.val_mask[val_nodes] = True
    data.test_mask[test_nodes] = True

    return data