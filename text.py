import scipy.io as sio
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity  # 提前导入相似度计算函数
from scipy.sparse import coo_matrix


def process_mat_with_vars(mat_dir, dataset_name, feature_key='CC200',
                          save_dir='./data_new/graphs',
                          train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
                          adj_threshold=0.6):  # 新增：邻接矩阵相似度阈值（核心参数）
    """
    处理.mat文件（无预定义邻接矩阵），基于节点特征构建邻接矩阵，生成ACGA可读取的图数据

    参数:
        mat_dir: .mat文件所在目录
        dataset_name: 输出数据集名称
        feature_key: 特征变量名（AAL/CC200/DOS160/HO中选择）
        save_dir: 输出.pkl文件的目录
        train_ratio/val_ratio/test_ratio: 训练/验证/测试集划分比例
        adj_threshold: 特征相似度阈值（高于此值的节点间建立边）
    """
    # 1. 初始化目录和存储列表
    os.makedirs(save_dir, exist_ok=True)
    mat_files = [f for f in os.listdir(mat_dir) if f.endswith('.mat')]
    if not mat_files:
        raise ValueError(f"{mat_dir}目录下未找到.mat文件")

    all_adj = []  # 存储每个子图的邻接矩阵
    all_features = []  # 存储每个子图的节点特征
    all_labels = []  # 存储每个子图的节点标签
    node_counts = []  # 记录每个子图的节点数（用于合并时偏移ID）

    # 辅助函数：基于节点特征构建邻接矩阵（无向图，移除自环）
    def build_adj_from_feature(features, threshold):
        # 计算节点间余弦相似度矩阵（n_nodes × n_nodes）
        sim_matrix = cosine_similarity(features)
        # 相似度高于阈值设为1（有边），否则设为0（无边），并转为整数类型
        adj_matrix = (sim_matrix > threshold).astype(np.int8)
        # 移除自环（节点不与自身连接）
        np.fill_diagonal(adj_matrix, 0)
        return adj_matrix

    # 2. 遍历每个.mat文件（每个文件对应一个子图）
    for file in mat_files:
        mat_path = os.path.join(mat_dir, file)
        try:
            mat_data = sio.loadmat(mat_path)  # 读取.mat文件
        except Exception as e:
            print(f"跳过错误文件 {file}: {str(e)}")
            continue

        # ----------------------
        # 步骤1：处理节点标签（lab变量）
        # ----------------------
        labels = mat_data.get('lab')
        if labels is None:
            print(f"文件 {file} 缺少'lab'标签变量，跳过")
            continue
        # 展平为一维数组（原形状(1,148) → 目标形状(148,)）
        labels = labels.flatten()
        n_nodes = len(labels)
        if n_nodes == 0:
            print(f"文件 {file} 标签为空，跳过")
            continue

        # ----------------------
        # 步骤2：处理节点特征（如CC200/AAL）
        # ----------------------
        feat_data = mat_data.get(feature_key)
        # 检查特征变量是否存在且格式正确（必须是object数组，存储每个节点的高维特征）
        if (feat_data is None
                or not isinstance(feat_data, np.ndarray)
                or feat_data.dtype != 'object'
                or feat_data.shape != (1, n_nodes)):  # 确保是(1, n_nodes)的object数组
            print(f"文件 {file} 缺少{feature_key}或特征格式异常，跳过")
            continue

        # 提取每个节点的特征并降维（高维→一维）
        features_list = []
        valid_flag = True  # 标记特征提取是否有效
        for i in range(n_nodes):
            # 获取第i个节点的高维特征（如(200, 1833)）
            node_feat = feat_data[0, i]
            # 检查节点特征是否为有效数值数组
            if not isinstance(node_feat, np.ndarray) or node_feat.size == 0:
                valid_flag = False
                break
            # 降维逻辑（示例：按时间维度求均值，可根据数据调整）
            # 若node_feat是(特征维度, 时间点)，按时间点求均值→(特征维度,)
            if node_feat.ndim >= 2:
                node_feat_flat = np.mean(node_feat, axis=1)
            else:  # 若已为一维，直接使用
                node_feat_flat = node_feat
            # 转为(1, 特征维度)的形状，方便后续堆叠
            features_list.append(node_feat_flat.reshape(1, -1))

        # 跳过特征提取失败的文件
        if not valid_flag or len(features_list) != n_nodes:
            print(f"文件 {file} 特征提取失败，跳过")
            continue
        # 合并为子图的特征矩阵（n_nodes × 特征维度）
        features = np.vstack(features_list).astype(np.float32)

        # ----------------------
        # 步骤3：构建邻接矩阵（基于特征相似度）
        # ----------------------
        adj_matrix = build_adj_from_feature(features, adj_threshold)
        # 转换为COO格式稀疏矩阵（ACGA项目兼容格式）
        adj = coo_matrix(adj_matrix)

        # ----------------------
        # 步骤4：验证数据维度一致性
        # ----------------------
        if (adj.shape != (n_nodes, n_nodes)  # 邻接矩阵必须是n×n方阵
                or features.shape[0] != n_nodes):  # 特征数必须匹配节点数
            print(f"文件 {file} 数据维度不匹配，跳过")
            continue

        # ----------------------
        # 步骤5：暂存当前子图数据
        # ----------------------
        all_adj.append(adj)
        all_features.append(features)
        all_labels.append(labels)
        node_counts.append(n_nodes)
        print(f"成功提取 {file}：节点数={n_nodes}, 特征维度={features.shape[1]}")

    # 3. 检查是否有有效数据
    if not all_adj:
        raise ValueError("未成功提取任何有效图数据，请检查.mat文件格式")

    # ----------------------
    # 4. 合并所有子图为一个大图（节点ID全局唯一）
    # ----------------------
    # 计算节点ID偏移量（如子图1节点0→全局0，子图2节点0→全局n1，以此类推）
    offsets = np.cumsum([0] + node_counts[:-1], dtype=np.int64)  # 避免溢出
    merged_adj_data = []  # 存储合并后的边权重
    merged_adj_row = []  # 存储合并后的边行索引
    merged_adj_col = []  # 存储合并后的边列索引

    # 遍历每个子图的邻接矩阵，偏移节点ID后合并
    for i in range(len(all_adj)):
        sub_adj = all_adj[i]
        offset = offsets[i]
        # 偏移行/列索引（确保全局节点ID唯一）
        merged_adj_row.extend(sub_adj.row + offset)
        merged_adj_col.extend(sub_adj.col + offset)
        merged_adj_data.extend(sub_adj.data)

    # 构建合并后的全局邻接矩阵
    total_nodes = sum(node_counts)
    merged_adj = coo_matrix(
        (merged_adj_data, (merged_adj_row, merged_adj_col)),
        shape=(total_nodes, total_nodes),
        dtype=np.int8
    )

    # 合并全局特征矩阵和标签数组
    merged_features = np.vstack(all_features).astype(np.float32)
    merged_labels = np.hstack(all_labels).astype(np.int8)  # 标签转为整数类型

    # ----------------------
    # 5. 划分训练/验证/测试集（按节点分层划分，保证类别分布）
    # ----------------------
    indices = np.arange(total_nodes, dtype=np.int64)
    # 第一步：划分训练集和临时集（临时集=验证集+测试集）
    train_indices, temp_indices, _, temp_labels = train_test_split(
        indices, merged_labels,
        test_size=1 - train_ratio,
        stratify=merged_labels,  # 分层抽样，保持类别比例
        random_state=1024  # 固定随机种子，保证可复现
    )
    # 第二步：划分验证集和测试集
    val_indices, test_indices, _, _ = train_test_split(
        temp_indices, temp_labels,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_labels,
        random_state=1024
    )

    # 生成掩码（True表示对应集的节点）
    train_mask = np.zeros(total_nodes, dtype=bool)
    val_mask = np.zeros(total_nodes, dtype=bool)
    test_mask = np.zeros(total_nodes, dtype=bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    tvt_nids = (train_mask, val_mask, test_mask)  # 训练/验证/测试掩码元组

    # ----------------------
    # 6. 保存为ACGA可读取的.pkl文件
    # ----------------------
    save_files = {
        f"{dataset_name}_adj.pkl": merged_adj,  # 邻接矩阵（COO稀疏矩阵）
        f"{dataset_name}_features.pkl": merged_features,  # 特征矩阵
        f"{dataset_name}_labels.pkl": merged_labels,  # 标签数组
        f"{dataset_name}_tvt_nids.pkl": tvt_nids  # 训练/验证/测试掩码
    }

    for filename, data in save_files.items():
        save_path = os.path.join(save_dir, filename)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)  # 高效序列化
        print(f"已保存: {save_path}")

    # ----------------------
    # 7. 返回处理结果（便于后续分析）
    # ----------------------
    return {
        'adj': merged_adj,
        'features': merged_features,
        'labels': merged_labels,
        'tvt_nids': tvt_nids,
        'total_nodes': total_nodes,
        'feature_dim': merged_features.shape[1],
        'node_counts': node_counts  # 各子图的节点数
    }


# ----------------------
# 使用示例（主程序入口）
# ----------------------
if __name__ == "__main__":
    # 1. 配置参数（根据你的数据路径和需求调整）
    mat_dir = "data_mdd"  # .mat文件所在目录（如SITE1.mat~SITE17.mat）
    dataset_name = "combined"  # 输出数据集名称（将生成combined_*.pkl）
    feature_key = "CC200"  # 选择特征变量（AAL/CC200/DOS160/HO）
    adj_threshold = 0.6  # 特征相似度阈值（可调整，如0.5/0.7）

    # 2. 执行数据处理
    try:
        processed_data = process_mat_with_vars(
            mat_dir=mat_dir,
            dataset_name=dataset_name,
            feature_key=feature_key,
            adj_threshold=adj_threshold
        )
    except Exception as e:
        print(f"数据处理失败：{str(e)}")
        exit(1)

    # 3. 输出处理结果摘要
    print(f"\n=== 数据处理完成 ===")
    print(f"总节点数：{processed_data['total_nodes']}")
    print(f"特征维度：{processed_data['feature_dim']}")
    print(f"边数量：{processed_data['adj'].nnz // 2}（无向图，除以2去重）")
    print(f"标签类别数：{len(np.unique(processed_data['labels']))}")
    print(f"子图数量：{len(processed_data['node_counts'])}（各子图节点数：{processed_data['node_counts']}）")
    print(f"输出文件目录：{os.path.abspath('./data_new/graphs')}")