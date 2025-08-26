import scipy.io as sio
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.sparse import coo_matrix


def process_mat_with_vars(mat_dir, dataset_name, feature_key='CC200',
                          save_dir='./data_new/graphs',
                          train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
                          adj_threshold=0.6):
    """
    处理所有.mat文件（动态适配PCA维度，确保无文件被跳过）
    """
    os.makedirs(save_dir, exist_ok=True)
    mat_files = [f for f in os.listdir(mat_dir) if f.endswith('.mat')]
    if not mat_files:
        raise ValueError(f"{mat_dir}目录下未找到.mat文件")

    # ----------------------
    # 预扫描：统计所有文件的最大可能维度（为了确定全局安全维度）
    # ----------------------
    max_possible_dims = []  # 存储每个文件支持的最大维度（min(节点数, 原始特征维度)）
    valid_files = []  # 存储有效文件路径

    for file in mat_files:
        mat_path = os.path.join(mat_dir, file)
        try:
            mat_data = sio.loadmat(mat_path)
        except Exception as e:
            print(f"预扫描：跳过错误文件 {file}: {str(e)}")
            continue

        # 检查标签
        labels = mat_data.get('lab')
        if labels is None:
            print(f"预扫描：{file} 缺少'lab'，跳过")
            continue
        labels = labels.flatten()
        n_nodes = len(labels)
        if n_nodes == 0:
            print(f"预扫描：{file} 标签为空，跳过")
            continue

        # 检查特征
        feat_data = mat_data.get(feature_key)
        if (feat_data is None
                or not isinstance(feat_data, np.ndarray)
                or feat_data.dtype != 'object'
                or feat_data.shape != (1, n_nodes)):
            print(f"预扫描：{file} 特征格式异常，跳过")
            continue

        # 计算该文件的原始特征维度（取第一个有效节点的特征维度）
        try:
            first_feat = feat_data[0, 0]
            if not isinstance(first_feat, np.ndarray) or first_feat.size == 0:
                print(f"预扫描：{file} 特征数据无效，跳过")
                continue
            raw_feat_dim = first_feat.flatten().shape[0]  # 展平后的维度
            max_possible_dim = min(n_nodes, raw_feat_dim)  # 该文件支持的最大维度
            max_possible_dims.append(max_possible_dim)
            valid_files.append(file)  # 记录有效文件
        except Exception as e:
            print(f"预扫描：{file} 特征维度计算失败：{str(e)}，跳过")
            continue

    if not valid_files:
        raise ValueError("无有效文件可处理")

    # 确定全局安全维度（所有文件都能支持的最大维度）
    global_safe_dim = min(max_possible_dims)
    print(f"\n=== 预扫描完成 ===")
    print(f"有效文件数：{len(valid_files)}")
    print(f"全局安全维度（所有文件均支持）：{global_safe_dim}")

    # ----------------------
    # 正式处理：用全局安全维度统一所有文件
    # ----------------------
    all_adj = []
    all_features = []
    all_labels = []
    node_counts = []

    def build_adj_from_feature(features, threshold):
        sim_matrix = cosine_similarity(features)
        adj_matrix = (sim_matrix > threshold).astype(np.int8)
        np.fill_diagonal(adj_matrix, 0)
        return adj_matrix

    for file in valid_files:  # 只处理预扫描通过的文件
        mat_path = os.path.join(mat_dir, file)
        mat_data = sio.loadmat(mat_path)

        # 处理标签
        labels = mat_data.get('lab').flatten()
        n_nodes = len(labels)

        # 处理特征
        feat_data = mat_data.get(feature_key)
        raw_features = []
        valid_flag = True
        for i in range(n_nodes):
            node_feat = feat_data[0, i]
            if not isinstance(node_feat, np.ndarray) or node_feat.size == 0:
                valid_flag = False
                break
            raw_features.append(node_feat.flatten())
        if not valid_flag or len(raw_features) != n_nodes:
            print(f"{file} 特征提取失败，跳过（预扫描未发现的异常）")
            continue
        raw_features = np.array(raw_features, dtype=np.float32)

        # ----------------------
        # 动态PCA降维 + 填充（核心优化）
        # ----------------------
        try:
            # 第一步：PCA降到该文件支持的最大维度（不超过global_safe_dim）
            file_max_dim = min(n_nodes, raw_features.shape[1], global_safe_dim)
            pca = PCA(n_components=file_max_dim, random_state=1024)
            features_pca = pca.fit_transform(raw_features)

            # 第二步：用0填充到全局安全维度（确保所有文件维度统一）
            features = np.pad(
                features_pca,
                ((0, 0), (0, global_safe_dim - file_max_dim)),
                mode='constant'
            )
        except Exception as e:
            print(f"{file} 处理失败：{str(e)}，跳过")
            continue

        # 构建邻接矩阵
        adj_matrix = build_adj_from_feature(features, adj_threshold)
        adj = coo_matrix(adj_matrix)

        # 验证并保存
        if adj.shape == (n_nodes, n_nodes) and features.shape == (n_nodes, global_safe_dim):
            all_adj.append(adj)
            all_features.append(features)
            all_labels.append(labels)
            node_counts.append(n_nodes)
            print(f"成功处理 {file}：节点数={n_nodes}, 特征维度={features.shape[1]}（统一后）")
        else:
            print(f"{file} 维度不匹配，跳过")

    # 后续合并、划分、保存逻辑（与之前一致）
    if not all_adj:
        raise ValueError("未成功处理任何文件")

    # 合并子图
    offsets = np.cumsum([0] + node_counts[:-1], dtype=np.int64)
    merged_adj_data = []
    merged_adj_row = []
    merged_adj_col = []
    for i in range(len(all_adj)):
        sub_adj = all_adj[i]
        offset = offsets[i]
        merged_adj_row.extend(sub_adj.row + offset)
        merged_adj_col.extend(sub_adj.col + offset)
        merged_adj_data.extend(sub_adj.data)

    total_nodes = sum(node_counts)
    merged_adj = coo_matrix(
        (merged_adj_data, (merged_adj_row, merged_adj_col)),
        shape=(total_nodes, total_nodes),
        dtype=np.int8
    )

    # 合并特征（此时所有特征维度均为global_safe_dim）
    merged_features = np.vstack(all_features).astype(np.float32)
    merged_labels = np.hstack(all_labels).astype(np.int8)

    # 划分数据集
    indices = np.arange(total_nodes, dtype=np.int64)
    train_indices, temp_indices, _, temp_labels = train_test_split(
        indices, merged_labels,
        test_size=1 - train_ratio,
        stratify=merged_labels,
        random_state=1024
    )
    val_indices, test_indices, _, _ = train_test_split(
        temp_indices, temp_labels,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_labels,
        random_state=1024
    )

    train_mask = np.zeros(total_nodes, dtype=bool)
    val_mask = np.zeros(total_nodes, dtype=bool)
    test_mask = np.zeros(total_nodes, dtype=bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    tvt_nids = (train_mask, val_mask, test_mask)

    # 保存结果
    save_files = {
        f"{dataset_name}_adj.pkl": merged_adj,
        f"{dataset_name}_features.pkl": merged_features,
        f"{dataset_name}_labels.pkl": merged_labels,
        f"{dataset_name}_tvt_nids.pkl": tvt_nids
    }

    for filename, data in save_files.items():
        save_path = os.path.join(save_dir, filename)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"已保存: {save_path}")

    return {
        'adj': merged_adj,
        'features': merged_features,
        'labels': merged_labels,
        'tvt_nids': tvt_nids,
        'total_nodes': total_nodes,
        'feature_dim': global_safe_dim,
        'node_counts': node_counts
    }


if __name__ == "__main__":
    mat_dir = "data_mdd"
    dataset_name = "combined"
    feature_key = "CC200"
    adj_threshold = 0.6

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

    print(f"\n=== 数据处理完成 ===")
    print(f"总节点数：{processed_data['total_nodes']}")
    print(f"统一后特征维度：{processed_data['feature_dim']}（全局安全维度）")
    print(f"边数量：{processed_data['adj'].nnz // 2}")
    print(f"标签类别数：{len(np.unique(processed_data['labels']))}")
    print(f"处理文件数：{len(processed_data['node_counts'])}")
    print(f"输出目录：{os.path.abspath('./data_new/graphs')}")
