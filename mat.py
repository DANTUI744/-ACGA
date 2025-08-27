import scipy.io as sio
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.sparse import coo_matrix


def process_mat_with_vars(mat_dir, dataset_name, feature_key='CC200',
                          save_dir='./data_new/graphs',
                          train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
                          adj_top_k=10,  # 动态阈值：前k%邻居
                          pca_var_threshold=0.8):  # PCA保留80%信息
    os.makedirs(save_dir, exist_ok=True)
    mat_files = [f for f in os.listdir(mat_dir) if f.endswith('.mat')]
    if not mat_files:
        raise ValueError(f"{mat_dir}目录下未找到.mat文件")

    # ----------------------
    # 第一步：预扫描（必须先执行！获取file_pca_info，包含每个文件的关键信息）
    # ----------------------
    file_pca_info = []  # (文件路径, 80%信息维度, 节点数, 原始特征维度)
    valid_files = []

    for file in mat_files:
        mat_path = os.path.join(mat_dir, file)
        try:
            mat_data = sio.loadmat(mat_path)
            # 1. 检查标签和特征存在性
            labels = mat_data.get('lab')
            feat_data = mat_data.get(feature_key)
            if labels is None or feat_data is None or not isinstance(feat_data, np.ndarray):
                print(f"预扫描：{file} 标签/特征缺失，跳过")
                continue
            labels = labels.flatten()
            n_nodes = len(labels)
            if n_nodes < 5:  # 节点数过少，无意义
                print(f"预扫描：{file} 节点数<5，跳过")
                continue

            # 2. 提取原始特征并验证维度
            raw_features = []
            valid_feat = True
            for i in range(n_nodes):
                node_feat = feat_data[0, i]
                if not isinstance(node_feat, np.ndarray) or node_feat.size == 0:
                    valid_feat = False
                    break
                raw_features.append(node_feat.flatten())
            if not valid_feat or len(raw_features) != n_nodes:
                print(f"预扫描：{file} 特征无效，跳过")
                continue
            raw_features = np.array(raw_features, dtype=np.float32)
            raw_feat_dim = raw_features.shape[1]
            if raw_feat_dim < 2:  # 特征维度不足，无法PCA
                print(f"预扫描：{file} 原始特征维度<2，跳过")
                continue

            # 3. 单文件标准化+PCA，计算80%信息维度
            scaler_temp = StandardScaler()
            raw_features_scaled_temp = scaler_temp.fit_transform(raw_features)
            pca_temp = PCA(n_components=None, random_state=1024)
            pca_temp.fit_transform(raw_features_scaled_temp)
            cumulative_var_temp = np.cumsum(pca_temp.explained_variance_ratio_)
            var_80_dim = np.argmax(cumulative_var_temp >= pca_var_threshold) + 1  # 最小维度
            var_80_dim = max(var_80_dim, 2)  # 至少2维
            if var_80_dim > n_nodes:  # 维度不超过节点数（避免过拟合）
                var_80_dim = n_nodes

            # 4. 记录有效文件信息（后续正式处理用）
            file_pca_info.append((mat_path, var_80_dim, n_nodes, raw_feat_dim))
            valid_files.append(file)
            print(f"预扫描：{file} → 80%信息维度={var_80_dim}，节点数={n_nodes}，原始特征维度={raw_feat_dim}")

        except Exception as e:
            print(f"预扫描：{file} 错误：{str(e)}，跳过")
            continue

    if not valid_files:
        raise ValueError("无有效文件可处理！请检查.mat文件格式")

    # 确定PCA相关的全局统一维度（所有文件80%信息维度的最大值）
    global_pca_dim = max([info[1] for info in file_pca_info])
    print(f"\n=== 预扫描完成 ===")
    print(f"有效文件数：{len(valid_files)}")
    print(f"PCA全局统一维度（保留80%+信息）：{global_pca_dim}")

    # ----------------------
    # 第二步：正式处理 - 收集所有文件特征+统一维度（核心修复！）
    # ----------------------
    all_raw_features = []  # 存储统一维度后的所有原始特征
    all_labels = []        # 存储所有标签（后续合并用）
    node_counts = []       # 存储每个文件的节点数（后续合并邻接矩阵用）
    scaler_global = StandardScaler()  # 全局标准化器（拟合统一维度后的特征）

    # 1. 第一步：确定特征维度统一的目标（选所有文件原始特征维度的最小值，避免信息丢失）
    raw_feat_dims = [info[3] for info in file_pca_info]  # 所有文件的原始特征维度
    target_feat_dim = min(raw_feat_dims)  # 统一到最小维度（截断长特征，填充短特征）
    print(f"\n=== 特征维度统一 ===")
    print(f"所有文件原始特征维度：{raw_feat_dims}")
    print(f"统一目标维度：{target_feat_dim}（最小维度，避免信息丢失）")

    # 2. 第二步：逐文件收集特征并统一维度
    for idx, (mat_path, var_80_dim, n_nodes, raw_feat_dim) in enumerate(file_pca_info):
        filename = os.path.basename(mat_path)
        try:
            mat_data = sio.loadmat(mat_path)
            # 提取标签（后续合并用）
            labels = mat_data.get('lab').flatten()
            # 提取原始特征并统一维度
            feat_data = mat_data.get(feature_key)
            raw_features = []
            for i in range(n_nodes):
                node_feat = feat_data[0, i].flatten()  # 展平为1D
                # 统一维度逻辑：
                if len(node_feat) > target_feat_dim:
                    # 特征过长：截断到目标维度（保留前N维，通常前维信息更重要）
                    node_feat = node_feat[:target_feat_dim]
                elif len(node_feat) < target_feat_dim:
                    # 特征过短：用0填充（后续标准化会抵消0的影响）
                    pad_length = target_feat_dim - len(node_feat)
                    node_feat = np.pad(node_feat, (0, pad_length), mode='constant', constant_values=0)
                raw_features.append(node_feat)
            # 转换为numpy数组
            raw_features = np.array(raw_features, dtype=np.float32)
            # 验证统一后维度是否正确
            assert raw_features.shape[1] == target_feat_dim, f"{filename} 维度统一失败！"

            # 暂存数据
            all_raw_features.append(raw_features)
            all_labels.append(labels)
            node_counts.append(n_nodes)
            print(f"  处理 {filename} → 统一后特征维度：{raw_features.shape[1]}，节点数：{n_nodes}")

        except Exception as e:
            print(f"  警告：{filename} 特征收集失败：{str(e)}，跳过该文件")
            continue

    # 3. 第三步：拟合全局标准化器（基于统一维度后的所有特征）
    if not all_raw_features:
        raise ValueError("所有文件特征收集失败！")
    # 拼接所有统一维度后的特征
    all_raw_features_merged = np.vstack(all_raw_features)
    scaler_global.fit(all_raw_features_merged)
    # 检查是否存在方差为0的列（会导致标准化时出现inf）
    zero_std_cols = np.where(scaler_global.scale_ == 0)[0]
    if len(zero_std_cols) > 0:
        print(f"\n⚠️ 全局警告：发现 {len(zero_std_cols)} 列方差为0（列索引：{zero_std_cols}）")
        print("  建议：后续可删除这些列，避免标准化后出现inf")

    # ----------------------
    # 第三步：正式处理 - 逐文件生成PCA特征+邻接矩阵
    # ----------------------
    all_adj = []  # 存储所有文件的邻接矩阵
    all_pca_features = []  # 存储所有文件的PCA特征（统一到global_pca_dim）

    def build_adj_dynamic(similarity_matrix, top_k_ratio):
        """动态构建无向邻接矩阵（取每个节点前k%高相似度邻居）"""
        n_nodes = similarity_matrix.shape[0]
        adj = np.zeros_like(similarity_matrix, dtype=np.int8)
        top_k = max(1, int(n_nodes * top_k_ratio / 100))  # 至少1个邻居
        for i in range(n_nodes):
            sim_i = similarity_matrix[i].copy()
            sim_i[i] = -np.inf  # 排除自身
            top_neighbors = np.argsort(sim_i)[-top_k:]  # 前top_k个邻居
            adj[i, top_neighbors] = 1
        adj = np.maximum(adj, adj.T)  # 确保对称（无向图）
        return adj

    # 逐文件处理
    for idx, (mat_path, var_80_dim, n_nodes, raw_feat_dim) in enumerate(file_pca_info):
        filename = os.path.basename(mat_path)
        try:
            # 1. 获取统一维度后的原始特征+标准化
            raw_features = all_raw_features[idx]  # 之前已统一维度
            raw_features_scaled = scaler_global.transform(raw_features)
            # 检查标准化后是否有异常值
            if np.isnan(raw_features_scaled).any() or np.isinf(raw_features_scaled).any():
                raise ValueError("标准化后含NaN/inf（可能因方差为0列导致）")

            # 2. PCA降维（保留80%信息）
            pca = PCA(n_components=var_80_dim, random_state=1024)
            features_pca = pca.fit_transform(raw_features_scaled)
            cumulative_var = np.cumsum(pca.explained_variance_ratio_)
            if cumulative_var[-1] < pca_var_threshold - 0.1:  # 允许10%误差
                raise ValueError(f"PCA累计方差不足：{cumulative_var[-1]:.2%} < {pca_var_threshold-0.1:.2%}")

            # 3. PCA特征扩展到全局统一维度（global_pca_dim）
            features_pca_unified = np.zeros((n_nodes, global_pca_dim), dtype=np.float32)
            pca_cols = min(var_80_dim, global_pca_dim)
            features_pca_unified[:, :pca_cols] = features_pca[:, :pca_cols]
            # 剩余列用PCA最后一列填充（保持特征相关性，避免0噪声）
            if pca_cols < global_pca_dim:
                fill_col = features_pca[:, -1].reshape(-1, 1)
                features_pca_unified[:, pca_cols:] = fill_col

            # 4. 构建邻接矩阵（基于标准化后的原始特征，避免PCA失真）
            sim_matrix = cosine_similarity(raw_features_scaled)
            adj_matrix = build_adj_dynamic(sim_matrix, adj_top_k)
            adj = coo_matrix(adj_matrix)  # 转换为稀疏矩阵（节省内存）
            if adj.nnz == 0:
                raise ValueError("邻接矩阵无有效边（可提高adj_top_k阈值）")

            # 5. 暂存结果
            all_adj.append(adj)
            all_pca_features.append(features_pca_unified)
            print(f"\n处理 {filename} 成功：")
            print(f"  - PCA累计方差：{cumulative_var[-1]:.2%}")
            print(f"  - 统一后PCA维度：{features_pca_unified.shape[1]}")
            print(f"  - 邻接边数：{adj.nnz//2}（无向图）")

        except Exception as e:
            print(f"\n处理 {filename} 失败：{str(e)}，跳过该文件")
            # 移除之前暂存的该文件数据（保持列表同步）
            del all_raw_features[idx]
            del all_labels[idx]
            del node_counts[idx]
            continue

    if not all_adj:
        raise ValueError("所有文件处理失败！请检查上述错误日志")

    # ----------------------
    # 第四步：合并数据+划分数据集+保存
    # ----------------------
    # 1. 合并邻接矩阵（稀疏矩阵）
    total_nodes = sum(node_counts)
    offsets = np.cumsum([0] + node_counts[:-1], dtype=np.int64)  # 节点索引偏移量
    merged_adj_data = []
    merged_adj_row = []
    merged_adj_col = []
    for i in range(len(all_adj)):
        sub_adj = all_adj[i]
        offset = offsets[i]
        merged_adj_row.extend(sub_adj.row + offset)
        merged_adj_col.extend(sub_adj.col + offset)
        merged_adj_data.extend(sub_adj.data)
    merged_adj = coo_matrix(
        (merged_adj_data, (merged_adj_row, merged_adj_col)),
        shape=(total_nodes, total_nodes),
        dtype=np.int8
    )

    # 2. 合并特征和标签
    merged_features = np.vstack(all_pca_features).astype(np.float32)
    merged_labels = np.hstack(all_labels).astype(np.int8)

    # 3. 划分训练/验证/测试集（分层抽样，保证标签分布一致）
    indices = np.arange(total_nodes, dtype=np.int64)
    # 划分训练集和临时集（6:4）
    train_indices, temp_indices, _, temp_labels = train_test_split(
        indices, merged_labels, test_size=1-train_ratio,
        stratify=merged_labels, random_state=1024
    )
    # 划分验证集和测试集（2:2）
    val_indices, test_indices, _, _ = train_test_split(
        temp_indices, temp_labels, test_size=test_ratio/(val_ratio+test_ratio),
        stratify=temp_labels, random_state=1024
    )
    # 生成掩码
    train_mask = np.zeros(total_nodes, dtype=bool)
    val_mask = np.zeros(total_nodes, dtype=bool)
    test_mask = np.zeros(total_nodes, dtype=bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    tvt_nids = (train_mask, val_mask, test_mask)

    # 4. 保存结果
    save_files = {
        f"{dataset_name}_adj.pkl": merged_adj,
        f"{dataset_name}_features.pkl": merged_features,
        f"{dataset_name}_labels.pkl": merged_labels,
        f"{dataset_name}_tvt_nids.pkl": tvt_nids,
        f"{dataset_name}_scaler.pkl": scaler_global
    }
    for filename, data in save_files.items():
        save_path = os.path.join(save_dir, filename)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\n已保存：{save_path}")

    # ----------------------
    # 输出处理结果汇总
    # ----------------------
    print(f"\n=== 数据处理完成 ===")
    print(f"总节点数：{total_nodes}")
    print(f"统一后特征维度：{merged_features.shape[1]}")
    print(f"合并后边数：{merged_adj.nnz//2}（无向图）")
    print(f"标签类别数：{len(np.unique(merged_labels))}")
    print(f"训练集节点数：{train_mask.sum()}（{train_ratio*100:.1f}%）")
    print(f"验证集节点数：{val_mask.sum()}（{val_ratio*100:.1f}%）")
    print(f"测试集节点数：{test_mask.sum()}（{test_ratio*100:.1f}%）")
    print(f"输出目录：{os.path.abspath(save_dir)}")

    return {
        'adj': merged_adj,
        'features': merged_features,
        'labels': merged_labels,
        'tvt_nids': tvt_nids,
        'scaler': scaler_global,
        'total_nodes': total_nodes
    }


if __name__ == "__main__":
    # 配置参数（根据实际情况修改）
    mat_dir = "data_mdd"          # .mat文件所在目录
    dataset_name = "combined"     # 输出数据集名称（文件前缀）
    feature_key = "CC200"         # .mat文件中特征字段的键
    adj_top_k = 10                # 邻接矩阵：取每个节点前10%邻居
    pca_var_threshold = 0.8       # PCA保留80%信息

    try:
        processed_data = process_mat_with_vars(
            mat_dir=mat_dir,
            dataset_name=dataset_name,
            feature_key=feature_key,
            adj_top_k=adj_top_k,
            pca_var_threshold=pca_var_threshold
        )
    except Exception as e:
        print(f"\n数据处理失败：{str(e)}")
        exit(1)