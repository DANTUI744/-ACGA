import os
import pickle
import torch
import numpy as np  # 新增：导入NumPy库
from torch_geometric.data import Data


def view_combined_dataset(data_dir='./data_new/graphs'):
    """查看combined数据集的基本信息"""
    # 定义数据集文件路径
    dataset_name = 'combined'
    files = {
        'features': f'{dataset_name}_features.pkl',
        'labels': f'{dataset_name}_labels.pkl',
        'adj': f'{dataset_name}_adj.pkl',
        'tvt_nids': f'{dataset_name}_tvt_nids.pkl'  # 训练/验证/测试集划分
    }

    # 检查文件是否存在
    for key, file in files.items():
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f"警告：{key}文件不存在 - {file_path}")
            return

    try:
        # 加载数据
        with open(os.path.join(data_dir, files['features']), 'rb') as f:
            features = pickle.load(f)
        with open(os.path.join(data_dir, files['labels']), 'rb') as f:
            labels = pickle.load(f)
        with open(os.path.join(data_dir, files['adj']), 'rb') as f:
            adj = pickle.load(f)
        with open(os.path.join(data_dir, files['tvt_nids']), 'rb') as f:
            tvt_nids = pickle.load(f)  # (train_mask, val_mask, test_mask)

        # 打印基本信息
        print(f"数据集: {dataset_name}")
        print(f"特征矩阵形状: {features.shape if hasattr(features, 'shape') else '未知'}")
        print(f"标签数量: {len(labels) if hasattr(labels, '__len__') else '未知'}")
        print(
            f"唯一标签数: {len(torch.unique(torch.tensor(labels))) if isinstance(labels, (list, np.ndarray, torch.Tensor)) else '未知'}")
        print(f"邻接矩阵类型: {type(adj)}")
        print(f"邻接矩阵形状: {adj.shape if hasattr(adj, 'shape') else '未知'}")
        print(
            f"训练集节点数: {sum(tvt_nids[0]) if isinstance(tvt_nids[0], (list, np.ndarray, torch.Tensor)) else '未知'}")
        print(
            f"验证集节点数: {sum(tvt_nids[1]) if isinstance(tvt_nids[1], (list, np.ndarray, torch.Tensor)) else '未知'}")
        print(
            f"测试集节点数: {sum(tvt_nids[2]) if isinstance(tvt_nids[2], (list, np.ndarray, torch.Tensor)) else '未知'}")

    except Exception as e:
        print(f"加载数据时出错: {str(e)}")


if __name__ == "__main__":
    view_combined_dataset()