import pickle
import numpy as np
import pandas as pd

# --------------------------
# 1. 加载掩码数据（train_mask, val_mask, test_mask）
# --------------------------
mask_path = "data_new/graphs/combined_tvt_nids.pkl"  # 替换为实际文件路径
with open(mask_path, "rb") as f:
    train_mask, val_mask, test_mask = pickle.load(f)

# 检查掩码格式（需为 NumPy 布尔数组，与标签长度匹配）
assert all(isinstance(mask, np.ndarray) and mask.dtype == bool for mask in [train_mask, val_mask, test_mask]), \
    "掩码需为 NumPy 布尔数组"

# --------------------------
# 2. 加载标签数据（需替换为实际加载逻辑！）
#    示例：若标签是 PyTorch Geometric Data 对象的属性，可这样加载：
#    from data.load_combined import load_combined_dataset
#    data = load_combined_dataset()
#    y = data.y.numpy()  # 转换为 NumPy 数组
# --------------------------
# ========== 以下为示例占位，需替换为真实标签加载代码 ==========
# 假设标签是长度与掩码匹配的 NumPy 数组（实际需根据数据集修改）
num_nodes = len(train_mask)  # 掩码长度即节点数
y = np.random.randint(0, 3, size=num_nodes)  # 模拟标签（实际需删除此行！）
# ========== 以上为示例占位，需替换为真实标签加载代码 ==========


# --------------------------
# 3. 筛选各集合的标签
# --------------------------
train_labels = y[train_mask]
val_labels = y[val_mask]
test_labels = y[test_mask]


# --------------------------
# 4. 定义统计函数：计算标签数量及占比
# --------------------------
def calculate_label_stats(labels, set_name):
    """
    统计单个集合（训练/验证/测试）的标签分布：
    - 输入：labels（该集合的标签数组）、set_name（集合名称，如"训练集"）
    - 输出：DataFrame，包含标签值、数量、占比
    """
    if len(labels) == 0:
        # 处理空集合（理论上不应出现，若出现需检查划分逻辑）
        return pd.DataFrame({
            "集合": [set_name],
            "标签": ["空集合"],
            "数量": [0],
            "占比": [0.0]
        })

    # 统计每个标签的出现次数
    unique_labels, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    proportions = counts / total  # 计算占比

    # 构建结果 DataFrame
    stats_df = pd.DataFrame({
        "集合": [set_name] * len(unique_labels),
        "标签": unique_labels,
        "数量": counts,
        "占比": proportions
    })

    # 添加“总计”行
    total_row = pd.DataFrame({
        "集合": [set_name],
        "标签": ["总计"],
        "数量": [total],
        "占比": [1.0]  # 占比固定为100%
    }, index=[len(unique_labels)])  # 插入到最后一行

    return pd.concat([stats_df, total_row])


# --------------------------
# 5. 统计三个集合的标签分布
# --------------------------
train_stats = calculate_label_stats(train_labels, "训练集")
val_stats = calculate_label_stats(val_labels, "验证集")
test_stats = calculate_label_stats(test_labels, "测试集")

# --------------------------
# 6. 合并结果并输出
# --------------------------
final_stats = pd.concat([train_stats, val_stats, test_stats], ignore_index=True)
print("=== 各集合标签分布统计 ===")
print(final_stats)

# （可选）保存为 CSV 文件
# final_stats.to_csv("label_distribution.csv", index=False)