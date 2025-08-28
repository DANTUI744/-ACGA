import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# 加载原始特征
with open('data_new/graphs/combined_features.pkl', 'rb') as f:
    features = pickle.load(f)

# 初始化标准化器（均值0，标准差1）
scaler = StandardScaler()
# 对特征进行标准化（注意：sklearn要求输入是二维数组，这里正好符合）
features_scaled = scaler.fit_transform(features)

# 保存标准化后的特征（覆盖原文件）
with open('data_new/graphs/combined_features.pkl', 'wb') as f:
    pickle.dump(features_scaled, f)

# 验证标准化结果（可选）
print("标准化后最大值：", np.max(features_scaled))  # 应在 3~5 左右
print("标准化后最小值：", np.min(features_scaled))  # 应在 -3~-5 左右
print("标准化后均值：", np.mean(features_scaled))    # 接近 0
print("标准化后标准差：", np.std(features_scaled))  # 接近 1