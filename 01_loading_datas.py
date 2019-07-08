import pandas as pd
import numpy as np
from time import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


# 1. 加载数据
col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
# 读取kddcup99数据集文件
kdd_data_10percent = pd.read_csv("kddcup_data_10_percent.csv", header=None, names=col_names)
# 设置显示内容长度(去掉省略号)
pd.set_option('display.max_columns', 1000)
# 打印统计结果
# print(kdd_data_10percent.describe())
# 打印'label'统计结果
# print(kdd_data_10percent['label'].value_counts())

# 2. 特征选择
num_features = [
    "duration","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
]
features = kdd_data_10percent[num_features].astype(float)
# print(features.describe())
# 把输出结果种类减少到正常和攻击
labels = kdd_data_10percent['label'].copy()
labels[labels != 'normal.'] = 'attack.'
# print(labels.value_counts())

# 3. 特征缩放(数据归一化)
# features.apply(lambda x: MinMaxScaler().fit_transform(x))
features_nor = features.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
print(features_nor.describe())