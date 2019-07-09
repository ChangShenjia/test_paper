import pandas as pd
import numpy as np
from time import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


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
# 读取kddcup_data_10_percent数据集文件
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
# 把输出结果种类减少到正常和攻击两种
labels_ten = kdd_data_10percent['label'].copy()
labels_ten[labels_ten != 'normal.'] = 'attack.'
# print(labels_ten.value_counts())

# 3. 特征缩放(数据归一化)
# features.apply(lambda x: MinMaxScaler().fit_transform(x))
features_nor = features.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
# print(features_nor.describe())
f_t = features_nor.isnull().any()
f_train = features_nor.fillna('0')
# print(f_train)

# 4. 训练分类器(knn)
clf = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', leaf_size=500)
t0 = time()
clf.fit(f_train, labels_ten)
tt = time()-t0
print("Classifier trained in {} seconds".format(round(tt, 3)))
# 读取corrected数据集文件
# kdd_data_corrected = pd.read_csv(r"corrected.csv", header=None, names=col_names)
# print(kdd_data_corrected['label'].value_counts())
# 把输出结果种类减少到正常和攻击两种
# kdd_data_corrected['label'][kdd_data_corrected['label'] != 'normal.'] = 'attack.'
# labels_c = kdd_data_corrected['label'].copy()
# labels_c[labels_c != 'normal.'] = 'attack.'
# print(labels_c.value_counts())
# 特征缩放(数据归一化)
# features_c = kdd_data_corrected[num_features].astype(float)
# features_c_nor = features_c.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
# print(features_c_nor)
# features_nor_train, features_nor_test, labels_ten_train, labels_ten_test = train_test_split(
#     features_c,
#     kdd_data_corrected['label'],
#     test_size=0.1,
#     random_state=42)
# 使用分类器和测试数据进行预测
# t0 = time()
# pred = clf.predict(features_nor_test)
# tt = time() - t0
# print("Predicted in {} seconds".format(round(tt, 3)))