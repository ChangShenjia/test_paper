import pandas as pd
import numpy as np
from time import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans


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
features_ten = kdd_data_10percent[num_features].astype(float)
# print(features_ten.describe())
# 把输出结果种类减少到正常和攻击两种
labels_ten = kdd_data_10percent['label'].copy()
labels_ten[labels_ten != 'normal.'] = 'attack.'
# print(labels_ten.value_counts())

# 3. 特征缩放(数据归一化)
# features_ten.apply(lambda x: MinMaxScaler().fit_transform(x))
features_ten_nor = features_ten.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
# print(features_ten_nor.describe())
f_t = features_ten_nor.isnull().any()
f_t_train = features_ten_nor.fillna('0')
# print(f_t_train)

# 4. 训练/使用分类器(knn)
# 1) 使用10%的数据集进行训练
clf = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', leaf_size=500)
t0 = time()
clf.fit(f_t_train, labels_ten)
tt = time()-t0
print("Classifier trained in {} seconds".format(round(tt, 3)))
# 2) 使用corrected数据集对分类器进行测试
# 读取corrected数据集文件
kdd_data_corrected = pd.read_csv("corrected.csv", header=None, names=col_names)
# print(kdd_data_corrected['label'].value_counts())
# 把输出结果种类减少到正常和攻击两种
# kdd_data_corrected['label'][kdd_data_corrected['label'] != 'normal.'] = 'attack.'
labels_c = kdd_data_corrected['label'].copy()
labels_c[labels_c != 'normal.'] = 'attack.'
# print(labels_c.value_counts())
# 特征缩放(数据归一化)
features_c = kdd_data_corrected[num_features].astype(float)
features_c_nor = features_c.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
# print(features_c_nor.describe())
f_c = features_c_nor.isnull().any()
f_c_train = features_c_nor.fillna('0')
# print(f_c_train)
features_train, features_test, labels_train, labels_test = train_test_split(
    f_c_train,
    labels_c,
    test_size=0.1,
    random_state=42)
# 使用分类器和测试数据进行预测
t0 = time()
pred_kn = clf.predict(features_test)
tt = time() - t0
print("Predicted in {} seconds".format(round(tt, 3)))
# 使用测试标签计算R平方值
acc = accuracy_score(pred_kn, labels_test)
print("R squared is {}.".format(round(acc, 4)))

# 5. K-Means聚类
k = 30
km = KMeans(n_clusters=k)
t0 = time()
km.fit(f_t_train)
tt = time() - t0
print("Clustered in {} seconds".format(round(tt, 3)))
# 检查集群大小
print(pd.Series(km.labels_).value_counts())
# 使用完整的标签集
labels_ten = kdd_data_10percent['label']
label_names = list(map(
    lambda x: pd.Series([labels_ten[i] for i in range(len(km.labels_)) if km.labels_[i] == x]),
    range(k)))
# 为每个集群打印标签
for i in range(k):
    print("Cluster {} labels:".format(i))
    print(label_names[i].value_counts())
# 预测使用我们的测试数据
t0 = time()
pred_km = km.predict(f_c_train)
tt = time() - t0
print("Assigned clusters in {} seconds".format(round(tt, 3)))
