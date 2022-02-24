import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import classification_report, confusion_matrix

import os
#查看当前数据集里有多少数据文件
for dirname, _, filenames in os.walk('E:/被动冲击/迁移学习/程序/machine-learning-fault/archive'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from utils import create_directory
from utils import read_all_datasets
from utils import transform_labels

from constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES
from constants import UNIVARIATE_DATASET_NAMES as ALL_DATASET_NAMES
from constants import UNIVARIATE_ARCHIVE_NAMES  as ARCHIVE_NAMES
#导入训练数据集
datasets_dict = read_all_datasets('E:/被动冲击/迁移学习/data/data06','UCR_TS_Archive_2015')
for dataset_name in ALL_DATASET_NAMES:
			print('该文件夹内有以下文件：',dataset_name)
#选取哪个训练数据集
dataset_name_set='TG16_0704'
#提取训练集
x_train = datasets_dict[dataset_name_set][0]
y_train = datasets_dict[dataset_name_set][1]
#提取测试集
x_test = datasets_dict[dataset_name_set][-2]
y_test = datasets_dict[dataset_name_set][-1]
#总共有多少类
nb_classes = len(np.unique(np.concatenate((y_train,y_test),axis =0)))

# 将标签转换为数字标签
y_train,y_test = transform_labels(y_train,y_test)

                            
#svm


import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

results = []
names = []

#导入sklearn模块里的svc
from sklearn.svm import SVC

clf = SVC(C = 1, kernel='rbf',random_state=42)
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)
print(prediction)

from sklearn.metrics import accuracy_score

Accuracy = accuracy_score(prediction, y_test)
print('Accuracy with SVM classifier:\n',Accuracy*100)



print('confusion_matrix:\n', confusion_matrix(y_test, prediction))
print('Classification Report')
print(classification_report(y_test, prediction))
                                              
names.append('SVM')
results.append(Accuracy*100)

#画混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction)
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

index = ['Normal']  
columns = ['Normal']  
cm_df = pd.DataFrame(cm)  

plt.figure(figsize=(10,8))
sn.set(font_scale=1) # for label size
sn.heatmap(cm_df, annot=True, fmt='g') # font size
plt.title('Confusion matrix of SVM')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

plt.show()


#随机森林
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

clf = RandomForestClassifier(bootstrap=True, class_weight=None,
                             max_depth=30, max_features='auto', max_leaf_nodes=None,
                             min_samples_leaf=1,min_samples_split=2, min_weight_fraction_leaf=0.0,n_estimators=50, n_jobs=1,
                             oob_score=False, random_state=42,verbose=0, warm_start=False)

clf.fit(x_train, y_train)
#RandomForestClassifier(...)
prediction = (clf.predict(np.array(x_test)))
print(prediction)
          
Accuracy = accuracy_score(prediction, y_test)
print('Accuracy with RandomForestClassifier classifier:\n',Accuracy*100)
print('confusion_matrix:\n', confusion_matrix(y_test, prediction))
print('Classification Report')
print(classification_report(y_test, prediction))
names.append('RandomForestClassifier')
results.append(Accuracy*100)
#
depth = np.arange(1, 40)
err_list = []
for d in depth:
    clf = RandomForestClassifier(bootstrap=True, class_weight=None,
                             max_depth=d, max_features='auto', max_leaf_nodes=None,
                             min_samples_leaf=1,min_samples_split=2, min_weight_fraction_leaf=0.0,n_estimators=50, n_jobs=1,
                             oob_score=False, random_state=42,verbose=0, warm_start=False)

    clf.fit(x_train, y_train)
    y_test_hat = clf.predict(x_test)
    result = (y_test_hat == y_test)
    # 生成一个长度为验证集数量的数组，每一个元素是yhat和y是否相等的结果，
    print(list(result))
    if d == 1:
        print(result)
    #生成错误率
    err = 1 - np.mean(result)
    print(100 * err)
    err_list.append(err)
    print(d, ' 错误率：%.2f%%' % (100 * err))
plt.figure(facecolor='w')
plt.plot(depth, err_list, 'ro-', lw=2)
plt.xlabel('depth of DT', fontsize=15)
plt.ylabel('loss', fontsize=15)
plt.title('RF',fontsize=18)

plt.grid(True)
plt.show()
#画混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction)
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

index = ['Normal']  
columns = ['Normal']  
cm_df = pd.DataFrame(cm)  

plt.figure(figsize=(10,8))
sn.set(font_scale=1) # for label size
sn.heatmap(cm_df, annot=True, fmt='g') # font size
plt.title('Confusion matrix of RF')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

plt.show()
#XGBoost
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
import xgboost as xgb

clf = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
clf.fit(x_train, y_train)
#RandomForestClassifier(...)
prediction = (clf.predict(x_test))
print(prediction)
          
Accuracy = accuracy_score(prediction, y_test)
print('Accuracy with XGBoost classifier:\n',Accuracy*100)

print('confusion_matrix:\n', confusion_matrix(y_test, prediction))
print('Classification Report')
print(classification_report(y_test, prediction))

names.append('XGBoost')
results.append(Accuracy*100)

#画混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction)
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

index = ['Normal']  
columns = ['Normal']  
cm_df = pd.DataFrame(cm)  

plt.figure(figsize=(10,8))
sn.set(font_scale=1) # for label size
sn.heatmap(cm_df, annot=True, fmt='g') # font size
plt.title('Confusion matrix of XGBoost')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

plt.show()
#GradientBoostingClassifier
from sklearn.ensemble import  GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators = 150 , random_state = 42 )
clf.fit(x_train, y_train)
prediction = (clf.predict(x_test))
print(prediction)
          
Accuracy = accuracy_score(prediction, y_test)
print('Accuracy with GradientBoostingClassifier classifier:\n',Accuracy*100)

print('confusion_matrix:\n', confusion_matrix(y_test, prediction))
print('Classification Report')
print(classification_report(y_test, prediction))

names.append('GradientBoostingClassifier')
results.append(Accuracy*100)
#画混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction)
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

index = ['Normal']  
columns = ['Normal']  
cm_df = pd.DataFrame(cm)  

plt.figure(figsize=(10,8))
sn.set(font_scale=1) # for label size
sn.heatmap(cm_df, annot=True, fmt='g') # font size
plt.title('Confusion matrix of GradientBoostingClassifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

plt.show()
#KNN
from sklearn.neighbors import KNeighborsClassifier 

clf = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
clf.fit(x_train, y_train)
prediction = (clf.predict(x_test))
print(prediction)
          
Accuracy = accuracy_score(prediction, y_test)
print('Accuracy with KNN classifier:\n',Accuracy*100)

print('confusion_matrix:\n', confusion_matrix(y_test, prediction))
print('Classification Report')
print(classification_report(y_test, prediction))

names.append('KNN')
results.append(Accuracy*100)

#画混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction)
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

index = ['Normal']  
columns = ['Normal']  
cm_df = pd.DataFrame(cm)  

plt.figure(figsize=(10,8))
sn.set(font_scale=1) # for label size
sn.heatmap(cm_df, annot=True, fmt='g') # font size
plt.title('Confusion matrix of KNN')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

plt.show()
#决策树
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth = 48 , random_state = 42 )
clf.fit(x_train, y_train)
prediction = (clf.predict(x_test))
print(prediction)
          
Accuracy = accuracy_score(prediction, y_test)
print('Accuracy with DecisionTreeClassifier classifier:\n',Accuracy*100)

print('confusion_matrix:\n', confusion_matrix(y_test, prediction))
print('Classification Report')
print(classification_report(y_test, prediction))

names.append('DecisionTreeClassifier')
results.append(Accuracy*100)

#PCA+LR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#标准化单位方差
sc=StandardScaler()
x_train_std=sc.fit_transform(x_train)
x_test_std=sc.fit_transform(x_test)

pca=PCA(n_components=120)
lr = LogisticRegression()
x_train_pca = pca.fit_transform(x_train_std)
x_test_pca = pca.fit_transform(x_test_std)  # 预测时候特征向量正负问题，乘-1反转镜像
lr.fit(x_train_pca, y_train)

plt.figure(figsize=(6, 7), dpi=100)  # 画图高宽，像素
plt.subplot(2, 1, 1)
plot_decision_regions(x_train_pca, y_train, classifier=lr)
plt.title('Training Result')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
 
plt.subplot(2, 1, 2)
plot_decision_regions(x_test_pca, y_test, classifier=lr)
plt.title('Testing Result')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.tight_layout()  # 子图间距
plt.show()

prediction = (lr.predict(x_test_pca))
print(prediction)
          
Accuracy = accuracy_score(prediction, y_test)
print('Accuracy with DecisionTreeClassifier classifier:\n',Accuracy*100)

print('confusion_matrix:\n', confusion_matrix(y_test, prediction))
print('Classification Report')
print(classification_report(y_test, prediction)) 


from sklearn.neighbors import KNeighborsClassifier 

knn = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
knn.fit(x_train_pca, y_train)
prediction = (knn.predict(x_test_pca))
print(prediction)
          
Accuracy = accuracy_score(prediction, y_test)
print('Accuracy with KNN classifier:\n',Accuracy*100)

print('confusion_matrix:\n', confusion_matrix(y_test, prediction))
print('Classification Report')
print(classification_report(y_test, prediction))

names.append('KNN')
results.append(Accuracy*100)


#生成一个数组
depth = np.arange(1, 15)
err_list = []
for d in depth:
    clf = DecisionTreeClassifier(criterion='gini', max_depth=d)
    clf.fit(x_train, y_train)
    y_test_hat = clf.predict(x_test)
    result = (y_test_hat == y_test)
    # 生成一个长度为验证集数量的数组，每一个元素是yhat和y是否相等的结果，
    print(list(result))
    if d == 1:
        print(result)
    #生成错误率
    err = 1 - np.mean(result)
    print(100 * err)
    err_list.append(err)
    print(d, ' 错误率：%.2f%%' % (100 * err))
plt.figure(facecolor='w')
plt.plot(depth, err_list, 'ro-', lw=2)
plt.xlabel('depth of DT', fontsize=15)
plt.ylabel('loss', fontsize=15)
plt.title('DT',fontsize=18)

plt.grid(True)
plt.show()



#画混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction)
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

index = ['Normal']  
columns = ['Normal']  
cm_df = pd.DataFrame(cm)  

plt.figure(figsize=(10,8))
sn.set(font_scale=1) # for label size
sn.heatmap(cm_df, annot=True, fmt='g') # font size
plt.title('Confusion matrix of DecisionTree')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

plt.show()
#对前面算法进行总结
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

name_df = pd.DataFrame(names) 
result_df = pd.DataFrame(results) 

name_df['Accuracy'] = result_df
name_df.rename(columns={ 0: 'Algorithm'
                   ,}, 
                 inplace=True)

print(name_df)
#print(result_df)


fig = px.line(name_df,  x='Algorithm', y='Accuracy')
fig.update_layout(yaxis_range = [0.00, 100.00],
                  title_text="Algorithm Comparison")
fig.show()