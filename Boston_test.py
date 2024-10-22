# I AM I
import torch

#data 解析数据
import numpy as np
import re
ff = open("BostonHousing.csv").readlines()  #读取所有的行形成一个所有行列表,注意后面有s
# print(ff)
data = [] #定义一个data列表，将数据加入其中
for item in ff[1:]:
    out = re.sub(r"\s{2,}", " ", item).strip()  #将多个空格合并成一个并且去掉其中的换行符
    #print(out)
    data.append(out.split(" ")) #将数据以空格分开并添加到data列表中

    # 清理数据，移除空字符串
    cleansed_data = []
    for row in data:
        cleansed_row = []
        for element in row:
            if element == '':
                cleansed_row.append('0')  # 如果遇到空字符串，将其设置为0
            else:
                cleansed_row.append(element)
        cleansed_data.append(cleansed_row)
# 更新 data 为清理后的数据
data = cleansed_data

data = np.array(data).astype(np.float_) #将数据转换成np矩阵，数据类型为float
#print(data)
print(data.shape)
Y = data[:, -1].reshape(506, 1)
X = data[:, 0:-1].reshape(506, 13)

Y_train = Y[0:496, :]
X_train = X[0:496, :]

Y_test = Y[496:, :]
X_test = X[496:, :]
print(Y_test.shape)
print(X_test.shape)

#net 搭建网络
class Net(torch.nn.Module):  #继承nn.Module
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, 100)
        self.predict = torch.nn.Linear(100, n_output)  #线性回归模型
    def forward(self, x):
        out = self.hidden(x)
        out = torch.relu(out)
        out = self.predict(out)

        return out

net = torch.load("model/model.pkl")  **#加载模型**
#loss 定义损失函数
loss_func = torch.nn.MSELoss()  #均方损失

#test 测试模型
x_data = torch.tensor(X_test, dtype=torch.float32)
y_data = torch.tensor(Y_test, dtype=torch.float32)
pred = net.forward(x_data)  #计算预测值
loss_test = loss_func(pred, y_data)*0.001  #计算loss
print("loss_test:{}".format(loss_test))

