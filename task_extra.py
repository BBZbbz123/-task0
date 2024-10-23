# I AM I
import NET
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.testing._internal.common_quantization import accuracy as quant_accuracy
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO:解释参数含义，在?处填入合适的参数
batch_size =64 # 批处理大小
#batch_size指的是在训练神经网络时，一次训练过程中使用的样本数量。
#1：内存限制：较大的 batch_size 会占用更多的显存。如果显存不足，则需要减小 batch_size。
#2：训练稳定性和速度：较大的 batch_size 使得梯度估计更稳定，但每次参数更新的计算成本更高。较小的 batch_size 反之。
learning_rate =0.001 # 初始学习率
#learning_rate（学习率）是训练神经网络时需要调试的重要超参数。设置合适的学习率对模型的收敛速度和最终精度都有很大影响。
#调整学习率时，通常遵循以下几条经验法则：
#初始值选择：一个常见的初始值是 0.001 或 0.01 。
#观察梯度下降过程：如果在训练过程中发现损失函数下降得非常缓慢，可以尝试增大学习率；如果发现训练损失上下震荡或者发散（越来越大），则需要减小学习率。
#学习率衰减：在训练过程中逐渐减小学习率也是一种常见的方法，可以帮助模型更好地收敛。
num_epochs = 15 # 训练周期数

transform = transforms.Compose([
    transforms.ToTensor()
])

# root可以换为你自己的路径
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # TODO:这里补全你的网络层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 输入通道数为3（RGB），输出通道数为32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 输入通道数为32，输出通道数为64
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # 全连接层
        self.fc2 = nn.Linear(512, 10)  # 输出层，10个类
        pass

    def forward(self, x):
        # TODO:这里补全你的前向传播
        x = self.pool(F.relu(self.conv1(x)))  # 第一卷积层
        x = self.pool(F.relu(self.conv2(x)))  # 第二卷积层
        x = x.view(-1, 64 * 8 * 8)  # 展平
        x = F.relu(self.fc1(x))  # 第一全连接层
        x = self.fc2(x)  # 输出层
        return x
        pass

# TODO:补全
model = Network().to(device)
criterion =nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 用于存储loss和accuracy的数据
train_losses = []
train_accuracies = []

def train():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {epoch_accuracy:.2f}%')

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')

if __name__ == "__main__":
    train()
    test()


# 绘制损失和准确率曲线
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 6))
    # 绘制训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'r', label='Training loss')
    plt.title('Training loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # 绘制训练准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
    plt.title('Training accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
