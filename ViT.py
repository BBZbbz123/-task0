# I AM I
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import timm
import time
from tqdm import tqdm
import csv

# 检查CUDA是否可用
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')

def train(model, train_loader, criterion, optimizer, device, pbar=None):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if pbar is not None:
            pbar.set_postfix(loss=running_loss / len(train_loader))
    return running_loss / len(train_loader)


def test(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


if __name__ == '__main__':
    # 数据集预处理和加载
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 加载 CIFAR-10 数据集
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # 定义ViT模型
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #开始实验并记录结果
    # 训练和测试模型
    log_file = 'experiment_log.csv'
    num_epochs = 10
    results = []

    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Test Accuracy', 'Epoch Time'])

    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}]')

        # 记录开始时间
        start_time = time.time()

        pbar = tqdm(train_loader, total=len(train_loader), desc="Training")
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_accuracy = test(model, test_loader, criterion, device)

        # 打印一个周期所需时间
        epoch_time = time.time() - start_time
        print(f"Time for one epoch: {epoch_time:.2f} seconds")

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {test_accuracy:.2f}%')

        # 记录结果
        results.append([epoch + 1, train_loss, test_accuracy, epoch_time])
        writer.writerow([epoch + 1, train_loss, test_accuracy, epoch_time])

        # 估计总时间
        # 汇总实验结果
        total_time = sum([result[3] for result in results])
        final_accuracy = results[-1][2]
        print(f"Total training time: {total_time / 60:.2f} minutes")
        print(f"Final test accuracy: {final_accuracy:.2f}%")
