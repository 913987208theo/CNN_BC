import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import random
from PIL import Image

# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 12 * 12)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# 数据预处理和归一化
transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 函数：添加噪声标签
def add_label_noise(dataset, noise_ratio=0.05):
    """
    对数据集的标签添加噪声
    :param dataset: 数据集 (ImageFolder 等)
    :param noise_ratio: 噪声比例，默认5%
    """
    # 获取狗和猫的标签索引
    dog_idx = dataset.class_to_idx['dogs']
    cat_idx = dataset.class_to_idx['cats']

    # 统计狗的样本索引
    dog_samples = [i for i, (_, label) in enumerate(dataset.imgs) if label == dog_idx]

    # 随机选择 5% 的狗样本
    noisy_dog_samples = random.sample(dog_samples, int(noise_ratio * len(dog_samples)))

    # 将这些狗样本的标签设置为猫
    for i in noisy_dog_samples:
        dataset.imgs[i] = (dataset.imgs[i][0], cat_idx)

    return dataset

# 加载训练数据集并添加噪声标签
print("加载训练数据集...")
train_data = ImageFolder(root='./cats_dogs/training_set', transform=transform)
train_data = add_label_noise(train_data, noise_ratio=0.05)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
print(f"训练数据集加载完成，数据集大小：{len(train_data)} 张图片")

# 加载测试数据集
print("加载测试数据集...")
test_data = ImageFolder(root='./cats_dogs/test_set', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
print(f"测试数据集加载完成，数据集大小：{len(test_data)} 张图片")

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型
model = CNN().to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 使用二分类交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 20
print(f"开始训练，共 {epochs} 个 epoch")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    print(f"第 {epoch + 1} 个 epoch 开始...")

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device).float().view(-1, 1)
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 记录损失和准确率
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if (i + 1) % 10 == 0:
            print(f"  进度：{i + 1}/{len(train_loader)} 批次，当前损失：{running_loss / (i + 1):.4f}")

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{epochs}] 完成，平均损失：{running_loss / total:.4f}，准确率：{accuracy:.2f}%")

# 保存模型
torch.save(model.state_dict(), 'cnn_model.pth')
print("模型已保存至 cnn_model.pth")

# 测试模型
def evaluate_model(test_loader, model):
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device).float().view(-1, 1)
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if (i + 1) % 10 == 0:
                print(f"  测试进度：{i + 1}/{len(test_loader)} 批次")

    accuracy = 100 * correct / total
    return accuracy

# 加载模型权重并测试
print("开始测试模型...")
model.load_state_dict(torch.load('cnn_model.pth'))
test_accuracy = evaluate_model(test_loader, model)
print(f"测试完成，测试集准确率：{test_accuracy:.2f}%")
