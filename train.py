import torch
import torch.nn as nn
import torch.optim as optim #用于优化模型参数的模块，提供各种优化器（如 Adam、SGD 等）。
import torchvision.transforms as transforms    #提供图像的预处理和数据增强功能
from torchvision.datasets import ImageFolder  #用于从目录加载图像数据集，并根据文件夹名称自动分类。
from torch.utils.data import DataLoader  #将数据集打包成小批次，并支持并行数据加载和打乱数据。


# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 12 * 12, 128) #全连接层，将卷积层提取的特征映射到 128 维的特征空间。
        self.fc2 = nn.Linear(128, 1)  #输出层，用于二分类任务，输出 1 个值（表示分类结果是猫或狗）。

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 12 * 12)  #将池化后的特征展平（view 方法将张量拉平成一维），以输入全连接层。
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  #通过全连接层进行分类，最终输出通过 sigmoid 激活函数。
                                        # 输出一个介于 0 到 1 的概率值，用于二分类。
        return x


# 数据增强和归一化
transform = transforms.Compose([  #transforms.Compose：将多个图像处理步骤按顺序组合起来。
    transforms.Resize((50, 50)),
    transforms.RandomHorizontalFlip(),  #随机水平翻转图像，用于数据增强。
    transforms.RandomAffine(degrees=0, shear=0.2),  #对图像进行随机剪切变换，增加数据的多样性。
    transforms.ToTensor(),  #将图像数据转换为 PyTorch 的张量格式。
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  #将图像的 RGB 值归一化到 [-1, 1] 范围（归一化后的值 = (原始值 - 0.5) / 0.5）。
])

# 加载训练数据集
print("加载训练数据集...")
train_data = ImageFolder(root='./cats_dogs/training_set', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
print(f"训练数据集加载完成，数据集大小：{len(train_data)} 张图片")

# 创建模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)  #将模型移动到指定设备（GPU 或 CPU）上。
criterion = nn.BCELoss()  # 二分类交叉熵损失函数（因为是二分类任务）。
optimizer = optim.Adam(model.parameters(), lr=0.001)  #使用 Adam 优化器，学习率为 0.001，优化模型的参数。


# 训练模型
epochs = 20
print(f"开始训练，共 {epochs} 个 epoch")

for epoch in range(epochs):
    model.train()  #将模型设置为训练模式（启用 dropout 和 batch normalization）
    running_loss = 0.0  #running_loss、correct、total 用于记录每个 epoch 的损失和准确率。
    correct = 0
    total = 0
    print(f"第 {epoch + 1} 个 epoch 开始...")

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device).float().view(-1, 1) #将图像和标签移动到指定设备（GPU 或 CPU）上。
        '''images 会被传输到 GPU 或 CPU，形状仍然是 [32, 3, 50, 50]。 32 表示批次大小，3 表示 RGB 三个通道，50x50 表示图像的大小
            labels 会被传输到 GPU 或 CPU，并转换为浮点数，同时形状变为 [32, 1]，即每个样本有一个单独的标签列。'''
        optimizer.zero_grad()  #清空上一轮的梯度。
        '''这是在每次处理一个小批次的数据时，都会先调用 optimizer.zero_grad() 来清空之前累积的梯度。'''

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)  #计算模型的损失值，比较输出与真实标签。

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()  #累加以计算整个 epoch 的总损失，以便在整个 epoch 结束时计算平均损失。
        predicted = (outputs > 0.5).float()  #预测狗和猫的概率，并转换为浮点数。
        correct += (predicted == labels).sum().item() #计算预测正确的数量。
        total += labels.size(0)  #计算当前批次的总样本数。

        if (i + 1) % 10 == 0:
            print(f"  进度：{i + 1}/{len(train_loader)} 批次，当前损失：{running_loss / (i + 1):.4f}")

    accuracy = 100 * correct / total  #计算！当前！ epoch 的准确率。
    print(f"Epoch [{epoch + 1}/{epochs}] 完成，平均损失：{running_loss / total:.4f}，准确率：{accuracy:.2f}%")

# 保存模型
torch.save(model.state_dict(), 'cnn_model.pth') #model.state_dict()：返回模型的状态字典，包含模型的所有参数（如权重和偏置）。
print("模型已保存至 cnn_model.pth")
