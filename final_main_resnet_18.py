import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import random
import torchvision.models as models
import torch.nn.functional as F

# 定义 ResNet-18 模型
def create_model():
    model = models.resnet18(weights=None)  # 使用最新的API，替代 pretrained=False
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid()  # 在最后一层添加 Sigmoid 激活函数
    )
    return model

# 数据预处理和归一化
transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 函数：添加噪声标签
def add_label_noise(dataset, noise_ratio):
    """
    对数据集的标签添加噪声
    :param dataset: 数据集 (ImageFolder 等)
    :param noise_ratio: 噪声比例，0-1之间
    """
    dog_idx = dataset.class_to_idx['dogs']
    cat_idx = dataset.class_to_idx['cats']

    dog_samples = [i for i, (_, label) in enumerate(dataset.imgs) if label == dog_idx]
    noisy_dog_samples = random.sample(dog_samples, int(noise_ratio * len(dog_samples)))

    for i in noisy_dog_samples:
        dataset.imgs[i] = (dataset.imgs[i][0], cat_idx)

    return dataset

# 测试模型
def evaluate_model(test_loader, model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float().view(-1, 1)
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return accuracy

# 主程序
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_data = ImageFolder(root='./cats_dogs/test_set', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 创建文件用于保存测试结果，并写入表头
with open("accuracy_results_resnet_18.txt", "w") as f:
    f.write("Noise Ratio (%) \t Test Accuracy (%)\n")

# 噪声率从0%到50%，每次增加1%
for noise_ratio in range(0, 51, 1):
    noise_ratio /= 100  # 将百分比转换为小数

    print(f"\n噪声率：{noise_ratio * 100}%")

    # 加载训练数据集并添加噪声标签
    train_data = ImageFolder(root='./cats_dogs/training_set', transform=transform)
    train_data = add_label_noise(train_data, noise_ratio)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # 创建 ResNet-18 模型
    model = create_model().to(device)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    epochs = 2
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().view(-1, 1)
            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{epochs}] 完成，平均损失：{running_loss / total:.4f}，准确率：{accuracy:.2f}%")

    # 测试模型
    test_accuracy = evaluate_model(test_loader, model)
    print(f"噪声率 {noise_ratio * 100}% 测试集准确率：{test_accuracy:.2f}%")

    # 将测试结果写入文件
    with open("accuracy_results_resnet_18.txt", "a") as f:
        f.write(f"{noise_ratio * 100:.2f}\t\t{test_accuracy:.2f}\n")
