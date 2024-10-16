import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

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

# 数据归一化
transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 加载测试数据集
print("加载测试数据集...")
test_data = ImageFolder(root='./cats_dogs/test_set', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
print(f"测试数据集加载完成，数据集大小：{len(test_data)} 张图片")

# 创建模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

# 加载保存的模型权重
print("加载模型权重...")
model.load_state_dict(torch.load('cnn_model.pth'))
print("模型权重加载完成")

# 测试模型
print("开始测试模型...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device).float().view(-1, 1)
        outputs = model(images)
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()  #.item() 方法用于将一个只包含一个元素的张量转换为 Python 的标量值（例如 float 或 int）。
        total += labels.size(0)

        if (i+1) % 10 == 0:
            print(f"  测试进度：{i+1}/{len(test_loader)} 批次")

accuracy = 100 * correct / total
print(f"测试完成，测试集准确率：{accuracy:.2f}%")
