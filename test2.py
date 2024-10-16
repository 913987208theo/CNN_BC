import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# 定义与训练模型时相同的 CNN 模型
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


# 定义图片的预处理步骤，保持与训练时一致
transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# 加载模型
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 切换到评估模式
    return model


# 预测图像类别
def predict_image(image_path, model):
    # 加载图片并进行预处理
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # 添加 batch 维度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)

    # 禁用梯度计算
    with torch.no_grad():
        output = model(image)
        prediction = (output > 0.5).float()

    # 输出预测结果
    if prediction.item() == 1:
        return "狗"
    else:
        return "猫"


if __name__ == "__main__":
    # 模型文件路径，确保与 train.py 保存的模型文件一致
    model_path = 'cnn_model.pth'

    # 图片路径，替换为你要预测的图片路径
    image_path = 'b_cat.jpg'

    # 加载模型
    model = load_model(model_path)

    # 预测图片类别
    result = predict_image(image_path, model)

    # 输出结果
    print(f"图片预测结果：{result}")
