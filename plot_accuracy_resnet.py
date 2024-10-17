import matplotlib

matplotlib.use('Agg')  # 使用 'Agg' 后端，避免 GUI 相关问题
import matplotlib.pyplot as plt


# 定义读取accuracy_results.txt文件的函数
def read_results(filename):
    noise_ratios = []
    accuracies = []

    with open(filename, 'r') as f:
        # 跳过第一行（表头）
        next(f)

        # 逐行读取数据
        for line in f:
            if line.strip():  # 确保跳过空行
                noise_ratio, accuracy = line.split()
                noise_ratios.append(float(noise_ratio))  # 将噪声率转为浮点数
                accuracies.append(float(accuracy))  # 将准确率转为浮点数

    return noise_ratios, accuracies


# 读取结果
noise_ratios, accuracies = read_results('accuracy_results_resnet.txt')

# 生成折线图
plt.figure(figsize=(8, 6))
plt.plot(noise_ratios, accuracies, marker='o', linestyle='-', color='b', label='Test Accuracy')

# 设置图表标题和标签
plt.title('Test Accuracy vs. Noise Ratio', fontsize=14)
plt.xlabel('Noise Ratio (%)', fontsize=12)
plt.ylabel('Test Accuracy (%)', fontsize=12)

# 添加网格和图例
plt.grid(True)
plt.legend()

# 保存图表为图片文件
plt.savefig('test_accuracy_plot_resnet.png')  # 保存为 PNG 图片
