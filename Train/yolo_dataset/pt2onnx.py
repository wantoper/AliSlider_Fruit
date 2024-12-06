from loguru import logger

import torch
import torchvision
from models.experimental import attempt_load
# 加载模型权重
model = attempt_load('runs/train/exp12/weights/best.pt', map_location=torch.device('cpu'))
# 设置模型为评估模式
model.eval()
# 准备一个示例输入
input_tensor = torch.randn(1, 3, 640, 640)  # 假设输入图像大小为 640x640
# 导出模型
#Lib\site-packages\torch\nn\modules\ activation.py
torch.onnx.export(model, input_tensor, 'AliFruit.onnx',opset_version=11)