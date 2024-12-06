from loguru import logger
import torch


model=torch.hub.load(".", "custom", path="./runs/train/exp12/weights/best.pt", source="local")

img=r'./testdetect.png'
reslut=model(img)
print(type(reslut))
#输出结果 类别，置信度，坐标
logger.info(reslut.pandas().xyxy[0])

reslut.show()