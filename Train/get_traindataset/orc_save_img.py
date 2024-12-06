import time

from loguru import logger
import re
import requests
import random
from base64 import b64decode
import numpy as np
import cv2
from PIL import Image
from ddddocr import DdddOcr

ocr = DdddOcr()

# 领域噪声处理
def is_noise(pixel, img, window_size=2, deal=5):
    h, w = img.shape[:2]
    half_window = window_size // 2

    # 计算邻域范围
    y_start = max(0, pixel[0] - half_window)
    y_end = min(h, pixel[0] + half_window + 1)
    x_start = max(0, pixel[1] - half_window)
    x_end = min(w, pixel[1] + half_window + 1)

    # 提取邻域
    neighborhood = img[y_start:y_end, x_start:x_end]

    # 计算非零像素数量
    non_zero_count = np.count_nonzero(neighborhood)

    # 减去中心像素自身
    if np.any(neighborhood[half_window, half_window]) != 0:
        non_zero_count -= 1

    # 判断是否是噪声
    if non_zero_count <= deal:  # 调整这里的阈值以适应不同类型的噪声
        return True
    else:
        return False

def deal_que_new_img(bin_image,bin_image2):
    # 将二进制数据转换为NumPy的字节数组
    nparr = np.frombuffer(bin_image, np.uint8)
    nparr2 = np.frombuffer(bin_image2, np.uint8)

    original = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    original2 = cv2.imdecode(nparr2, cv2.IMREAD_GRAYSCALE)

    height, width = original.shape[:2]
    x1 = 144
    x2 = 0
    for x in range(width - 1, 144,-1):
        if original[0, x] != original[0, x-1]:
            x2 = x
            break

    _, original = cv2.threshold(original, 205, 255, cv2.THRESH_BINARY)
    _, original2 = cv2.threshold(original2, 205, 255, cv2.THRESH_BINARY)

    original = original[0:height, x1-1:x2+1]
    original2 = original2[0:height, x1-1:x2+1]

    #同时遍历两张图片，找到不同的像素点 删除掉
    for y in range(original.shape[0]):
        for x in range(original.shape[1]):
            if np.any(original[y, x]) == 0 and np.any(original2[y, x]) == 0:
                original[y, x] = 0
                original2[y, x] = 0
            else:
                original[y, x] = 255
                original2[y, x] = 255

    result = np.copy(original)
    for y in range(original.shape[0]):
        for x in range(original.shape[1]):
            if np.any(original[y, x]) == 0 and not is_noise((y, x), original, deal=7):
                result[y, x] = 255  # 将判断为噪声的像素置为0


    # cv2.imshow('original', result)
    # cv2.imshow('img', result)
    ocr_res = ocr.classification(Image.fromarray(result)).replace("期蝶","蝴蝶").replace("瓤虫","瓢虫").replace("革果","苹果")
    logger.info(f"识别结果：{ocr_res}")

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return ocr_res


num_flag = 0

headers = {
    "accept": "*/*",
    "accept-language": "zh-CN,zh;q=0.9",
    "cache-control": "no-cache",
    "pragma": "no-cache",
    "priority": "u=1, i",
    "referer": "https://scportal.taobao.com/quali_show.htm?spm=a1z10.1-c-s.0.0.34175249ZXLZDr&uid=2206833789551&qualitype=1",
    "sec-ch-ua": "\"Not/A)Brand\";v=\"8\", \"Chromium\";v=\"126\", \"Google Chrome\";v=\"126\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
}
session = requests.session()
session.headers = headers

num = 1000
for i in range(num):
    res = session.get('https://scportal.taobao.com/quali_show.htm?spm=a1z10.1-c-s.0.0.34175249ZXLZDr&uid=2206833789551&qualitype=1').text

    NCAPPKEY = re.findall('"NCAPPKEY": "(.*?)",', res)[0]
    SECDATA = re.findall('"SECDATA": "(.*?)",', res)[0]
    NCTOKENSTR = re.findall('"NCTOKENSTR": "(.*?)",', res)[0]

    url = "https://scportal.taobao.com/quali_show.htm/_____tmd_____/newslidecaptcha"
    params = {
        "token": NCTOKENSTR,
        "appKey": NCAPPKEY,
        "x5secdata": SECDATA,
        "language": "cn",
        "v": "00736788154383{}".format(random.randint(1000, 10000))
    }
    res = session.get(url, params=params).json()['data']

    b64imgs=res['ques'].split('|')[1:]
    que_img = b64decode(b64imgs[0].split('base64,')[-1])
    que_img1 = b64decode(b64imgs[1].split('base64,')[-1])
    que = deal_que_new_img(que_img,que_img1)
    que = que.split("个")[-1]

    with open('labels.txt', 'r', encoding='utf-8') as f:
        if que not in f.read():
            with open('labels.txt', 'a', encoding='utf-8') as f:
                f.write(que + '\n')
    back_img = b64decode(res['imageData'].split('base64,')[-1])
    with open(f'origin_img/{int(time.time() * 10000)}.png', 'wb') as f:
        f.write(back_img)