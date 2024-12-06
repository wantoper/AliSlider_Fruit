#encoding:utf-8

from loguru import logger
from base64 import b64decode
import numpy as np
import cv2
from PIL import Image
from ddddocr import DdddOcr

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

    #二值化
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


    cv2.imshow('original', result)
    cv2.imshow('img', result)
    ocr_res = ocr.classification(Image.fromarray(result)).replace("期蝶","蝴蝶").replace("瓤虫","瓢虫").replace("革果","苹果").replace("字航员","宇航员")
    logger.info(f"识别结果：{ocr_res}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return ocr_res

if __name__=="__main__":
    #https://scportal.taobao.com/quali_show.htm?spm=a1z10.1-c-s.0.0.34175249ZXLZDr&uid=2206833789551&qualitype=1
    ocr = DdddOcr(show_ad=False)
    img1="data:image/jpeg;base64,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"
    img2="data:image/jpeg;base64,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"
    img1 = b64decode(img1.split(",")[1])
    img2 = b64decode(img2.split(",")[1])
    deal_que_new_img(img1,img2)


