import cv2
# from cv2 import cv2 
import numpy as np
import random

path = "D:/2Codefield/VS_code/python/Learn_Base/openCV/DATA/"
lenasrc = "Lena/"
baboonsrc = "Baboon/"
filepath = path + baboonsrc


# 等比例k修改灰度像素大小
def AlgebraChangeScale(img, k):
    last = np.ceil(img * k)  #对每个像素等比例k，向上取整
    last = last.astype(np.uint8)  #改为uint8(8位无符号数)的图片矩阵
    return last

# 二值化
def Thresholding(img, k):
    last = []
    for row in range(img.shape[0]):
        last.append([0 if x < k else 255 for x in img[row]])  #对每行进行操作
    last = np.array(last).astype(np.uint8)  #将list转为np.array后改类型为uint8
    return last

# 抽取RGB颜色
def extract_color(img, channel):
    last = []
    for row in range(img.shape[0]):
        if channel == 0:
            last.append([[x[0], 0, 0] for x in img[row]])  #对每行进行操作
        elif channel == 1:
            last.append([[0, x[1], 0] for x in img[row]])
        elif channel == 2:
            last.append([[0, 0, x[2]] for x in img[row]])
        elif channel == 3:
            last.append([x[2] for x in img[row]])
    last = np.array(last).astype(np.uint8)  #将list转为np.array后改类型为uint8
    return last

# 展示
def show(A, name):
    print(A)
    cv2.imshow(name, A)  #展示图片
    cv2.waitKey(0)  #延时显示
    cv2.imwrite(filepath + name + ".png", A)  #保存


if __name__ == '__main__':
    color_img = cv2.imread(filepath + "color.png")
    gray_img = cv2.imread(filepath + "color.png", cv2.IMREAD_GRAYSCALE)  #提取灰度图
    
    show(color_img, 'color')
    show(extract_color(color_img, 0), 'blue')
    show(extract_color(color_img, 1), 'green')
    show(extract_color(color_img, 2), 'red')
    # show(extract_color(color_img, 3), 'redgray')

    show(gray_img, 'gray')
    show(AlgebraChangeScale(gray_img, 0.5), 'scale')
    show(Thresholding(gray_img, 127), 'Binarization')
