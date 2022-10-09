import cv2
import numpy as np
import random
import matplotlib.pyplot as plt 

path = "D:/2Codefield/VS_code/python/Learn_Base/openCV/DATA/"
lenasrc = "Lena/"
baboonsrc = "Baboon/"
logosrc = "Logo/"
# 读取彩色图片和灰度图片
color_img1 = cv2.imread(path + lenasrc + "color.png", flags=1)  # 读取彩色图像(BGR)
gray_img1 = cv2.imread(path + lenasrc + "color.png", flags=0)  # 读取为灰度图像
color_img2 = cv2.imread(path + baboonsrc + "color.png", flags=1)
gray_img2 = cv2.imread(path + baboonsrc + "color.png", flags=0)
color_img3 = cv2.imread(path + logosrc + "color.png", flags=1)
color_img3_small = cv2.imread(path + logosrc + "color128.png", flags=1)


##########################################################################
# 2.1 图像的加法 (cv2.add)
imgAddCV = cv2.add(color_img1, color_img2)  # OpenCV 加法: 饱和运算(偏白)
imgAddNP = color_img1 + color_img2  # # Numpy 加法: 模运算(偏黑)
# 显示
plt.subplot(221), plt.title("1. img1"), plt.axis('off')
plt.imshow(cv2.cvtColor(color_img1, cv2.COLOR_BGR2RGB))  # 显示 img1(RGB)
plt.subplot(222), plt.title("2. img2"), plt.axis('off')
plt.imshow(cv2.cvtColor(color_img2, cv2.COLOR_BGR2RGB))  # 显示 img2(RGB)
plt.subplot(223), plt.title("3. cv2.add(img1, img2)"), plt.axis('off')
plt.imshow(cv2.cvtColor(imgAddCV, cv2.COLOR_BGR2RGB))  # 显示 imgAddCV(RGB)
plt.subplot(224), plt.title("4. img1 + img2"), plt.axis('off')
plt.imshow(cv2.cvtColor(imgAddNP, cv2.COLOR_BGR2RGB))  # 显示 imgAddNP(RGB)
plt.show()


##########################################################################
# 2.2 图像的加法 (与标量相加)
Value = 100  # 常数
Scalar = np.ones((1, 3), dtype="float") * Value  # 标量
imgAddV = cv2.add(color_img1, Value)  # OpenCV 加法: 图像 + 常数(加在B通道，偏蓝)
imgAddS = cv2.add(color_img1, Scalar)  # OpenCV 加法: 图像 + 标量(每个通道都加，偏白)
# 打印pixal实例
print("Shape of scalar", Scalar)
for i in range(1, 6):
    x, y = i*10, i*10
    print("(x,y)={},{}, color_img1:{}, imgAddV:{}, imgAddS:{}"
            .format(x,y,color_img1[x,y],imgAddV[x,y],imgAddS[x,y]))
# 显示
plt.subplot(131), plt.title("1. img1"), plt.axis('off')
plt.imshow(cv2.cvtColor(color_img1, cv2.COLOR_BGR2RGB))  # 显示 img1(RGB)
plt.subplot(132), plt.title("2. img + constant"), plt.axis('off')
plt.imshow(cv2.cvtColor(imgAddV, cv2.COLOR_BGR2RGB))  # 显示 imgAddV(RGB)
plt.subplot(133), plt.title("3. img + scalar"), plt.axis('off')
plt.imshow(cv2.cvtColor(imgAddS, cv2.COLOR_BGR2RGB))  # 显示 imgAddS(RGB)
plt.show()


##########################################################################
# 2.3 图像的混合(加权加法)
imgAddW1 = cv2.addWeighted(color_img1, 0.2, color_img2, 0.8, 0)  # 加权相加, a=0.2, b=0.8
imgAddW2 = cv2.addWeighted(color_img1, 0.5, color_img2, 0.5, 0)  # 加权相加, a=0.5, b=0.5
imgAddW3 = cv2.addWeighted(color_img1, 0.8, color_img2, 0.2, 0)  # 加权相加, a=0.8, b=0.2
# 显示
plt.subplot(131), plt.title("1. a=0.2, b=0.8"), plt.axis('off')
plt.imshow(cv2.cvtColor(imgAddW1, cv2.COLOR_BGR2RGB))  # 显示 img1(RGB)
plt.subplot(132), plt.title("2. a=0.5, b=0.5"), plt.axis('off')
plt.imshow(cv2.cvtColor(imgAddW2, cv2.COLOR_BGR2RGB))  # 显示 imgAddV(RGB)
plt.subplot(133), plt.title("3. a=0.8, b=0.2"), plt.axis('off')
plt.imshow(cv2.cvtColor(imgAddW3, cv2.COLOR_BGR2RGB))  # 显示 imgAddS(RGB)
plt.show()


##########################################################################
# 2.4 不同尺寸的图像加法
x,y = 300,50  # 叠放位置
W1, H1 = color_img1.shape[1::-1]  # 大图尺寸
W2, H2 = color_img3_small.shape[1::-1]  # 小图尺寸
if (x + W2) > W1: x = W1 - W2  # 调整图像叠放位置，避免溢出
if (y + H2) > H1: y = H1 - H2
# 叠加
imgCrop = color_img1[y:y + H2, x:x + W2]  # 裁剪大图，与小图 imgS 的大小相同
imgAdd = cv2.add(imgCrop, color_img3_small)  # cv2 加法，裁剪图与小图叠加
imgAddW = cv2.addWeighted(imgCrop, 0.2, color_img3_small, 0.8, 0)  # 加权加法，裁剪图与小图叠加
# 替换
imgAddM = np.array(color_img1)
imgAddM[y:y + H2, x:x + W2] = imgAddW  # 用叠加小图替换原图 imgL 的叠放位置
# 显示
cv2.imshow("imgAdd", imgAdd)
cv2.imshow("imgAddW", imgAddW)
cv2.imshow("imgAddM", imgAddM)
cv2.waitKey(0)


##########################################################################
# 2.5 图像的掩模加法 (image mask)
Mask = np.zeros((color_img1.shape[0], color_img1.shape[1]), dtype=np.uint8)  # 返回与图像 img1 尺寸相同的全零数组
xmin, ymin, w, h = 180, 190, 200, 200  # 矩形裁剪区域 (ymin:ymin+h, xmin:xmin+w) 的位置参数
Mask[ymin:ymin+h, xmin:xmin+w] = 255  # 掩模图像，ROI 为白色，其它区域为黑色
print(color_img1.shape, color_img2.shape, Mask.shape)
# 函数 cv2.add 进行加法运算，对被掩模图像遮蔽的黑色区域不进行处理，保持黑色。
imgAddMask1 = cv2.add(color_img1, color_img2, mask=Mask)  # 带有掩模 mask 的加法
imgAddMask2 = cv2.add(color_img1, np.zeros(np.shape(color_img1), dtype=np.uint8), mask=Mask)  # 提取 ROI
# 显示
cv2.imshow("MaskImage", Mask)  # 显示掩模图像 Mask
cv2.imshow("MaskAdd", imgAddMask1)  # 显示掩模加法结果 imgAddMask1
cv2.imshow("MaskROI", imgAddMask2)  # 显示从 img1 提取的 ROI
key = cv2.waitKey(0)  # 等待按键命令


##########################################################################
# 2.6 图像的其他形状的掩模加法 (圆形和椭圆的遮罩)
Mask1 = np.zeros((color_img1.shape[0], color_img1.shape[1]), dtype=np.uint8)  # 返回与图像 img1 尺寸相同的全零数组
Mask2 = Mask1.copy()
cv2.circle(Mask1, (285, 285), 110, (255, 255, 255), -1)  # -1 表示实心
cv2.ellipse(Mask2, (285, 285), (100, 150), 0, 0, 360, 255, -1)  # -1 表示实心

imgAddMask1 = cv2.add(color_img1, np.zeros(np.shape(color_img1), dtype=np.uint8), mask=Mask1)  # 提取圆形 ROI
imgAddMask2 = cv2.add(color_img1, np.zeros(np.shape(color_img1), dtype=np.uint8), mask=Mask2)  # 提取椭圆 ROI

cv2.imshow("circularMask", Mask1)  # 显示掩模图像 Mask
cv2.imshow("circularROI", imgAddMask1)  # 显示掩模加法结果 imgAddMask1
cv2.imshow("ellipseROI", imgAddMask2)  # 显示掩模加法结果 imgAddMask2
key = cv2.waitKey(0)  # 等待按键命令


##########################################################################
# 2.7 图像的位操作
imgAnd = cv2.bitwise_and(color_img1, color_img2)  # 按位 与(AND)
imgOr = cv2.bitwise_or(color_img1, color_img2)  # 按位 或(OR)
imgNot = cv2.bitwise_not(color_img1)  # 按位 非(NOT)
imgXor = cv2.bitwise_xor(color_img1, color_img2)  # 按位 异或(XOR)

plt.figure(figsize=(9,6))
titleList = ["color_img1", "color_img2", "and", "or", "not", "xor"]
imageList = [color_img1, color_img2, imgAnd, imgOr, imgNot, imgXor]
for i in range(6):
    plt.subplot(2,3,i+1), plt.title(titleList[i]), plt.axis('off')
    plt.imshow(cv2.cvtColor(imageList[i], cv2.COLOR_BGR2RGB), 'gray')
plt.show()


##########################################################################
# 2.8 图像的叠加
x, y = (0, 10)  # 图像叠加位置
W1, H1 = color_img1.shape[1::-1]
W2, H2 = color_img3_small.shape[1::-1]
if (x + W2) > W1: x = W1 - W2
if (y + H2) > H1: y = H1 - H2
print(W1,H1,W2,H2,x,y)
imgROI = color_img1[x:x+W2, y:y+H2]  # 从背景图像裁剪出叠加区域图像
# 创建img3灰度图和遮罩
img3Gray = cv2.cvtColor(color_img3_small, cv2.COLOR_BGR2GRAY)  # color_img3_small: 转换为灰度图像
ret, mask = cv2.threshold(img3Gray, 200, 255, cv2.THRESH_BINARY)  # 转换为二值图像，生成遮罩，LOGO 区域黑色遮盖
maskInv = cv2.bitwise_not(mask)  # 按位非(黑白转置)，生成逆遮罩，LOGO 区域白色开窗，LOGO 以外区域黑色
# mask 黑色遮盖区域输出为黑色，mask 白色开窗区域与运算（原图像素不变）
img1Bg = cv2.bitwise_and(imgROI, imgROI, mask=mask)  # 生成背景，imgROI 的遮罩区域输出黑色
img2Fg = cv2.bitwise_and(color_img3_small, color_img3_small, mask=maskInv)  # 生成前景，LOGO 的逆遮罩区域输出黑色
imgROIAdd = cv2.add(img1Bg, img2Fg)  # 前景与背景合成，得到裁剪部分的叠加图像
imgAdd = color_img1.copy()
imgAdd[x:x+W2, y:y+H2] = imgROIAdd  # 用叠加图像替换背景图像中的叠加位置，得到叠加 Logo 合成图像
# 展示
plt.figure(figsize=(9,6))
titleList = ["1. imgGray", "2. imgMask", "3. MaskInv", "4. img2FG", "5. img1BG", "6. imgROIAdd"]
imageList = [img3Gray, mask, maskInv, img2Fg, img1Bg, imgROIAdd]
for i in range(6):
    plt.subplot(2,3,i+1), plt.title(titleList[i]), plt.axis('off')
    if (imageList[i].ndim==3):  # 判断彩色图像 ndim=3
        plt.imshow(cv2.cvtColor(imageList[i], cv2.COLOR_BGR2RGB))  # 彩色图像需要转换为 RGB 格式
    else:  # 灰度图像 ndim=2
        plt.imshow(imageList[i], 'gray')
plt.show()
cv2.imshow("imgAdd", imgAdd)  # 显示叠加图像 imgAdd
key = cv2.waitKey(0)  # 等待按键命令


##########################################################################
# 2.9 图像添加非中文文字
text = "OpenCV, xxayt"
fontList = [cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_PLAIN,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_TRIPLEX,
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
            cv2.FONT_ITALIC]
fontScale = 1  # 字体缩放比例
color = (255, 255, 255)  # 字体颜色
for i in range(10):
    pos = (10, 50*(i+1))
    imgPutText = cv2.putText(color_img1, text, pos, fontList[i], fontScale, color)
cv2.imshow("imgPutText", imgPutText)  # 显示叠加图像 imgAdd
key = cv2.waitKey(0)  # 等待按键命令
