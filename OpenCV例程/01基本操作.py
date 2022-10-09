import cv2
import numpy as np
import random
import matplotlib.pyplot as plt 

path = "D:/2Codefield/VS_code/python/Learn_Base/openCV/DATA/"
lenasrc = "Lena/"
baboonsrc = "Baboon/"

##########################################################################
# 1.1 图像的读取
color_img = cv2.imread(path + lenasrc + "color.png", flags=1)  # flags=1 读取彩色图像(BGR)
gray_img = cv2.imread(path + lenasrc + "color.png", flags=0)  # flags=0 读取为灰度图像
color_img2 = cv2.imread(path + baboonsrc + "color.png")  # 读取彩色图像(BGR)


##########################################################################
# 1.2 图像的保存
cv2.imwrite(path + lenasrc + "gray2.png", gray_img)  # 保存图像文件


##########################################################################
# 1.3 图像的显示(cv2.imshow)
cv2.imshow("Demo1", color_img)  # 在窗口 "Demo1" 显示图像 color_img
cv2.imshow("Demo2", gray_img)  # 在窗口 "Demo2" 显示图像 gray_img
cv2.waitKey(0)  # 显示时长为无限


##########################################################################
# 1.4 图像显示(plt.imshow)
color_imgRGB = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)  # 图片格式转换：BGR(OpenCV) -> RGB(PyQt5)
gray_imgRGB = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)  # 图片格式转换：BGR(OpenCV) -> Gray
# 用plt显示
plt.rcParams['font.sans-serif'] = ['FangSong']  # 支持中文标签
plt.subplot(221), plt.title("1. RGB 格式(mpl)"), plt.axis('off')
plt.imshow(color_imgRGB)  # matplotlib 显示彩色图像(RGB格式)
plt.subplot(222), plt.title("2. BGR 格式(OpenCV)"), plt.axis('off')
plt.imshow(color_img)    # matplotlib 显示彩色图像(BGR格式)
plt.subplot(223), plt.title("3. 设置 Gray 参数"), plt.axis('off')
plt.imshow(gray_imgRGB, cmap='gray')  # matplotlib 显示灰度图像，设置 Gray 参数
plt.subplot(224), plt.title("4. 未设置 Gray 参数"), plt.axis('off')
plt.imshow(gray_imgRGB)  # matplotlib 显示灰度图像，未设置 Gray 参数
plt.show()


##########################################################################
# 1.5 图像的裁剪
xmin, ymin, w, h = 180, 190, 200, 200  # 矩形裁剪区域 (ymin:ymin+h, xmin:xmin+w) 的位置参数
imgCrop = color_img[ymin:ymin+h, xmin:xmin+w].copy()  # 切片获得裁剪后保留的图像区域
cv2.imshow("DemoCrop", imgCrop)  # 在窗口显示 彩色随机图像
cv2.waitKey(0)


##########################################################################
# 1.6 图像的鼠标手动裁剪 (ROI)
roi = cv2.selectROI(color_img, showCrosshair=True, fromCenter=False)
xmin, ymin, w, h = roi  # 矩形裁剪区域 (ymin:ymin+h, xmin:xmin+w) 的位置参数
imgROI = color_img[ymin:ymin+h, xmin:xmin+w].copy()  # 切片获得裁剪后保留的图像区域
cv2.imshow("DemoRIO", imgROI)
cv2.waitKey(0)


##########################################################################
# 1.7 图像缩放后拼接
color_img_sz = cv2.resize(color_img, (400, 400))  # 缩放到(横, 高)
color_img2_sz = cv2.resize(color_img2, (300, 400))
img3_sz = cv2.resize(color_img2_sz, (400, 300))
imgStackH = np.hstack((color_img_sz, color_img2_sz))  # 高度相同图像可以横向水平拼接
imgStackV = np.vstack((color_img_sz, img3_sz))  # 宽度相同图像可以纵向垂直拼接
# 打印
print("Horizontal stack:\nShape of color_img, color_img2 and imgStackH: ", color_img_sz.shape, color_img2_sz.shape, imgStackH.shape)
print("Vertical stack:\nShape of color_img, img3 and imgStackV: ", color_img_sz.shape, img3_sz.shape, imgStackV.shape)
cv2.imshow("DemoStackH", imgStackH)  # 在窗口显示图像 imgStackH
cv2.imshow("DemoStackV", imgStackV)  # 在窗口显示图像 imgStackV
cv2.waitKey(0)  # 等待按键命令


##########################################################################
# 1.8 图像拆分通道
bImg, gImg, rImg = cv2.split(color_img)  # 拆分为 BGR 独立通道
cv2.imshow("rImg", rImg)  # 直接显示红色分量 rImg 显示为灰度图像
# 将单通道扩展为三通道
imgZeros = np.zeros_like(color_img)  # 创建与 color_img 相同形状的黑色图像
imgZeros[:, :, 2] = rImg  # 在黑色图像模板添加红色分量 rImg
cv2.imshow("channel R", imgZeros)  # 扩展为 BGR 通道
print(color_img.shape, rImg.shape, imgZeros.shape)
cv2.waitKey(0)


##########################################################################
# 1.9 图像通道的合并
bImg, gImg, rImg = cv2.split(color_img)  # 拆分为 BGR 独立通道
# cv2.merge 实现图像通道的合并
imgMerge = cv2.merge([bImg, gImg, rImg])
cv2.imshow("cv2Merge", imgMerge)
# Numpy 拼接实现图像通道的合并
imgStack = np.stack((bImg, gImg, rImg), axis=2)
cv2.imshow("npStack", imgStack)
# 两种方法结果相同
print(imgMerge.shape, imgStack.shape)
print("imgMerge is imgStack?", np.array_equal(imgMerge, imgStack))
cv2.waitKey(0)
