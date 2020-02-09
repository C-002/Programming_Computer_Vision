import cv2
import os
import numpy as np

im = cv2.imread('../data/empire.jpg')

h, w = im.shape[:2]
#print(h, w)
cv2.imwrite('empire_result.png', im)

#读取图像
im2 = cv2.imread('../data/fisherman.jpg')
#创建灰度图像
gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
#显示图像
#cv2.imshow('fisherman', im2)
#cv2.waitKey()
#计算积分图像
intim = cv2.integral(gray)
#归一化并保存
intim = (255.0*intim) / intim.max()
cv2.imwrite('fisherman_IntimResult.jpg', intim)

h, w = im2.shape[:2]
print(h, w)
#泛洪填充
diff = (6, 6, 6)
mask = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(im2, mask, (10, 10), (255,255,0), diff, diff)
#在OpenCV窗口中显示结果
cv2.imshow('foold fill', im2)
cv2.waitKey()

cv2.imwrite('fisherman_FooldFillResult.jpg', im2)