import cv2
import numpy as np

# 画像の読み込みと前処理
image = cv2.imread('P1_a.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)

# エッジ検出
low_threshold = 0
high_threshold = 0
edges = cv2.Canny(blurred, low_threshold, high_threshold)

# 輪郭の検出
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 一番面積が大きい輪郭を見つける
max_contour = max(contours, key=cv2.contourArea)

# 面積が最大の輪郭を塗りつぶす
cv2.drawContours(image, [max_contour], -1, (0, 255, 0), thickness=cv2.FILLED)

# 結果の表示
cv2.imshow('Largest Contour', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
