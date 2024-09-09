import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Detect_png.png",cv2.IMREAD_GRAYSCALE)
dst_cv = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 1001, 50)
                                                    # 軸目盛、軸ラベルを消す
                                                  # 1行3列の3番目(右)の領域にプロットを設定
cv2.imshow("image",dst_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("image.png",dst_cv)

black_area = np.sum(dst_cv == 0)
white = np.sum(dst_cv == 255)
size = black_area + white
print(size)
print(black_area, white )
area = black_area / size * 100
print(area)