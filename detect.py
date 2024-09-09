import cv2
from matplotlib import pyplot as plt
import numpy as np

# クリックした点を格納するリスト
points_list = []

# 2点間の距離を計算する関数
def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# マウスイベント処理関数
def mouseEvents(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        points_list.append([x, y])

# 画像を読み込み
img = cv2.imread("./PNG_E/Pattern_1/1.5F_Pole/az30.png")  # 画像のパスを指定

# 画像を表示してマウスイベントを設定
cv2.imshow("Select the 4 points", img)
cv2.setMouseCallback("Select the 4 points", mouseEvents)
cv2.waitKey(0)
cv2.destroyAllWindows()

# クリックした4つの点を取得
points = np.array(points_list, dtype="float32")
print("Selected points:", points)

# 最も長い辺の長さを計算
lengths = [dist(points[i], points[(i+1) % 4]) for i in range(4)]
max_length = int(max(lengths))

# 正方形のターゲット座標を設定
square = np.array([[0, 0], [max_length, 0], 
                   [max_length, max_length], [0, max_length]], dtype="float32")

# 射影変換行列を計算
M = cv2.getPerspectiveTransform(points, square)

# 変換後の画像サイズ
output_size = (max_length, max_length)

# 射影変換を適用
warped = cv2.warpPerspective(img, M, output_size)

# 結果を表示
cv2.imshow("Warped Image", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("Detect_png.png",warped)

img1 = cv2.imread("Detect_png.png",cv2.IMREAD_GRAYSCALE)
dst_cv = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 1001, 50)
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