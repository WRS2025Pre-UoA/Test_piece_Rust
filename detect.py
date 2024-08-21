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

# 画像を保存する場合
# cv2.imwrite("output_image.png", warped)

hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)  # 画像をHSV色空間に変換

# 銀色の範囲を設定 (例: 明るいグレーから暗いグレー)
lower_silver = np.array([0, 0, 150])
upper_silver = np.array([180, 50, 255])
mask_silver = cv2.inRange(hsv, lower_silver, upper_silver)
result_silver = cv2.bitwise_and(warped, warped, mask=mask_silver)

cv2.imshow("Silver Mask", result_silver)
cv2.waitKey(0)
cv2.destroyAllWindows()