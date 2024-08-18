import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像の読み込み
image = cv2.imread('P3_a.png', cv2.IMREAD_COLOR)

# BGRからHSVに変換
img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# さび色の範囲を定義（HSV）
rust_hues = [
    (np.array([0, 100, 100]), np.array([10, 255, 255])),   # Red-like rust
    (np.array([10, 100, 100]), np.array([25, 255, 255])),  # Orange-like rust
]

# マスクの初期化
mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)

# 各色範囲に対してマスクを作成
for hsv_min, hsv_max in rust_hues:
    # 現在の色範囲に基づいてマスクを作成
    temp_mask = cv2.inRange(img_hsv, hsv_min, hsv_max)
    
    # 現在のマスクを全体のマスクに追加
    mask = cv2.bitwise_or(mask, temp_mask)

# マスクを反転して背景を除去
mask = cv2.bitwise_not(mask)

# マスクを適用してさび以外の部分を除去
img_hsv_masked = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)

# 背景を白くする
img_hsv_masked[:, :, 2] = np.where(
    img_hsv_masked[:, :, 2] < 10,
    255 * np.ones_like(img_hsv_masked[:, :, 2]),
    img_hsv_masked[:, :, 2]
)

# HSVからBGRを経由してグレースケールに変換
img_bgr = cv2.cvtColor(img_hsv_masked, cv2.COLOR_HSV2BGR)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# ノイズ除去のためのブラー処理
img_blur = cv2.medianBlur(img_gray, 5)

# エッジ検出（Canny）
edges = cv2.Canny(img_blur, 50, 150)

# モルフォロジー変換（膨張と収縮）
kernel = np.ones((3, 3), np.uint8)
morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# 輪郭検出
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 検出された輪郭を描画
img_contours = image.copy()
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 500:  # 一定以上の面積の輪郭のみを描画
        cv2.drawContours(img_contours, [contour], -1, (0, 255, 0), 2)

# 結果の表示
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(edges, cmap='gray')
plt.title('Edges')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
plt.title('Rust Detection')
plt.axis('off')

plt.tight_layout()
plt.show()