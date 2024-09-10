import cv2
from matplotlib import pyplot as plt
import numpy as np

# クリックした点を格納するリスト
# points_list = []

def resize_func(img):
    width = 1280
    h,w = img.shape[:2]

    aspect_ratio= h / w
    new_height = int(width * aspect_ratio)

    # resized_img = cv2.resize(img, (width,new_height),interpolation=cv2.INTER_AREA)
    resized_img = cv2.resize(img, (1280,720),interpolation=cv2.INTER_AREA)
    return resized_img

# 2点間の距離を計算する関数
def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# マウスイベント処理関数
def mouseEvents(event, x, y, flags, points_list):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        points_list.append([x, y])

def adapt(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dst_cv = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,1001,20)
    cv2.imshow("image",dst_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # cv2.imwrite("image.png",dst_cv)

    black_area = np.sum(dst_cv == 0)
    white = np.sum(dst_cv == 255)
    size = black_area + white
    # print(size)
    # print(black_area, white )
    area = black_area / size * 100
    # print(area)
    return area

def extract_test_piece(img,points_list):
    # 画像を表示してマウスイベントを設定
    cv2.imshow("Select the 4 points", img)
    cv2.setMouseCallback("Select the 4 points", mouseEvents,points_list)
    # 十分なクリックが行われるまで待機
    while len(points_list) < 4:
        cv2.waitKey(1)  # 小さな待機時間で処理を継続する

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
    return warped

def main():
    # 画像を読み込み
    # img = cv2.imread("../PNG_E/Pattern_1/1.5F_Pole/az-30.png")  # 画像のパスを指定
    # img = resize_func(img)
    # points_list = []
    # new_img=extract_test_piece(img,points_list)

    # # 結果を表示
    # cv2.imshow("Warped Image", new_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imwrite("Detect_png.png",new_img)
    new_img = cv2.imread("Detect_png.png")
    print(adapt(new_img))

if __name__ == '__main__':
    main()
