import numpy as np
import cv2
import matplotlib.pyplot as plt


#############################################################

def hue_histogram(image, mask):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv_image[:, :, 0]
    masked_hue = hue[mask]  
    hist, _ = np.histogram(masked_hue, bins=180, range=(0, 180))
    HUEHIST = hist.astype(np.uint32).reshape((180, 1))

    return HUEHIST

#############################################################

height, width = 480, 640

# 空の画像を作成（3チャンネル、uint8型）
image = np.zeros((height, width, 3), dtype=np.uint8)

# 矩形領域の定義
half_h, half_w = height // 2, width // 2
# 上半分：赤（BGR = (0, 0, 255)）
image[0:half_h, 0:width] = (0, 0, 255)
# 左下：青（BGR = (255, 0, 0)）
image[half_h:height, 0:half_w] = (255, 0, 0)
# 右下：緑（BGR = (0, 255, 0)）
image[half_h:height, half_w:width] = (0, 255, 0)

# 480行×640列のゼロ配列を作成
mask = np.zeros((480, 640), dtype=bool)
# 内側の領域をtrueにする
mask[120:360, 160:400] = True

###########################################################
# #マスク内のBGRもってくる

# #　mask配列で1の部分のピクセル内のRGB
# masked_pixels = image[mask]

# # チャンネルごとに分ける
# B_channel = masked_pixels[:, 0]
# G_channel = masked_pixels[:, 1]
# R_channel = masked_pixels[:, 2]

# R_hist = np.histogram(R_channel, bins=256, range=(0, 256))[0]
# G_hist = np.histogram(G_channel, bins=256, range=(0, 256))[0]
# B_hist = np.histogram(B_channel, bins=256, range=(0, 256))[0]

# plt.plot(range(256), R_hist, color='red', label='Red channel')
# plt.plot(range(256), G_hist, color='green', label='Green channel')
# plt.plot(range(256), B_hist, color='blue', label='Blue channel')
###########################################################

#色相ヒストグラム作成する関数を使用
huehist = hue_histogram(image,mask)

# plt.hist(masked_hue, bins=180, range=(0, 180), color='purple')