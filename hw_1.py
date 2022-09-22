import numpy as np
import matplotlib.pyplot as plt
import cv2

class LSB_Embed():
    def __init__(self):
        pass

    @staticmethod
    def get_bitPlane(img):
        ''' 
        獲取灰階圖像的八個位元平面
        :param img: 灰階圖像
        :return: 八個位元平面的矩陣
        '''
        h, w = img.shape

        flag = 0b00000001
        bitPlane = np.zeros(shape=(h, w, 8))
        for i in range(bitPlane.shape[-1]):
            bitplane = img & flag # 獲取圖像的某一位元，從最後一位開始處理
            bitplane[bitplane != 0] = 1 # 臨界點處理，非 0 即 1
            bitPlane[..., i] = bitplane # 處理後的數據載入到某個位元平面
            flag <<= 1 # 獲取下一個位元的訊息
        return bitPlane.astype(np.uint8)

    @staticmethod
    def lsb_embed(background, watermark, embed_bit=3):
        ''' 
        在 bg 的最後三位元進行嵌入水印，具體為將 wm 的高三位元訊息替換掉 bg 的低三位元訊息
        :param background: 灰階背景圖像
        :param watermark: 灰階水印圖像
        :return: 嵌入水印的圖像
        '''
        # 1. 判斷是否滿足可嵌入的條件
        w_h, w_w = watermark.shape
        b_h, b_w = background.shape
        assert w_w < b_w and w_h < b_h, "請保證 wm 尺寸小於 bg 尺寸 \r\n Current size watermark:{}, background:{}".format(watermark.shape, background.shape)
        # 2. 獲取位元平面
        bitPlane_background = lsb.get_bitPlane(background) # 獲取的平面順序是從低位到高位的 0 1 2 3 4 5 6 7
        bitPlane_watermark = lsb.get_bitPlane(watermark)
        # 3. 在位元平面嵌入訊息
        for i in range(embed_bit):
            # 訊息主要集中在高位元，此外將 wm 的高三位元訊息放置在 bg 低三位元的訊息中
            bitPlane_background[0:w_h, 0:w_w, i] = bitPlane_watermark[0:w_h, 0:w_w, (8 - embed_bit) + i]
        # 4. 得到 watermark_img 水印嵌入的圖像
        synthesis = np.zeros_like(background)
        for i in range(8):
            synthesis += bitPlane_background[..., i] * np.power(2, i)
        return synthesis.astype(np.uint8)
    
    @staticmethod
    def lsb_extract(synthesis, embed_bit): # embed_bit=3
        bitPlane_synthesis = lsb.get_bitPlane(synthesis)
        extract_watermark = np.zeros_like(synthesis)
        extract_background = np.zeros_like(synthesis)
        for i in range(8):
            if i < embed_bit:
                extract_watermark += bitPlane_synthesis[..., i] * np.power(2, (8 - embed_bit) + i)
            else:
                extract_background += bitPlane_synthesis[..., i] * np.power(2, i)
        return extract_watermark.astype(np.uint8), extract_background.astype(np.uint8)

if __name__ == '__main__': 
    root = '..'
    lsb = LSB_Embed()
    
    bg = cv2.imread('./doris_tsai_greyscale.jpg', cv2.IMREAD_GRAYSCALE) # 1-1. 獲取背景
    wm = cv2.imread('./sticker_greyscale.jpg', cv2.IMREAD_GRAYSCALE) # 1-2. 獲取水印
    
    bg_backup = bg.copy() # 1-3. 獲取背景備份
    wm_backup = wm.copy() # 1-4. 獲取水印備份

    # 2. 進行水印嵌入
    embed_bit = 3 # 1 / 2 / 3 
    synthesis = lsb.lsb_embed(bg, wm, embed_bit)

    # 3. 進行水印提取
    extract_wm, extract_bg = lsb.lsb_extract(synthesis, embed_bit)
    imgs = [bg_backup, wm_backup, synthesis, extract_wm, extract_bg]
    title = ["background", "watermark", "synthesis", "extract_watermark", "extract_background"]

    for i in range(len(imgs)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(imgs[i], cmap="gray")
        plt.axis("off")
        plt.title(title[i])
    plt.show()