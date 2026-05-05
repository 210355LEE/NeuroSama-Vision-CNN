import cv2
import numpy as np
import tensorflow as tf

# 1. 直接載入你剛剛訓練好的強大模型
try:
    model = tf.keras.models.load_model('best_model.h5')
    print("成功載入 98.69% 準確率模型！")
except:
    print("找不到 best_model.h5，請先執行 vision_logic.py 並存檔。")

canvas = np.zeros((300, 300), dtype="uint8")
drawing = False

def draw(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN: drawing = True
    elif event == cv2.EVENT_LBUTTONUP: drawing = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing: cv2.circle(canvas, (x, y), 12, (255), -1) # 筆跡適中

cv2.namedWindow("AI_Test")
cv2.setMouseCallback("AI_Test", draw)

while True:
    cv2.imshow("AI_Test", canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('c'): canvas.fill(0)
    elif key == ord('p'):
        # --- 核心預處理：對齊 MNIST 規格 ---
        # 1. 調整大小
        roi = cv2.resize(canvas, (28, 28))
        # 2. 數值歸一化 (0-1)，對應你 vision_logic 第 9 行的處理
        roi = roi.astype('float32') / 255.0
        # 3. 增加維度符合 Input(28, 28, 1) 的要求
        final_input = np.expand_dims(roi, axis=(0, -1)) 
        
        # 4. 預測
        prediction = model.predict(final_input)
        result = np.argmax(prediction)
        confidence = np.max(prediction)
        
        print(f"\nAI 辨識結果: {result} (信心度: {confidence:.2%})")

cv2.destroyAllWindows()