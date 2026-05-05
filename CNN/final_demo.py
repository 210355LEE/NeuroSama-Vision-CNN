import cv2
import numpy as np
import tensorflow as tf

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
        if drawing: cv2.circle(canvas, (x, y), 12, (255), -1) 

cv2.namedWindow("AI_Test")
cv2.setMouseCallback("AI_Test", draw)

while True:
    cv2.imshow("AI_Test", canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('c'): canvas.fill(0)
    elif key == ord('p'):
        roi = cv2.resize(canvas, (28, 28)
        roi = roi.astype('float32') / 255.0
        final_input = np.expand_dims(roi, axis=(0, -1)) 
        prediction = model.predict(final_input)
        result = np.argmax(prediction)
        confidence = np.max(prediction)
        print(f"\nAI 辨識結果: {result} (信心度: {confidence:.2%})")

cv2.destroyAllWindows()
