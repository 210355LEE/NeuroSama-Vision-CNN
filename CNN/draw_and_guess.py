import cv2
import numpy as np
import tensorflow as tf

# 1. 載入你剛剛訓練好的模型 (假設 model 還在記憶體裡，或者我們直接現場快速定義一個)
# 為了方便你直接執行，這裡用一個簡單的方法獲取模型
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("正在快速校準 AI 視力...")
model.fit(x_train/255.0, y_train, epochs=1, verbose=0) 

# 2. 設定畫布
canvas = np.zeros((300, 300), dtype="uint8")
drawing = False

def draw(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas, (x, y), 18, (255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow("Handwriting Recognition")
cv2.setMouseCallback("Handwriting Recognition", draw)

print("\n使用說明：")
print("1. 在黑色視窗裡用滑鼠寫一個數字 (0-9)")
print("2. 按下 'p' 讓 AI 辨識")
print("3. 按下 'c' 清除畫布")
print("4. 按下 'q' 退出")

while True:
    cv2.imshow("Handwriting Recognition", canvas)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas[0:300, 0:300] = 0
    elif key == ord('p'):
        # 1. 轉成黑白二值（去除灰色雜訊）
        _, thresh = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)
        
        # 2. 自動抓取數字範圍（裁切置中）
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # 確保數字在正中央，並維持正方形比例（這對 6, 9 極其重要！）
            side = max(w, h)
            new_canvas = np.zeros((side+20, side+20), dtype="uint8")
            new_canvas[10:10+h, 10:10+w] = thresh[y:y+h, x:x+w]
            roi = cv2.resize(new_canvas, (28, 28))
        else:
            roi = cv2.resize(thresh, (28, 28))

        # 3. 形態學運算：閉合運算 (把點點連成實線)
        kernel = np.ones((2,2), np.uint8)
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

        # 4. 預測
        input_data = roi.reshape(1, 28, 28, 1).astype('float32') / 255.0
        prediction = model.predict(input_data)
        result = np.argmax(prediction)
        
        print(f"辨識結果: {result}")

cv2.destroyAllWindows()