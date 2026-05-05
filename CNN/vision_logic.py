import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# 1. 載入數據集（這就像是給 AI 看 6 萬張練習題）
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# 2. 歸一化（將像素 0-255 轉為 0-1，這是電機訊號處理的常見手法，能加快計算）
train_images, test_images = train_images / 255.0, test_images / 255.0

# 3. 建立 CNN 模型（卷積神經網路）
model = models.Sequential([
    # 第一層卷積：提取特徵（例如線條、邊緣）
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)), # 池化：壓縮數據，保留精華
    
    # 第二層卷積：提取更複雜的特徵
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # 全連接層：判斷這到底是什麼數字
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') # 輸出 0~9 的機率
])

# 4. 編譯模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. 開始訓練（訓練 3 次就好，免得筆電跑太久）
print("AI 正在學習辨識特徵中...")
model.fit(train_images, train_labels, epochs=3)

# 6. 測試 AI 的準確率
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f'\n這台 AI 的視力（準確率）達到: {test_acc*100:.2f}%')

# 訓練完後存檔
model.save('best_model.h5')
print("--- 腦袋已存檔：best_model.h5 ---")