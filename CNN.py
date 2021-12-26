
# setup
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras.utils import to_categorical

# prepare the data
dataset = scipy.io.loadmat('/Users/fuyalun/Documents/EE5907-PatternRecognize/CA2/CA2-FuYalun/facedata.mat')
train_data = np.transpose(np.array(dataset[ "train_data" ]))
test_data = np.transpose(np.array(dataset[ "test_data" ]))

train_label = np.zeros((1, 119))
train_self = 25 * np.ones((1,7))
test_label = np.zeros((1, 51))
test_self = 25 * np.ones((1,3))

for i in range (1, 25):
  label = i * np.ones((1, 119))
  train_label = np.hstack((train_label, label))
train_label = np.hstack((train_label, train_self))

for j in range (1, 25):
  label = j * np.ones((1, 51))
  test_label = np.hstack((test_label, label))
test_label = np.hstack((test_label, test_self))

x_train = train_data.reshape(2982, 32, 32, 1)
# y_train = np.transpose(train_label)
x_test = test_data.reshape(1278, 32, 32, 1)
# y_test = np.transpose(test_label)
# print(y_train.shape)

y_train = to_categorical(train_label, num_classes = 26)
y_train = y_train.reshape((2982, 26))
y_test = to_categorical(test_label, num_classes = 26)
y_test = y_test.reshape((1278, 26))

# build model
model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 1)),
        layers.Conv2D(20, kernel_size=(5, 5), strides = 1, padding = 'same'),
        layers.MaxPooling2D(pool_size=(2, 2), strides = 2),
        layers.Conv2D(50, kernel_size=(5, 5), strides = 1, padding = 'same'),
        layers.MaxPooling2D(pool_size=(2, 2), strides = 2),
        layers.Flatten(), # 1250
        layers.Dense(500, activation="relu"),
        layers.Dense(26, activation="softmax"),
    ]
)

model.summary()

# train model
batch = 32
epochs = 20
#adam = keras.optimizers.Adam(learning_rate = 1e-4)
model.compile(loss="categorical_crossentropy", optimizer= "adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size = batch, epochs=epochs)

# test model
test_score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", "%.4f"%test_score[0])
print("Test accuracy:" + str("%.2f"%(test_score[1]*100)) + "%")

train_score = model.evaluate(x_train, y_train, verbose=0)
print("Train loss:", "%.4f"%train_score[0])
print("Train accuracy:" + str("%.2f"%(train_score[1]*100)) + "%")