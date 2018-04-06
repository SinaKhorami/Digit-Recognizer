'''
  Mnist Digit Recognizer
  Author: Sina
'''

import numpy  as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta

#### Globals ####
num_classes = 10
num_epoches = 12
batch_size = 128
img_width = 28
img_height = 28

#### Data Loader ####
train = pd.read_csv("Data/train.csv")
X_test = pd.read_csv("Data/test.csv").values
y_train = train["label"].values
X_train = train.drop(labels=["label"], axis=1).values

#### PreProcessing ####
X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(-1, img_width, img_height, 1)
X_test = X_test.reshape(-1, img_width, img_height, 1)

y_train = to_categorical(y_train, num_classes=num_classes)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=92)

#### Model ####
model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3),
                 activation='relu',
                 input_shape=(img_width, img_height, 1)))
model.add(Conv2D(64, kernel_size=(3,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=categorical_crossentropy,
              optimizer=Adadelta(),
              metrics=['accuracy'])

#### Train ####
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=num_epoches,
          validation_data=(X_val, y_val))

#### Evaluation ####
score = model.evaluate(X_val, y_val)
print score