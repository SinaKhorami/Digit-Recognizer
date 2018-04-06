
#### Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Dropout, Flatten
from keras.optimizers import RMSprop
from keras.losses import categorical_crossentropy
from keras.callbacks import ReduceLROnPlateau

#### Load Data
train = pd.read_csv("Data/train.csv").values
test = pd.read_csv("Data/test.csv").values

X_train = train[:, 1:].astype('float32')
Y_train = train[:, 0].astype('int32')
X_test = test[:, :].astype('float32')

#### Pre Processing
X_train = X_train/255.
X_test = X_test/255.

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

Y_train = to_categorical(Y_train, num_classes=10)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15, random_state=23)

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15)
datagen.fit(X_train)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=2)

#### Model
model = Sequential()

model.add(Conv2D(16, kernel_size=(5, 5),
                activation='relu',
                input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=(3, 3),
                padding='same',
                activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3),
                padding='same',
                activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3),
                padding='same',
                activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss=categorical_crossentropy,
              optimizer=RMSprop(lr=1e-3),
              metrics=['accuracy'])

#### Train
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=70),
                    epochs=20,
                    validation_data=(X_val, Y_val),
                    callbacks=[reduce_lr])

model.save_weights("model")

Y_val_pred = model.predict_classes(X_val)
Y_val_true = np.argmax(Y_val, axis=1)

print(confusion_matrix(Y_val_true, Y_val_pred))

Y_test_pred = model.predict_classes(X_test)

submission = pd.DataFrame({ 'ImageId': range(1, 28001), 'Label': Y_test_pred })
submission.to_csv("submission.csv", index=False)