import os
import glob
import cv2
import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.optimizers import Adam
from keras.models import *
from keras.layers import *
from matplotlib import pyplot as plt
import keras.backend as K

# визуализация снимков

SIZE_X = 256
SIZE_Y = 256
TRAIN_PATH = 'liver-ct-tiffs'

images = []

for directory_path in sorted(glob.glob(TRAIN_PATH + '/images/')):
    for img_path in sorted(glob.glob(os.path.join(directory_path, "*.tiff"))):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = img * 255
        img = img.astype(np.uint8)
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        img = img.astype(np.uint8)
        img = cv2.merge([img, img, img])
        images.append(img)

masks = []
for directory_path in sorted(glob.glob(TRAIN_PATH + '/masks/')):
    for mask_path in sorted(glob.glob(os.path.join(directory_path, "*.tiff"))):
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.uint8)
        masks.append(mask)

images = np.array(images)
masks = np.array(masks)

n_samples = 3
plt.figure(figsize=(10, 10))
for i in range(n_samples):
    plt.subplot(n_samples, n_samples, 1 + i)
    plt.axis('off')
    plt.imshow(images[i], cmap='gray')

for i in range(n_samples):
    plt.subplot(n_samples, n_samples, 1 + n_samples + i)
    plt.axis('off')
    plt.imshow(masks[i], cmap='gray')
plt.show()

print(f'IMAGES:\n'
      f'Unique values: {np.unique(images)}\n'
      f'Shape: {images.shape}\n'
      f'Type: {images.dtype}\n\n'
      f'MASKS:\n'
      f'Unique values: {np.unique(masks)}\n'
      f'Shape: {masks.shape}\n'
      f'Type: {masks.dtype}')

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
n, h, w = masks.shape
train_masks_reshaped = masks.reshape(-1, 1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped.ravel())
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)
train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis = 3)

print(f'Unique values of train_masks_encoded_original_shape: {np.unique(train_masks_encoded_original_shape)}\n'
        f'Shape: {train_masks_encoded_original_shape.shape}\n'
        f'Shape of train_mask_input: {train_masks_input.shape}')

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

X_train, X_val, y_train, y_val = train_test_split(images, train_masks_input,
                                                  test_size = 0.2, shuffle = True, random_state = 42)

n_classes = 2
train_masks_cat = to_categorical(y_train, num_classes = n_classes)
val_masks_cat = to_categorical(y_val, num_classes = n_classes)

print(f'TRAIN SET:\n'
     f'X_train shape: {X_train.shape}\n'
     f'y_train shape: {train_masks_cat.shape}\n\n'
     f'VALIDATION SET:\n'
     f'X_val shape: {X_val.shape}\n'
     f'y_val shape: {val_masks_cat.shape}')

# Сверточный блок
def double_conv_block(x, n_filters):
    # свертка
    x = Conv2D(n_filters, 3, padding = "same", activation = 'relu', kernel_initializer = "he_normal")(x)
    x = Conv2D(n_filters, 3, padding = "same", activation = 'relu', kernel_initializer = "he_normal")(x)
    return x

# downsample
def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    # слой подвыборки
    p = MaxPool2D(2)(f)
    return f, p

# upsample
def upsample_block(x, conv_features, n_filters):
    # транспонированная свертка
    x = Conv2DTranspose(n_filters, 3, 2, padding = "same")(x)
    # concatenation
    x = concatenate([x, conv_features])
    x = double_conv_block(x, n_filters)
    return x

# Модель

LR = 0.0001 # скорость обучения
optimizer = Adam(LR) # оптимизатор
activation = 'sigmoid' # функция активации выходного слоя
loss = 'binary_crossentropy' # функция потерь

# метрика: коэффициент Дайса-Сёренсена
from keras import backend as K
from tensorflow.keras.layers import Flatten

def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 2  # два класса: опухоль и фон
    total_dice = 0
    for i in range(class_num):
        y_true_f = tf.reshape(y_true[:,:,:,i], [-1])  # Преобразуем в одномерный вектор
        y_pred_f = tf.reshape(y_pred[:,:,:,i], [-1])  # Аналогично для предсказания
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        total_dice += dice
    return total_dice / class_num

metrics = dice_coef
img_size = (256, 256, 3) # размерность изображения

def build_unet_model(img_size, num_classes):
    # вход
    inputs = Input(shape = img_size)
    # кодировщик
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)
    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)
    # декодировщик
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)
    # выход
    outputs = Conv2D(num_classes, 1, padding = "same", activation = activation)(u9)
    unet_model = tf.keras.Model(inputs, outputs, name = "U-Net")
    return unet_model

# Модель
K.clear_session()
unet_model = build_unet_model(img_size, n_classes)

# вывод сводной информации о модели
unet_model.summary()

unet_model.compile(optimizer = optimizer,
                  loss = loss,
                  metrics = [metrics])

# обучение

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model

earlystopper = EarlyStopping(patience = 5, verbose = 1)
checkpointer = ModelCheckpoint(filepath = '/kaggle/working/checkpoint.weights.h5',
                               verbose = 1,
                               save_best_only = True, save_weights_only = True)

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.65,
                              patience = 3, min_lr = 0.000001,
                              verbose = 1,  cooldown = 1)

history = unet_model.fit(X_train, train_masks_cat, batch_size = 2, epochs = 20, verbose = 1,
                        shuffle = True, callbacks = [earlystopper, checkpointer, reduce_lr],
                        validation_data = (X_val, val_masks_cat))

# вывод графиков обучения
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'b', label = 'Training Loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
plt.title('Training and validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

dice = history.history['dice_coef']
val_dice = history.history['val_dice_coef']

plt.plot(epochs, dice, 'b', label = 'Training Dice Score')
plt.plot(epochs, val_dice, 'r', label = 'Validation Dice Score')
plt.title('Training and validation Dice Score')
plt.xlabel('Epochs')
plt.ylabel('Dice Score')
plt.legend()
plt.show()

# оценка модели на проверочных снимках
unet_model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = [metrics])
print("Evaluate on test data")
results = unet_model.evaluate(X_val, val_masks_cat, batch_size = 4)

# загрузка тестовых снимков

test_images = []

for directory_path in sorted(glob.glob(TRAIN_PATH + '/test_image/')):
    for img_path in sorted(glob.glob(os.path.join(directory_path, "*.tiff"))):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = img * 255
        img = img.astype(np.uint8)
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        img = cv2.merge([img, img, img])
        test_images.append(img)

test_images = np.array(test_images)

# визуализация снимков
n_samples = 3
plt.figure(figsize = (10, 10))
for i in range(n_samples):
    plt.subplot(n_samples, n_samples, 1 + i)
    plt.axis('off')
    plt.imshow(test_images[i], cmap = 'gray')
plt.show()

print(f'IMAGES:\n'
f'Unique values: {np.unique(test_images)}\n'
f'Shape: {test_images.shape}\n'
f'Type: {test_images.dtype}\n')

# сегментация тестовых снимков
from random import randint

test_img_number = randint(0, len(test_images) - 1)
test_img = test_images[test_img_number]
test_img_input = np.expand_dims(test_img, 0)
test_pred1 = unet_model.predict(test_img_input)
test_prediction1 = np.argmax(test_pred1, axis = 3)[0, :, :]

fig, (ax1, ax2) = plt.subplots(ncols = 2)
ax1.set_title('Testing Image')
ax1.axis('off')
ax1.imshow(test_img[:, :, 0], cmap = 'gray')
ax2.set_title('Prediction on test image')
ax2.axis('off')
ax2.imshow(test_prediction1, cmap = 'gray')
plt.show()

# RLE-кодировка / декодировка для submission

def encode_mask_to_rle(mask):
    '''
    mask: бинарная масска в виде numpy массива
    1 - печень
    0 - фон
    Возвращает закодированную длину серии
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def decode_rle_to_mask(rle, height, width, viz = False):
    '''
    rle : длина серии в строковом формате (начальное значение, число элементов)
    height : высота изображения маски
    width : ширина изображения маски
    Возвращает бинарную маску
    '''
    rle = np.array(rle.split(' ')).reshape(-1, 2)
    mask = np.zeros((height * width, 1, 3))
    if viz:
        color = np.random.rand(3)
    else:
        color = [1, 1, 1]
    for i in rle:
        mask[int(i[0]):int(i[0]) + int(i[1]), :, :] = color

    return cv2.cvtColor(mask.reshape(height, width, 3).astype(np.uint8), cv2.COLOR_BGR2GRAY)

import pandas as pd

prediction = []
for i in range(len(test_images)):
    test_img = test_images[i]
    test_img_input = np.expand_dims(test_img, 0)
    test_pred1 = unet_model.predict(test_img_input)
    test_prediction1 = np.argmax(test_pred1, axis = 3)[0, :, :]
    prediction.append(test_prediction1)

rle_pred = []
for i in range(len(prediction)):
    encoded = encode_mask_to_rle(prediction[i])
    rle_pred.append(encoded)

pred = pd.DataFrame(data = {'id':range(len(rle_pred)),
                          'target':rle_pred})

# сохранение модели
unet_model.save('liver_tumor_seg.h5')
# сохранение файла с результатами
pred.to_csv('submission.csv', index = False)


