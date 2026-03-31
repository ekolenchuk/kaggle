import os
import glob
import cv2
import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np

# Импортируем Keras напрямую, а не через TensorFlow
import keras
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import (
    Input, Conv2D, MaxPooling2D, Conv2DTranspose, 
    concatenate, BatchNormalization, Dropout, Activation
)
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from matplotlib import pyplot as plt
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import pandas as pd

# Проверка версий
print(f"Keras version: {keras.__version__}")

# ============================================
# 1. КОНФИГУРАЦИЯ
# ============================================
CONFIG = {
    "SEED": 42,
    "EPOCHS": 20,
    "BATCH_SIZE": 8,
    "LR": 0.0003,
    "IMG_SIZE": 256,
    "TRAIN_PATH": 'liver-ct-tiffs',
    "N_CLASSES": 2,
    "EARLY_STOPPING_PATIENCE": 5,
    "REDUCE_LR_PATIENCE": 3,
    "REDUCE_LR_FACTOR": 0.5
}

# Установка seed для воспроизводимости
np.random.seed(CONFIG["SEED"])
# tf.random.set_seed(CONFIG["SEED"])  # Закомментировано, так как используем Keras

# ============================================
# 2. ЗАГРУЗКА ДАННЫХ
# ============================================
SIZE_X = CONFIG["IMG_SIZE"]
SIZE_Y = CONFIG["IMG_SIZE"]
TRAIN_PATH = CONFIG["TRAIN_PATH"]

def load_and_preprocess_image(img_path, is_mask=False):
    """Улучшенная загрузка с нормализацией"""
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"Warning: Could not read {img_path}")
        if is_mask:
            return np.zeros((SIZE_Y, SIZE_X), dtype=np.uint8)
        else:
            return np.zeros((SIZE_Y, SIZE_X, 3), dtype=np.uint8)
    
    # Нормализация
    img = img.astype('float32')
    img = img - img.min()
    img = img / (img.max() + 1e-6)
    img = (img * 255).astype('uint8')
    
    if not is_mask:
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
    else:
        img = cv2.resize(img, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.uint8)
    
    return img

# Загрузка данных
print("Загрузка данных...")
images = []
for directory_path in sorted(glob.glob(TRAIN_PATH + '/images/')):
    for img_path in sorted(glob.glob(os.path.join(directory_path, "*.tiff"))):
        img = load_and_preprocess_image(img_path, is_mask=False)
        images.append(img)

masks = []
for directory_path in sorted(glob.glob(TRAIN_PATH + '/masks/')):
    for mask_path in sorted(glob.glob(os.path.join(directory_path, "*.tiff"))):
        mask = load_and_preprocess_image(mask_path, is_mask=True)
        masks.append(mask)

images = np.array(images)
masks = np.array(masks)

print(f'IMAGES shape: {images.shape}, dtype: {images.dtype}')
print(f'MASKS shape: {masks.shape}, dtype: {masks.dtype}')
print(f'Unique mask values: {np.unique(masks)}')

# ============================================
# 3. ВИЗУАЛИЗАЦИЯ
# ============================================
n_samples = 3
plt.figure(figsize=(10, 10))
for i in range(n_samples):
    plt.subplot(2, n_samples, i + 1)
    plt.axis('off')
    plt.imshow(images[i], cmap='gray')
    plt.title(f'Image {i+1}')

for i in range(n_samples):
    plt.subplot(2, n_samples, n_samples + i + 1)
    plt.axis('off')
    plt.imshow(masks[i], cmap='gray')
    plt.title(f'Mask {i+1}')
plt.suptitle('Original Images and Masks')
plt.show()

# ============================================
# 4. ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ
# ============================================
labelencoder = LabelEncoder()
n, h, w = masks.shape
train_masks_reshaped = masks.reshape(-1, 1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped.ravel())
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)
train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

# Разделение данных
X_train, X_val, y_train, y_val = train_test_split(
    images, train_masks_input,
    test_size=0.15,
    shuffle=True, 
    random_state=CONFIG["SEED"]
)

# One-hot encoding
train_masks_cat = to_categorical(y_train, num_classes=CONFIG["N_CLASSES"])
val_masks_cat = to_categorical(y_val, num_classes=CONFIG["N_CLASSES"])

print(f'TRAIN SET: X_train shape: {X_train.shape}, y_train shape: {train_masks_cat.shape}')
print(f'VALIDATION SET: X_val shape: {X_val.shape}, y_val shape: {val_masks_cat.shape}')

# ============================================
# 5. АРХИТЕКТУРА U-NET
# ============================================
def double_conv_block(x, n_filters, dropout_rate=0.1):
    x = Conv2D(n_filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    x = Conv2D(n_filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def downsample_block(x, n_filters, dropout_rate=0.1):
    f = double_conv_block(x, n_filters, dropout_rate)
    p = MaxPooling2D(2)(f)
    return f, p

def upsample_block(x, conv_features, n_filters, dropout_rate=0.1):
    x = Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = concatenate([x, conv_features])
    x = double_conv_block(x, n_filters, dropout_rate)
    return x

def build_improved_unet(img_size, num_classes):
    inputs = Input(shape=img_size)
    
    # Encoder
    f1, p1 = downsample_block(inputs, 64, dropout_rate=0.1)
    f2, p2 = downsample_block(p1, 128, dropout_rate=0.1)
    f3, p3 = downsample_block(p2, 256, dropout_rate=0.2)
    f4, p4 = downsample_block(p3, 512, dropout_rate=0.2)
    
    # Bottleneck
    bottleneck = double_conv_block(p4, 1024, dropout_rate=0.3)
    
    # Decoder
    u6 = upsample_block(bottleneck, f4, 512, dropout_rate=0.2)
    u7 = upsample_block(u6, f3, 256, dropout_rate=0.2)
    u8 = upsample_block(u7, f2, 128, dropout_rate=0.1)
    u9 = upsample_block(u8, f1, 64, dropout_rate=0.1)
    
    # Output
    outputs = Conv2D(num_classes, 1, padding="same", activation='softmax')(u9)
    
    model = Model(inputs, outputs, name="Improved_U-Net")
    return model

# ============================================
# 6. ФУНКЦИЯ ПОТЕРЬ
# ============================================
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1 - dice

def combined_loss(y_true, y_pred):
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.reshape(y_true[:,:,:,1], [-1])
    y_pred_f = K.reshape(y_pred[:,:,:,1], [-1])
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

# ============================================
# 7. СОЗДАНИЕ И КОМПИЛЯЦИЯ МОДЕЛИ
# ============================================
K.clear_session()
improved_model = build_improved_unet((SIZE_X, SIZE_Y, 3), CONFIG["N_CLASSES"])
improved_model.summary()

improved_model.compile(
    optimizer=Adam(learning_rate=CONFIG["LR"]),
    loss=combined_loss,
    metrics=[dice_coef]
)

# ============================================
# 8. CALLBACKS
# ============================================
earlystopper = EarlyStopping(
    monitor='val_loss',
    patience=CONFIG["EARLY_STOPPING_PATIENCE"],
    verbose=1,
    restore_best_weights=True
)

checkpointer = ModelCheckpoint(
    filepath='best_model.h5',
    verbose=1,
    save_best_only=True,
    save_weights_only=False
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=CONFIG["REDUCE_LR_FACTOR"],
    patience=CONFIG["REDUCE_LR_PATIENCE"],
    min_lr=1e-7,
    verbose=1
)

callbacks = [earlystopper, checkpointer, reduce_lr]

# ============================================
# 9. ОБУЧЕНИЕ
# ============================================
print("\n" + "="*60)
print(f"НАЧАЛО ОБУЧЕНИЯ НА {CONFIG['EPOCHS']} ЭПОХ")
print("="*60)

history = improved_model.fit(
    X_train, train_masks_cat,
    batch_size=CONFIG["BATCH_SIZE"],
    epochs=CONFIG["EPOCHS"],
    verbose=1,
    shuffle=True,
    callbacks=callbacks,
    validation_data=(X_val, val_masks_cat)
)

# ============================================
# 10. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# ============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
ax1.plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['dice_coef'], 'b-', label='Training Dice', linewidth=2)
ax2.plot(history.history['val_dice_coef'], 'r-', label='Validation Dice', linewidth=2)
ax2.set_title('Dice Coefficient', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Dice Score')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# 11. ВЫВОД РЕЗУЛЬТАТОВ
# ============================================
print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
print("="*60)
print(f"Лучшая Dice Score на обучении: {max(history.history['dice_coef']):.4f}")
print(f"Лучшая Dice Score на валидации: {max(history.history['val_dice_coef']):.4f}")

# Сохранение модели
improved_model.save('improved_liver_segmentation.h5')
print("✅ Модель сохранена как 'improved_liver_segmentation.h5'")

print("\n" + "="*60)
print("ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
print("="*60)
