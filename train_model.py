import os
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, regularizers
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import json

print("Starting training pipeline...")

source_dir = './chest_xray_data'
temp_dir = './temp_3class_split'
target_classes = ['NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']

for split in ['train', 'val', 'test']:
    for cls in target_classes:
        os.makedirs(os.path.join(temp_dir, split, cls), exist_ok=True)

all_images = {cls: [] for cls in target_classes}

for split in ['train', 'val', 'test']:
    for cls in target_classes:
        src_path = os.path.join(source_dir, split, cls)
        if os.path.exists(src_path):
            images = [os.path.join(src_path, f) for f in os.listdir(src_path)
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            all_images[cls].extend(images)

print(f"Collected {sum(len(imgs) for imgs in all_images.values())} images across 3 classes")

for cls, images in all_images.items():
    train_imgs, temp_imgs = train_test_split(images, test_size=0.30, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.50, random_state=42)
    
    for img in train_imgs:
        shutil.copy2(img, os.path.join(temp_dir, 'train', cls, os.path.basename(img)))
    for img in val_imgs:
        shutil.copy2(img, os.path.join(temp_dir, 'val', cls, os.path.basename(img)))
    for img in test_imgs:
        shutil.copy2(img, os.path.join(temp_dir, 'test', cls, os.path.basename(img)))

print("Dataset split complete: 70% train, 15% val, 15% test")

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 40

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.15,
    brightness_range=[0.85, 1.15],
    fill_mode='nearest'
)

valid_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(temp_dir, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

validation_generator = valid_test_datagen.flow_from_directory(
    os.path.join(temp_dir, 'val'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = valid_test_datagen.flow_from_directory(
    os.path.join(temp_dir, 'test'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)

class_counts = [len(os.listdir(os.path.join(temp_dir, 'train', cls))) for cls in class_names]
total = sum(class_counts)
class_weights = {i: total / (num_classes * count) for i, count in enumerate(class_counts)}

print(f"Data loaded: {train_generator.samples} train, {validation_generator.samples} val, {test_generator.samples} test")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall'),
             keras.metrics.AUC(name='auc')]
)

print(f"Model built with {model.count_params():,} parameters")

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    keras.callbacks.ModelCheckpoint('chest_xray_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
]

print("Starting training...")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

print("Training complete")

test_generator.reset()
test_loss, test_acc, test_prec, test_rec, test_auc = model.evaluate(test_generator, verbose=0)

print(f"Test Results - Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, AUC: {test_auc:.4f}")

model.save('chest_xray_model_final.keras')

with open('class_names.json', 'w') as f:
    json.dump(class_names, f)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(history.history['precision'], label='Precision')
plt.plot(history.history['recall'], label='Recall')
plt.title('Metrics')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)

shutil.rmtree(temp_dir)

print("Model saved: chest_xray_model.keras")
print("Training complete!")